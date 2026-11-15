// vault.js -- the pure logic of the second-brain plugin: a folder of markdown notes linked by [[wikilinks]].
// No Cordis, no DSH imports: this file is unit-tested with `node --test` against a temp directory.
import { mkdir, readdir, readFile, rename, writeFile } from 'node:fs/promises'
import path from 'node:path'

export const MAX_CONTENT_CHARS = 100_000
export const MAX_NOTES = 2_000
export const MAX_TITLE_CHARS = 120
const WIKILINK = /\[\[([^\[\]\n|]{1,200})(?:\|[^\[\]\n]*)?\]\]/g

export class VaultError extends Error {
  constructor(message) {
    super(message)
    this.name = 'VaultError'
  }
}

/** "Project X: Ideas!" -> "project-x-ideas". The slug is the file name, so it must never contain a path separator. */
export function slugify(title) {
  const slug = String(title ?? '')
    .normalize('NFKD').replace(/[̀-ͯ]/g, '')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80).replace(/-+$/g, '')
  if (!slug) throw new VaultError(`title "${title}" has no usable letters or digits`)
  return slug
}

/** Titles referenced as [[Title]] or [[Title|alias]], de-duplicated by slug, in first-seen order. */
export function extractLinks(content) {
  const seen = new Map()
  for (const match of String(content).matchAll(WIKILINK)) {
    const title = match[1].trim()
    try {
      const slug = slugify(title)
      if (!seen.has(slug)) seen.set(slug, title)
    } catch { /* an unusable link target such as [[???]] is ignored */ }
  }
  return [...seen].map(([slug, title]) => ({ slug, title }))
}

export function serialize({ title, tags, updated, body }) {
  return `---\ntitle: ${JSON.stringify(title)}\ntags: ${JSON.stringify(tags)}\nupdated: ${updated}\n---\n${body.replace(/\s+$/, '')}\n`
}

export function parse(raw, fallbackSlug) {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(raw)
  const note = { title: fallbackSlug, tags: [], updated: null, body: raw }
  if (!match) return note
  note.body = match[2]
  for (const line of match[1].split('\n')) {
    const at = line.indexOf(':')
    if (at < 0) continue
    const key = line.slice(0, at).trim()
    const value = line.slice(at + 1).trim()
    try {
      if (key === 'title') note.title = JSON.parse(value)
      else if (key === 'tags') note.tags = JSON.parse(value)
      else if (key === 'updated') note.updated = value
    } catch { /* a hand-edited, malformed header line is skipped, the note stays readable */ }
  }
  if (!Array.isArray(note.tags)) note.tags = []
  return note
}

export class Vault {
  /** @param {string} root absolute directory that holds the notes */
  constructor(root, { now = () => new Date() } = {}) {
    this.root = path.resolve(root)
    this.now = now
  }

  fileFor(slug) {
    const file = path.join(this.root, `${slug}.md`)
    if (path.dirname(file) !== this.root) throw new VaultError('invalid note name')   // belt and braces after slugify
    return file
  }

  async #slugs() {
    try {
      return (await readdir(this.root)).filter(f => f.endsWith('.md')).map(f => f.slice(0, -3)).sort()
    } catch (err) {
      if (err.code === 'ENOENT') return []
      throw err
    }
  }

  async #load(slug) {
    try {
      return { slug, ...parse(await readFile(this.fileFor(slug), 'utf8'), slug) }
    } catch (err) {
      if (err.code === 'ENOENT') return null
      throw err
    }
  }

  async #all() {
    return (await Promise.all((await this.#slugs()).map(s => this.#load(s)))).filter(Boolean)
  }

  /** Create or update a note. mode "append" adds to the end of an existing note. */
  async save({ title, content, tags = [], mode = 'replace' }) {
    title = String(title ?? '').trim()
    if (!title || title.length > MAX_TITLE_CHARS) throw new VaultError(`title must be 1-${MAX_TITLE_CHARS} characters`)
    if (typeof content !== 'string') throw new VaultError('content must be a string')
    if (!['replace', 'append'].includes(mode)) throw new VaultError('mode must be "replace" or "append"')
    if (!Array.isArray(tags) || tags.some(t => typeof t !== 'string')) throw new VaultError('tags must be an array of strings')
    const slug = slugify(title)
    const existing = await this.#load(slug)
    if (!existing && (await this.#slugs()).length >= MAX_NOTES) throw new VaultError(`vault is full (${MAX_NOTES} notes)`)
    const body = existing && mode === 'append' ? `${existing.body.replace(/\s+$/, '')}\n\n${content}` : content
    if (body.length > MAX_CONTENT_CHARS) throw new VaultError(`note too long (${body.length} > ${MAX_CONTENT_CHARS} chars)`)
    const mergedTags = [...new Set([...(existing && mode === 'append' ? existing.tags : []), ...tags.map(t => t.trim().toLowerCase()).filter(Boolean)])]
    await mkdir(this.root, { recursive: true })
    const file = this.fileFor(slug)
    const tmp = `${file}.${process.pid}.tmp`
    await writeFile(tmp, serialize({ title: existing?.title ?? title, tags: mergedTags, updated: this.now().toISOString(), body }), 'utf8')
    await rename(tmp, file)                                                  // atomic replace: no half-written note
    const links = extractLinks(body)
    const known = new Set(await this.#slugs())
    return { slug, created: !existing, links: links.map(l => ({ ...l, exists: known.has(l.slug) })) }
  }

  /** A note, its outgoing links (flagging dangling ones) and the notes that link to it. */
  async read(title) {
    const slug = slugify(title)
    const note = await this.#load(slug)
    if (!note) {
      const close = (await this.#slugs()).filter(s => s.includes(slug) || slug.includes(s)).slice(0, 5)
      throw new VaultError(`no note titled "${title}"` + (close.length ? `. Similar: ${close.join(', ')}` : ''))
    }
    const all = await this.#all()
    const known = new Set(all.map(n => n.slug))
    return {
      slug, title: note.title, tags: note.tags, updated: note.updated, body: note.body.trim(),
      links: extractLinks(note.body).map(l => ({ ...l, exists: known.has(l.slug) })),
      backlinks: all.filter(n => n.slug !== slug && extractLinks(n.body).some(l => l.slug === slug)).map(n => ({ slug: n.slug, title: n.title })),
    }
  }

  /** Ranked keyword search: title hits weigh most, then tags, then body term frequency. */
  async search(query, limit = 5) {
    const terms = String(query ?? '').toLowerCase().split(/[^a-z0-9]+/).filter(t => t.length > 1)
    if (!terms.length) throw new VaultError('query has no searchable words')
    const results = []
    for (const note of await this.#all()) {
      const title = note.title.toLowerCase(), body = note.body.toLowerCase(), tags = note.tags.join(' ')
      let score = 0
      for (const t of terms) {
        if (title.includes(t)) score += 10
        if (tags.includes(t)) score += 5
        score += Math.min(body.split(t).length - 1, 5)
      }
      if (score > 0) results.push({ slug: note.slug, title: note.title, tags: note.tags, score, snippet: snippet(note.body, terms) })
    }
    return results.sort((a, b) => b.score - a.score || a.slug.localeCompare(b.slug)).slice(0, Math.max(1, Math.min(limit, 20)))
  }

  /** Every note with its link counts; `orphans` have no links in either direction. */
  async list(tag) {
    const all = await this.#all()
    const incoming = new Map()
    for (const n of all) for (const l of extractLinks(n.body)) if (l.slug !== n.slug) incoming.set(l.slug, (incoming.get(l.slug) ?? 0) + 1)
    const wanted = tag ? String(tag).trim().toLowerCase() : null
    return all.filter(n => !wanted || n.tags.includes(wanted)).map(n => {
      const outgoing = extractLinks(n.body).filter(l => l.slug !== n.slug).length
      return { slug: n.slug, title: n.title, tags: n.tags, outgoing, incoming: incoming.get(n.slug) ?? 0, orphan: outgoing === 0 && !incoming.has(n.slug) }
    })
  }
}

function snippet(body, terms) {
  const flat = body.replace(/\s+/g, ' ').trim()
  const lower = flat.toLowerCase()
  const at = terms.map(t => lower.indexOf(t)).filter(i => i >= 0).sort((a, b) => a - b)[0] ?? 0
  const start = Math.max(0, at - 40)
  return (start > 0 ? '…' : '') + flat.slice(start, start + 160) + (start + 160 < flat.length ? '…' : '')
}
