import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { Vault, VaultError, extractLinks, parse, serialize, slugify } from '../src/vault.js'

const fresh = async () => new Vault(await mkdtemp(path.join(tmpdir(), 'brain-')), { now: () => new Date('2026-01-02T03:04:05Z') })

test('slugify makes safe file names and rejects empty titles', () => {
  assert.equal(slugify('Project X: Ideas!'), 'project-x-ideas')
  assert.equal(slugify('Café déjà vu'), 'cafe-deja-vu')
  assert.equal(slugify('../../etc/passwd'), 'etc-passwd')                 // no traversal characters survive
  assert.ok(slugify('x'.repeat(500)).length <= 80)
  assert.throws(() => slugify('???'), VaultError)
  assert.throws(() => slugify(''), VaultError)
})

test('extractLinks finds wikilinks, handles aliases, dedupes, ignores junk', () => {
  const links = extractLinks('See [[Alpha]] and [[Alpha|the first]] and [[Beta Note]], not [[]] or [[???]] or [x](y).')
  assert.deepEqual(links, [{ slug: 'alpha', title: 'Alpha' }, { slug: 'beta-note', title: 'Beta Note' }])
})

test('serialize/parse round-trip, including awkward titles', () => {
  const note = { title: 'He said "hi": ok', tags: ['a', 'b'], updated: '2026-01-02T03:04:05.000Z', body: 'text\nmore' }
  assert.deepEqual(parse(serialize(note), 'x'), { ...note, body: 'text\nmore\n' })
  assert.equal(parse('no front matter', 'slug').title, 'slug')
  assert.deepEqual(parse('---\ntitle: broken{\ntags: [1\n---\nbody', 's').tags, [])      // malformed header lines are skipped
})

test('save creates, then updates; same title is the same note', async () => {
  const v = await fresh()
  const first = await v.save({ title: 'Idea', content: 'one', tags: ['Work', 'work'] })
  assert.deepEqual([first.slug, first.created], ['idea', true])
  const second = await v.save({ title: 'IDEA', content: 'two' })
  assert.equal(second.created, false)
  const note = await v.read('idea')
  assert.equal(note.body, 'two')
  assert.equal(note.title, 'Idea')                                          // the original title is kept
  assert.deepEqual(note.tags, [])                                           // replace mode replaces tags too
})

test('append mode extends the body and merges tags', async () => {
  const v = await fresh()
  await v.save({ title: 'Log', content: 'day 1', tags: ['a'] })
  await v.save({ title: 'Log', content: 'day 2', tags: ['b'], mode: 'append' })
  const note = await v.read('Log')
  assert.equal(note.body, 'day 1\n\nday 2')
  assert.deepEqual(note.tags, ['a', 'b'])
})

test('links, dangling links and backlinks', async () => {
  const v = await fresh()
  await v.save({ title: 'Hub', content: 'Links to [[Spoke A]] and [[Missing]]' })
  await v.save({ title: 'Spoke A', content: 'Back to [[Hub]]' })
  const hub = await v.read('Hub')
  assert.deepEqual(hub.links.map(l => [l.slug, l.exists]), [['spoke-a', true], ['missing', false]])
  assert.deepEqual(hub.backlinks.map(b => b.slug), ['spoke-a'])
  assert.deepEqual((await v.read('spoke-a')).backlinks.map(b => b.slug), ['hub'])
})

test('read of a missing note suggests similar names', async () => {
  const v = await fresh()
  await v.save({ title: 'Travel plans', content: 'x' })
  await assert.rejects(() => v.read('travel'), /no note titled "travel".*travel-plans/)
})

test('search ranks title over body and returns snippets', async () => {
  const v = await fresh()
  await v.save({ title: 'Coffee beans', content: 'Buy arabica.' })
  await v.save({ title: 'Shopping', content: 'coffee coffee coffee tea milk' })
  await v.save({ title: 'Unrelated', content: 'nothing here' })
  const hits = await v.search('coffee')
  assert.deepEqual(hits.map(h => h.slug), ['coffee-beans', 'shopping'])
  assert.match(hits[1].snippet, /coffee/)
  await assert.rejects(() => v.search('a !'), /no searchable words/)
})

test('list reports orphans, link counts and filters by tag', async () => {
  const v = await fresh()
  await v.save({ title: 'A', content: '[[B]]', tags: ['x'] })
  await v.save({ title: 'B', content: 'leaf' })
  await v.save({ title: 'Lonely', content: 'alone', tags: ['x'] })
  const all = Object.fromEntries((await v.list()).map(n => [n.slug, n]))
  assert.deepEqual([all.a.outgoing, all.b.incoming, all.lonely.orphan, all.a.orphan], [1, 1, true, false])
  assert.deepEqual((await v.list('X')).map(n => n.slug), ['a', 'lonely'])
})

test('rejects bad input and cannot write outside the vault', async () => {
  const v = await fresh()
  await assert.rejects(() => v.save({ title: '', content: 'x' }), VaultError)
  await assert.rejects(() => v.save({ title: 'ok', content: 5 }), /content must be a string/)
  await assert.rejects(() => v.save({ title: 'ok', content: 'x', mode: 'nuke' }), /mode/)
  await assert.rejects(() => v.save({ title: 'ok', content: 'x'.repeat(100_001) }), /too long/)
  const evil = await v.save({ title: '../../escape', content: 'x' })
  assert.equal(evil.slug, 'escape')
  assert.deepEqual(await readdir(v.root), ['escape.md'])
})

test('a hand-edited file without front matter is still readable and searchable', async () => {
  const v = await fresh()
  await v.save({ title: 'seed', content: 'x' })
  await writeFile(path.join(v.root, 'manual.md'), 'plain notes about kayaks', 'utf8')
  assert.equal((await v.read('manual')).body, 'plain notes about kayaks')
  assert.equal((await v.search('kayaks'))[0].slug, 'manual')
})

test('writes are atomic: no temp files are left behind', async () => {
  const v = await fresh()
  await v.save({ title: 'One', content: 'a' })
  await v.save({ title: 'One', content: 'b' })
  assert.deepEqual((await readdir(v.root)).filter(f => f.includes('.tmp')), [])
  assert.match(await readFile(path.join(v.root, 'one.md'), 'utf8'), /updated: 2026-01-02T03:04:05\.000Z/)
})

test('empty vault: list/search work, directory is created lazily', async () => {
  const v = new Vault(path.join(await mkdtemp(path.join(tmpdir(), 'brain-')), 'does', 'not', 'exist'))
  assert.deepEqual(await v.list(), [])
  assert.deepEqual(await v.search('anything'), [])
})
