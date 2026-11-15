// second-brain -- a DeepSeek Harness plugin: a markdown vault with [[wikilinks]] and backlinks, exposed as four tools.
//
// Wiring only; the logic is in vault.js. Verified against the DSH docs (see ../../NOTES.md):
//   * a plugin is a module exporting `name`, `inject` and `apply(ctx, config)`
//   * tools:   ctx.tools.register(<ToolDefinition>)   (raw JSON-Schema definitions are accepted, so this file imports nothing from DSH)
//   * prompt:  ctx.systemPrompt.section({ name, order, text, interpolate })
import path from 'node:path'
import { Vault, VaultError } from './vault.js'

export const name = 'second-brain'
export const inject = ['tools', 'systemPrompt']

const PROMPT_ORDER = 10_300                       // just after the deployment persona suffix (10200)

/** Hand-written validation keeps the plugin dependency-free. The idiomatic alternative is an exported Schemastery `Config`. */
export function resolveConfig(config = {}, cwd = process.cwd()) {
  const root = config.root ?? path.join(cwd, '.second-brain')
  if (typeof root !== 'string' || !root.trim()) throw new TypeError('second-brain: config.root must be a non-empty string')
  const maxResults = config.maxSearchResults ?? 5
  if (!Number.isInteger(maxResults) || maxResults < 1 || maxResults > 20) throw new TypeError('second-brain: config.maxSearchResults must be an integer 1-20')
  return { root: path.resolve(cwd, root), maxSearchResults: maxResults, injectPrompt: config.injectPrompt !== false }
}

export const PROMPT = [
  '## Second brain',
  'You have a persistent markdown vault of notes that survives between sessions.',
  '- Before answering a question about the user\'s projects, decisions or preferences, call brain_search.',
  '- Save durable facts with brain_save (one idea per note, a clear title). Link related notes with [[Note Title]]; links create backlinks.',
  '- Use brain_read to see a note together with its backlinks, brain_list to see the whole map (orphans have no links).',
  '- Never store secrets, passwords or API keys in the vault.',
].join('\n')

function tool(name, description, properties, required, run) {
  return {
    name,
    description,
    parameters: { type: 'object', properties, ...(required.length ? { required } : {}) },
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    async execute(args) {
      try {
        return JSON.stringify(await run(args ?? {}))
      } catch (err) {
        if (err instanceof VaultError) throw new Error(err.message)       // a clear, model-readable tool error
        throw err
      }
    },
  }
}

export function createTools(vault, { maxSearchResults = 5 } = {}) {
  return [
    tool('brain_save', 'Create or update a note in the second-brain vault. Mention other notes as [[Note Title]] to link them.', {
      title: { type: 'string', description: 'Note title (also its identity: same title = same note)' },
      content: { type: 'string', description: 'Markdown body. Use [[Other Title]] to link notes.' },
      tags: { type: 'array', items: { type: 'string' }, description: 'Optional short tags' },
      mode: { type: 'string', enum: ['replace', 'append'], description: 'replace (default) or append to an existing note' },
    }, ['title', 'content'], args => vault.save(args)),
    tool('brain_read', 'Read one note with its outgoing links and the notes that link to it (backlinks).',
      { title: { type: 'string' } }, ['title'], args => vault.read(args.title)),
    tool('brain_search', 'Keyword-search the vault. Returns ranked notes with a snippet.', {
      query: { type: 'string' },
      limit: { type: 'integer', description: `max results (default ${maxSearchResults})` },
    }, ['query'], args => vault.search(args.query, args.limit ?? maxSearchResults)),
    tool('brain_list', 'List every note with link counts; optionally filter by tag.',
      { tag: { type: 'string' } }, [], args => vault.list(args.tag)),
  ]
}

export function apply(ctx, config = {}) {
  const cfg = resolveConfig(config)
  const vault = new Vault(cfg.root)
  for (const definition of createTools(vault, cfg)) ctx.tools.register(definition)
  if (cfg.injectPrompt) {
    ctx.systemPrompt.section({ name: 'second-brain', order: PROMPT_ORDER, text: PROMPT, interpolate: false })
  }
}

export default { name, inject, apply }
