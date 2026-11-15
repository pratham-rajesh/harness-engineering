import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import plugin, { PROMPT, apply, inject, name, resolveConfig } from '../src/index.js'

function fakeCtx() {
  const tools = [], sections = []
  return { tools: { register: t => tools.push(t) }, systemPrompt: { section: s => sections.push(s) }, registered: tools, sections }
}
const run = async (ctx, toolName, args) => ctx.registered.find(t => t.name === toolName).execute(args)

test('exports the plugin contract', () => {
  assert.equal(name, 'second-brain')
  assert.deepEqual(inject, ['tools', 'systemPrompt'])
  assert.equal(typeof apply, 'function')
  assert.equal(plugin.apply, apply)
})

test('registers four tools with valid-looking schemas and one prompt section', async () => {
  const ctx = fakeCtx()
  apply(ctx, { root: await mkdtemp(path.join(tmpdir(), 'brain-')) })
  assert.deepEqual(ctx.registered.map(t => t.name), ['brain_save', 'brain_read', 'brain_search', 'brain_list'])
  for (const t of ctx.registered) {
    assert.equal(t.parameters.type, 'object')
    assert.deepEqual(t.output.schema, { type: 'string' })
    assert.deepEqual(t.output.render({}, 'hi'), [{ type: 'text', text: 'hi' }])
    for (const r of t.parameters.required ?? []) assert.ok(r in t.parameters.properties)
  }
  assert.equal(ctx.sections.length, 1)
  assert.deepEqual(ctx.sections[0], { name: 'second-brain', order: 10_300, text: PROMPT, interpolate: false })
})

test('tools work end to end through execute() and return JSON strings', async () => {
  const ctx = fakeCtx()
  apply(ctx, { root: await mkdtemp(path.join(tmpdir(), 'brain-')) })
  const saved = JSON.parse(await run(ctx, 'brain_save', { title: 'Plan', content: 'ship [[Part B]]', tags: ['work'] }))
  assert.equal(saved.created, true)
  assert.equal(saved.links[0].exists, false)
  await run(ctx, 'brain_save', { title: 'Part B', content: 'plugins' })
  assert.deepEqual(JSON.parse(await run(ctx, 'brain_read', { title: 'Part B' })).backlinks.map(b => b.slug), ['plan'])
  assert.equal(JSON.parse(await run(ctx, 'brain_search', { query: 'plugins' }))[0].slug, 'part-b')
  assert.equal(JSON.parse(await run(ctx, 'brain_list', {})).length, 2)
})

test('vault errors surface as plain Errors with a model-readable message', async () => {
  const ctx = fakeCtx()
  apply(ctx, { root: await mkdtemp(path.join(tmpdir(), 'brain-')) })
  await assert.rejects(() => run(ctx, 'brain_read', { title: 'ghost' }), err => err.constructor === Error && /no note titled "ghost"/.test(err.message))
})

test('injectPrompt:false skips the prompt section', async () => {
  const ctx = fakeCtx()
  apply(ctx, { root: await mkdtemp(path.join(tmpdir(), 'brain-')), injectPrompt: false })
  assert.equal(ctx.sections.length, 0)
})

test('resolveConfig applies defaults and rejects nonsense', () => {
  assert.equal(resolveConfig({}, '/work').root, path.resolve('/work', '.second-brain'))
  assert.equal(resolveConfig({ root: 'notes' }, '/work').root, path.resolve('/work', 'notes'))
  assert.throws(() => resolveConfig({ root: '' }), /root/)
  assert.throws(() => resolveConfig({ maxSearchResults: 0 }), /maxSearchResults/)
})
