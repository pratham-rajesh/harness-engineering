import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import plugin, { apply, inject, name, resolveConfig } from '../src/index.js'

function fakeCtx() {
  const listeners = {}, tools = []
  return {
    registered: tools, listeners,
    tools: { register: t => tools.push(t) },
    on: (event, fn) => { listeners[event] = fn },
  }
}
const exec = (toolName, args = {}, extra = {}) => ({ name: toolName, arguments: args, callId: `c${Math.random()}`, agent: { id: 'sess-1' }, ...extra })
const allow = async () => ({ kind: 'allow' })
const setup = async config => {
  const dir = await mkdtemp(path.join(tmpdir(), 'tl-'))
  const ctx = fakeCtx()
  apply(ctx, { logFile: path.join(dir, 'audit.jsonl'), ...config })
  return { ctx, file: path.join(dir, 'audit.jsonl'), pre: ctx.listeners['tools/pre-execute'], post: ctx.listeners['tools/result'] }
}

test('exports the plugin contract and registers listeners + the summary tool', async () => {
  assert.equal(name, 'tool-ledger')
  assert.deepEqual(inject, ['tools'])
  assert.equal(plugin.apply, apply)
  const { ctx } = await setup({})
  assert.deepEqual(ctx.registered.map(t => t.name), ['tool_ledger'])
  assert.deepEqual(Object.keys(ctx.listeners).sort(), ['tools/pre-execute', 'tools/result'])
})

test('pre-execute denies by RETURNING a decision once the per-tool limit is reached', async () => {
  const { pre } = await setup({ limits: { bash: 2 } })
  assert.deepEqual(await pre(exec('bash'), allow), { kind: 'allow' })
  assert.deepEqual(await pre(exec('bash'), allow), { kind: 'allow' })
  const third = await pre(exec('bash'), allow)
  assert.equal(third.kind, 'deny')
  assert.match(third.reason, /^QUOTA: 'bash'/)
  assert.deepEqual(await pre(exec('read'), allow), { kind: 'allow' })       // other tools unaffected
})

test('a call denied by ANOTHER policy costs no quota', async () => {
  const { pre } = await setup({ limits: { bash: 1 } })
  const deny = async () => ({ kind: 'deny', reason: 'sandbox says no' })
  assert.equal((await pre(exec('bash'), deny)).reason, 'sandbox says no')
  assert.equal((await pre(exec('bash'), allow)).kind, 'allow')              // the one allowed call is still available
  assert.equal((await pre(exec('bash'), allow)).kind, 'deny')
})

test('a throwing downstream listener refunds the reservation and rethrows', async () => {
  const { pre } = await setup({ limits: { bash: 1 } })
  await assert.rejects(() => pre(exec('bash'), async () => { throw new Error('downstream') }), /downstream/)
  assert.equal((await pre(exec('bash'), allow)).kind, 'allow')
})

test('parallel calls cannot both slip under a limit of 1', async () => {
  const { pre } = await setup({ limits: { bash: 1 } })
  const slow = () => new Promise(r => setTimeout(() => r({ kind: 'allow' }), 20))
  const results = await Promise.all([pre(exec('bash'), slow), pre(exec('bash'), slow), pre(exec('bash'), slow)])
  assert.deepEqual(results.map(r => r.kind).sort(), ['allow', 'deny', 'deny'])
})

test('budgets are per agent session', async () => {
  const { pre } = await setup({ limits: { bash: 1 } })
  assert.equal((await pre(exec('bash', {}, { agent: { id: 'parent' } }), allow)).kind, 'allow')
  assert.equal((await pre(exec('bash', {}, { agent: { id: 'child' } }), allow)).kind, 'allow')
  assert.equal((await pre(exec('bash', {}, { agent: { id: 'parent' } }), allow)).kind, 'deny')
  assert.equal((await pre(exec('bash', {}, { agent: undefined }), allow)).kind, 'allow')   // no agent -> "global"
})

test('audit file: one ran row, one deny row, no duplicate row for our own denials; secrets redacted', async () => {
  const { pre, post, file } = await setup({ limits: { bash: 1 } })
  const ok = exec('bash', { command: 'ls', apiKey: 'sk-secret' })
  await pre(ok, allow)
  post(ok, { isError: false, content: [{ type: 'text', text: 'file.txt' }] })
  const blocked = exec('bash', { command: 'ls' })
  const decision = await pre(blocked, allow)
  post(blocked, { isError: true, content: [{ type: 'text', text: decision.reason }] })      // the executor also reports the denial as a result
  await new Promise(r => setTimeout(r, 50))
  const text = await readFile(file, 'utf8')
  const rows = text.trim().split('\n').map(r => JSON.parse(r))
  assert.equal(rows.length, 2)
  const ran = rows.find(r => r.decision === 'ran'), denied = rows.find(r => r.decision === 'deny')
  assert.equal(ran.args.apiKey, '[redacted]')
  assert.equal(ran.chars, 8)
  assert.match(denied.reason, /QUOTA/)
  assert.ok(!text.includes('sk-secret'))
})

test('tool_ledger reports usage for the calling session and is exempt from quotas', async () => {
  const { ctx, pre } = await setup({ defaultLimit: 0, limits: { bash: 3 } })
  await pre(exec('bash'), allow)
  const tool = ctx.registered[0]
  const out = JSON.parse(await tool.execute({ recent: 3 }, { agent: { id: 'sess-1' } }))
  assert.deepEqual([out.session, out.tools.bash.used, out.tools.bash.limit], ['sess-1', 1, 3])
  for (let i = 0; i < 10; i++) assert.equal((await pre(exec('tool_ledger'), allow)).kind, 'allow')
  assert.deepEqual(tool.output.render({}, 'x'), [{ type: 'text', text: 'x' }])
})

test('resolveConfig defaults and validation', () => {
  const cfg = resolveConfig({}, '/work')
  assert.equal(cfg.defaultLimit, 50)
  assert.equal(cfg.logFile, path.resolve('/work', '.tool-ledger/audit.jsonl'))
  assert.equal(resolveConfig({ logFile: null }).logFile, null)
  assert.ok(resolveConfig({ exempt: ['x'] }).exempt.includes('tool_ledger'))
  assert.throws(() => resolveConfig({ defaultLimit: -1 }), /defaultLimit/)
  assert.throws(() => resolveConfig({ limits: { bash: 1.5 } }), /limits\.bash/)
})
