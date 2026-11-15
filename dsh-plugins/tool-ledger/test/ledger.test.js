import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { Ledger, redact } from '../src/ledger.js'

test('redact masks secret-looking keys, truncates long strings, bounds depth and size', () => {
  const out = redact({ command: 'ls', apiKey: 'sk-123', nested: { Authorization: 'Bearer x', ok: 1 }, big: 'y'.repeat(50) }, 10)
  assert.equal(out.apiKey, '[redacted]')
  assert.equal(out.nested.Authorization, '[redacted]')
  assert.equal(out.nested.ok, 1)
  assert.match(out.big, /^y{10}…\[40 more chars\]$/)
  assert.equal(redact(Array.from({ length: 100 }, (_, i) => i)).length, 20)
  let deep = 'leaf'
  for (let i = 0; i < 10; i++) deep = { x: deep }
  assert.match(JSON.stringify(redact(deep)), /\[deep\]/)
})

test('quota: allows up to the limit, then denies with a QUOTA string; limits are per session and per tool', () => {
  const l = new Ledger({ defaultLimit: 2, limits: { bash: 1 } })
  assert.equal(l.check('s1', 'bash'), null)
  l.charge('s1', 'bash')
  assert.match(l.check('s1', 'bash'), /^QUOTA: 'bash'.*limit 1/)
  assert.equal(l.check('s2', 'bash'), null)                                // a different session has its own budget
  l.charge('s1', 'read'); l.charge('s1', 'read')
  assert.match(l.check('s1', 'read'), /limit 2/)
  assert.equal(l.check('s1', 'grep'), null)                                // unrelated tool untouched
})

test('exempt tools are never limited; limit 0 blocks a tool outright', () => {
  const l = new Ledger({ defaultLimit: 0, limits: { allowed_one: 3 }, exempt: ['tool_ledger', 'free'] })
  for (let i = 0; i < 100; i++) l.charge('s', 'tool_ledger')
  assert.equal(l.check('s', 'tool_ledger'), null)
  assert.equal(l.check('s', 'free'), null)
  assert.match(l.check('s', 'anything'), /limit 0/)
  assert.equal(l.check('s', 'allowed_one'), null)
})

test('refund returns a reservation', () => {
  const l = new Ledger({ defaultLimit: 1 })
  l.charge('s', 't')
  assert.ok(l.check('s', 't'))
  l.refund('s', 't')
  assert.equal(l.check('s', 't'), null)
  l.refund('s', 't'); l.refund('s', 't')                                   // never goes negative
  assert.equal(l.summary('s').tools.t.used, 0)
})

test('audit rows go to a JSONL file with redacted arguments, and summary reports them', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'ledger-'))
  const file = path.join(dir, 'sub', 'audit.jsonl')
  const l = new Ledger({ logFile: file, now: () => new Date('2026-01-01T00:00:00Z') })
  await l.noteResult('s', 'bash', { command: 'ls', token: 'abc' }, { isError: false, chars: 12 })
  await l.noteResult('s', 'bash', { command: 'boom' }, { isError: true, chars: 3 })
  const rows = (await readFile(file, 'utf8')).trim().split('\n').map(r => JSON.parse(r))
  assert.equal(rows.length, 2)
  assert.deepEqual(rows[0], { ts: '2026-01-01T00:00:00.000Z', session: 's', tool: 'bash', decision: 'ran', isError: false, chars: 12, args: { command: 'ls', token: '[redacted]' } })
  const s = l.summary('s', 1)
  assert.equal(s.tools.bash.errors, 1)
  assert.equal(s.recent.length, 1)
})

test('an unwritable log file never throws; failures are counted', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'ledger-'))
  const l = new Ledger({ logFile: path.join(dir, '\0bad', 'x.jsonl') })      // NUL in a path is always invalid
  await l.noteResult('s', 't', {}, { isError: false, chars: 0 })
  assert.equal(l.summary('s').auditWriteFailures, 1)
  assert.equal(l.summary('s').recent.length, 1)                              // still kept in memory
})

test('recent buffer is capped at 100 rows', async () => {
  const l = new Ledger()
  for (let i = 0; i < 150; i++) await l.noteResult('s', 't', { i }, { isError: false, chars: 0 })
  assert.equal(l.recent.length, 100)
  assert.equal(l.recent[0].args.i, 50)
})
