# tool-ledger

A DeepSeek Harness plugin that gives the agent a **budget** and gives you an **audit trail** — Part A's quotas (step 06) and
approval/audit ideas as a plugin.

| piece | DSH extension point | behaviour |
|---|---|---|
| quota | `tools/pre-execute` (reorderable allow/deny waterfall) | after N calls of a tool in one agent session, return `{kind:'deny', reason:"QUOTA: ..."}` — a returned decision, never a throw |
| audit | `tools/result` (observe-only) | append one redacted JSON line per finished call to a JSONL file |
| `tool_ledger` tool | `ctx.tools.register` | lets the model read its own usage vs limits (exempt from quotas) |

## Details that matter
- **Reserve, then ask**: the call is charged *before* `await next()` so parallel calls can't both slip under a limit of 1;
  if a later policy denies it (or throws), the reservation is refunded — a call that never ran costs no quota.
- **Per agent session** (`exec.agent.id`): a sub-agent has its own budget; no agent = bucket `global`.
- **Redaction**: argument keys matching `key|token|secret|password|passwd|authorization|credential` are masked; long strings truncated.
- **Never breaks the agent**: an unwritable audit file increments `auditWriteFailures` instead of throwing.
- Our own denials are audited once (as `deny`), not twice (the executor also reports a denied call as an error result).

## Config
```yaml
config:
  defaultLimit: 50          # calls per tool per session
  limits: { bash: 10 }      # per-tool overrides (0 blocks a tool)
  exempt: []                # extra tools that are never limited (tool_ledger always is)
  logFile: /abs/audit.jsonl # null = memory only; default <cwd>/.tool-ledger/audit.jsonl
```

## Tests
`node --test test/` -> 16 tests including the parallel-call race, refund-on-foreign-deny, per-session budgets, redaction, unwritable log.

## Not verified
Not launched inside DSH by the author. The `exec.agent.id`, `exec.callId` and `result.isError` fields come from the DSH source
(`packages/core/tools/src/contracts.ts`, `packages/core/agent/src/types.ts`); the unit tests use a fake `ctx` built from those shapes.
