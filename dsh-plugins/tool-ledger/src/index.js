// tool-ledger -- a DeepSeek Harness plugin: per-tool call quotas + a JSONL audit trail + a summary tool.
//
//   tools/pre-execute  (reorderable allow/deny waterfall)  -> enforce the quota, deny with a model-readable reason
//   tools/result       (observe-only, failures contained)  -> append one redacted audit row per finished call
//   tool_ledger        (a registered tool)                 -> lets the model see its own remaining budget
//
// Counting is per agent session (`exec.agent.id`), so a sub-agent's budget is separate from its parent's.
import path from 'node:path'
import { Ledger } from './ledger.js'

export const name = 'tool-ledger'
export const inject = ['tools']

export function resolveConfig(config = {}, cwd = process.cwd()) {
  const defaultLimit = config.defaultLimit ?? 50
  if (!(defaultLimit === Infinity || (Number.isInteger(defaultLimit) && defaultLimit >= 0))) throw new TypeError('tool-ledger: config.defaultLimit must be a non-negative integer')
  const limits = config.limits ?? {}
  for (const [tool, n] of Object.entries(limits)) {
    if (!Number.isInteger(n) || n < 0) throw new TypeError(`tool-ledger: limits.${tool} must be a non-negative integer`)
  }
  const logFile = config.logFile === null ? null : path.resolve(cwd, config.logFile ?? '.tool-ledger/audit.jsonl')
  return { defaultLimit, limits, exempt: ['tool_ledger', ...(config.exempt ?? [])], logFile, maxArgChars: config.maxArgChars ?? 300 }
}

const sessionOf = exec => String(exec?.agent?.id ?? 'global')

export function textOf(result) {
  return (result?.content ?? []).map(b => (b?.type === 'text' ? b.text : '')).join('')
}

/** The pre-execute listener, exported so it can be tested without Cordis. */
export function makePreExecute(ledger) {
  return async (exec, next) => {
    const session = sessionOf(exec)
    const reason = ledger.check(session, exec.name)
    if (reason) {
      await ledger.noteDenied(session, exec.name, reason, exec.arguments, exec.callId)
      return { kind: 'deny', reason }                       // a returned decision, never a throw
    }
    ledger.charge(session, exec.name)                       // reserve BEFORE awaiting, so parallel calls cannot both slip under the limit
    let decision
    try {
      decision = await next()                               // let the rest of the policy chain have its say
    } catch (err) {
      ledger.refund(session, exec.name)
      throw err
    }
    if (!decision || (decision.kind !== 'allow' && decision.kind !== 'ask')) {
      ledger.refund(session, exec.name)                     // someone else denied it: it never ran, so it costs no quota
      if (decision?.kind === 'deny') await ledger.noteDenied(session, exec.name, decision.reason, exec.arguments, exec.callId)
    }
    return decision
  }
}

export function makeOnResult(ledger) {
  return (exec, result) => {
    void ledger.noteResult(sessionOf(exec), exec.name, exec.arguments, { isError: result?.isError, chars: textOf(result).length }, exec.callId)
  }
}

export function makeSummaryTool(ledger) {
  return {
    name: 'tool_ledger',
    description: 'Show how many times each tool was used this session against its quota, plus the most recent audited calls. Free to call.',
    parameters: { type: 'object', properties: { recent: { type: 'integer', description: 'how many recent calls to include (default 5, max 50)' } } },
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    async execute(args, exec) {
      return JSON.stringify(ledger.summary(sessionOf(exec), args?.recent ?? 5))
    },
  }
}

export function apply(ctx, config = {}) {
  const ledger = new Ledger(resolveConfig(config))
  ctx.tools.register(makeSummaryTool(ledger))
  ctx.on('tools/pre-execute', makePreExecute(ledger))
  ctx.on('tools/result', makeOnResult(ledger))
}

export default { name, inject, apply }
