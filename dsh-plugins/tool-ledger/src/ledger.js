// ledger.js -- pure logic of the tool-ledger plugin: per-tool call quotas, argument redaction, an audit trail.
// Part A's step_06 (quotas) and step_05 (audit) ideas, as a DSH plugin. No Cordis imports: unit-tested with node --test.
import { appendFile, mkdir } from 'node:fs/promises'
import path from 'node:path'

const SECRET_KEY = /key|token|secret|password|passwd|authorization|credential/i

/** Deep copy of tool arguments with secret-looking fields masked and long strings truncated. */
export function redact(value, maxChars = 300, depth = 0) {
  if (depth > 6) return '[deep]'
  if (typeof value === 'string') return value.length > maxChars ? `${value.slice(0, maxChars)}…[${value.length - maxChars} more chars]` : value
  if (Array.isArray(value)) return value.slice(0, 20).map(v => redact(v, maxChars, depth + 1))
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).slice(0, 40).map(([k, v]) => [k, SECRET_KEY.test(k) ? '[redacted]' : redact(v, maxChars, depth + 1)]))
  }
  return value
}

export class Ledger {
  /**
   * @param {object} opts
   * @param {number} opts.defaultLimit    calls allowed per tool per agent session (Infinity = unlimited)
   * @param {Record<string, number>} opts.limits     per-tool overrides
   * @param {string[]} opts.exempt        tools never limited (the ledger's own tool, by default)
   * @param {string|null} opts.logFile    JSONL audit file (null = in-memory only)
   */
  constructor({ defaultLimit = 50, limits = {}, exempt = ['tool_ledger'], logFile = null, maxArgChars = 300, now = () => new Date() } = {}) {
    this.defaultLimit = defaultLimit
    this.limits = limits
    this.exempt = new Set(exempt)
    this.logFile = logFile
    this.maxArgChars = maxArgChars
    this.now = now
    this.used = new Map()           // `${session}\u0000${tool}` -> count
    this.stats = new Map()          // tool -> { allowed, denied, errors }
    this.recent = []                // last 100 audit rows, for the summary tool
    this.denied = new Set()         // callIds we denied: their (error) result is already audited as a denial
    this.writeFailures = 0
  }

  limitFor(tool) {
    return this.exempt.has(tool) ? Infinity : (this.limits[tool] ?? this.defaultLimit)
  }

  #stat(tool) {
    if (!this.stats.has(tool)) this.stats.set(tool, { allowed: 0, denied: 0, errors: 0 })
    return this.stats.get(tool)
  }

  /** Would this call exceed its quota? Returns a model-readable reason, or null. Does NOT charge. */
  check(session, tool) {
    const limit = this.limitFor(tool)
    const used = this.used.get(`${session}\u0000${tool}`) ?? 0
    return used >= limit ? `QUOTA: '${tool}' was already called ${used} times (limit ${limit} per session). Do not call it again; answer with what you have.` : null
  }

  /** Reserve one call. Call after the policy chain agreed to run it. */
  charge(session, tool) {
    const key = `${session}\u0000${tool}`
    this.used.set(key, (this.used.get(key) ?? 0) + 1)
    this.#stat(tool).allowed++
  }

  /** Give a reservation back: a later policy denied the call, so it never ran. */
  refund(session, tool) {
    const key = `${session}\u0000${tool}`
    this.used.set(key, Math.max(0, (this.used.get(key) ?? 1) - 1))
    this.#stat(tool).allowed = Math.max(0, this.#stat(tool).allowed - 1)
  }

  noteDenied(session, tool, reason, args, callId) {
    this.#stat(tool).denied++
    if (callId !== undefined) this.denied.add(callId)
    return this.audit({ session, tool, decision: 'deny', reason, args })
  }

  noteResult(session, tool, args, { isError, chars }, callId) {
    if (callId !== undefined && this.denied.delete(callId)) return Promise.resolve(null)   // already audited as a denial
    if (isError) this.#stat(tool).errors++
    return this.audit({ session, tool, decision: 'ran', isError: Boolean(isError), chars, args })
  }

  async audit(row) {
    const entry = { ts: this.now().toISOString(), ...row, args: redact(row.args, this.maxArgChars) }
    this.recent.push(entry)
    if (this.recent.length > 100) this.recent.shift()
    if (!this.logFile) return entry
    try {
      await mkdir(path.dirname(this.logFile), { recursive: true })
      await appendFile(this.logFile, `${JSON.stringify(entry)}\n`, 'utf8')
    } catch {
      this.writeFailures++            // an audit-log problem must never break the agent's tool call
    }
    return entry
  }

  summary(session, lastN = 10) {
    const tools = {}
    for (const [tool, s] of this.stats) {
      const limit = this.limitFor(tool)
      tools[tool] = { ...s, used: this.used.get(`${session}\u0000${tool}`) ?? 0, limit: Number.isFinite(limit) ? limit : 'unlimited' }
    }
    return { session, tools, recent: this.recent.slice(-Math.max(0, Math.min(lastN, 50))), auditWriteFailures: this.writeFailures }
  }
}
