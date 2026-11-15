# Part B — `dsh-plugins/`: extending DeepSeek Harness

DeepSeek Harness (DSH) is a Cordis-based agent harness. Everything is a plugin: a module exporting `name`, `inject` and
`apply(ctx, config)`. Part A built the harness ideas by hand; this part uses a real harness and extends it.

## Status — read this first
| claim | status |
|---|---|
| two custom plugins written and unit-tested | **done**: `node --test` -> 35 tests pass (`npm test` in this folder) |
| plugin API shapes match DSH | checked against the DSH docs and source (`defineTool`/`ToolDefinition`, `tools/pre-execute` + `PreToolDecision`, `tools/result`, `systemPrompt.section`, patch `insert`) |
| third-party plugins exist | **done**: each name checked with `npm view` (see `installed-plugins.txt`) |
| plugins load and work **inside a running DSH** | **NOT done.** Launching DSH here needs running a package fetched from npm (`npx @deepseek-ai/dsh`), which the environment's permission layer declined, so no live DSH session happened. Run the commands below yourself before recording. |
| third-party plugins installed / demoed | **NOT done** (same reason) |
| plugins authored in DSH **Creator mode** | **NOT done.** They were written by a coding assistant outside DSH. The prompts below describe the same specs for you to try in Creator mode; results untested. |

## The two plugins I wrote
| plugin | extension points used | Part A concept it mirrors |
|---|---|---|
| `second-brain` | `ctx.tools.register` x4, `ctx.systemPrompt.section` | tools + lazy context (steps 02, 08): a vault the model queries on demand instead of stuffing notes in the prompt |
| `tool-ledger` | `tools/pre-execute`, `tools/result`, `ctx.tools.register` | quotas (step 06), approval/audit (step 05): deny by *returned decision*, audit every call |

Design choices, short:
- **Logic split from wiring.** `vault.js` / `ledger.js` have no DSH imports; `index.js` is ~50 lines of Cordis glue. That is what makes them
  testable with `node --test` and a fake `ctx`, and it keeps the plugin free of build steps (plain ESM JavaScript).
- **Raw `ToolDefinition`s** (`{name, description, parameters (JSON Schema), output:{schema, render}, execute}`) instead of `defineTool` so the
  package imports nothing from DSH. The DSH docs state raw definitions are accepted by `ctx.tools.register()`. Trade-off: `defineTool` would
  validate arguments for me; here each tool validates its own input.
- **Hand-validated config** instead of an exported Schemastery `Config` (the idiomatic way). Trade-off: no automatic UI/schema generation.
- **Deny, don't throw** in `tools/pre-execute`: the model gets a readable `QUOTA:` reason, same as Part A's quota strings.
- **Reserve-then-refund quota accounting**, because `pre-execute` is async and parallel tool calls exist.

## Run it yourself (not yet verified by the author)
```bash
npm test --prefix dsh-plugins                       # 35 unit tests, no dependencies
node dsh-plugins/make-patch.mjs                     # writes dsh-plugins/cordis.local.yml with ABSOLUTE paths (DSH requires them)
# in a terminal, with Node >= 24.9:
npx @deepseek-ai/dsh@0.2.2-rc.1 --profile web --patch ./dsh-plugins/cordis.local.yml --dump-config   # prints composed config, no boot
npx @deepseek-ai/dsh@0.2.2-rc.1 --profile web --patch ./dsh-plugins/cordis.local.yml --no-open        # then open the printed URL
```
Flags are from DSH's `docs/cli-help.md`; the exact launcher name/flags for the published package are unverified. If `npx` cannot resolve a
plugin path with spaces or a drive letter, try a `file:///C:/...` URL or copy the folder to a path without spaces.
Free models: copy `openrouter.patch.example.yml` into your profile's `cordis.patch.yml`, `export OPENROUTER_API_KEY=...`.

## Demo script (for the recording)
1. Plugins page shows `second-brain` and `tool-ledger` loaded; open the tool list: `brain_*` and `tool_ledger` appear.
2. "Remember that my API design review is on Friday and links to [[Project Falcon]]." -> `brain_save`; then `brain_read Project Falcon` shows the backlink.
3. New session: "When is my API design review?" -> `brain_search`.
4. Ask for 11 shell commands: the 11th is denied with `QUOTA: 'bash' ...` (limit 10 in the generated patch); call `tool_ledger` to show usage;
   `cat .tool-ledger/audit.jsonl` shows redacted rows.

## Creator-mode prompts (untested here; paste into DSH Creator mode to reproduce)
**second-brain**
> Create a DSH plugin named `second-brain`. It keeps a folder of markdown notes (default `<cwd>/.second-brain`, configurable). Register four tools:
> `brain_save(title, content, tags?, mode)`, `brain_read(title)` (return outgoing `[[wikilinks]]`, flag dangling ones, and list backlinks),
> `brain_search(query, limit)` (rank title > tags > body) and `brain_list(tag?)` (flag orphans). Titles are slugified so they can never escape the
> folder; writes are atomic. Add one system-prompt section telling the model to search before answering questions about the user's projects.
> Keep the pure logic in a separate file with no DSH imports and write `node --test` tests.

**tool-ledger**
> Create a DSH plugin named `tool-ledger`. In `tools/pre-execute`, deny a tool once it has been called N times in an agent session (configurable
> default and per-tool limits), returning `{kind:'deny', reason:'QUOTA: ...'}` rather than throwing; reserve the quota before awaiting `next()` and refund it
> if a later policy denies. In `tools/result`, append a JSONL audit row with secrets in the arguments redacted. Register a `tool_ledger` tool that reports
> usage and is exempt from quotas. Write `node --test` tests including a parallel-call race.

## What I would verify next
- Does DSH invoke a `tools/pre-execute` listener once per call, and does a result for a denied call arrive at `tools/result` (the code assumes yes and de-duplicates)?
- Is `exec.agent.id` the sub-agent's own id for sub-agent calls (the quota assumes per-session buckets)?
- Does a `name:` that is an absolute `.js` path load without a transpile step? (The docs only show a `.ts` path; Node can import `.js` natively.)
