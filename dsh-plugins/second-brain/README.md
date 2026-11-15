# second-brain

A DeepSeek Harness plugin that gives the agent a persistent **markdown vault** of notes linked with `[[wikilinks]]`.

| tool | what it does |
|---|---|
| `brain_save(title, content, tags?, mode?)` | create/update a note (`mode: "append"` extends it). Same title = same note. Returns the links it found and whether each target exists. |
| `brain_read(title)` | the note, its outgoing links (dangling ones flagged) and its **backlinks** |
| `brain_search(query, limit?)` | ranked keyword search (title > tags > body frequency) with a snippet |
| `brain_list(tag?)` | every note with link counts; `orphan: true` means nothing links in or out |

It also adds one system-prompt section (`ctx.systemPrompt.section`) telling the model when to search and save.

## Design
- `src/vault.js` is pure Node (no DSH imports): slugging, wikilink parsing, atomic writes (temp file + rename), search, backlinks.
- `src/index.js` is the thin Cordis wiring: `export const name / inject / apply(ctx, config)`. Tools are registered as raw
  JSON-Schema `ToolDefinition`s, so the package imports nothing from DSH and its tests run with plain `node --test`.
- Safety: titles become slugs `[a-z0-9-]{1,80}` so a title cannot name a path (`../../x` becomes `x`); content <= 100 000 chars;
  <= 2000 notes; hand-edited files with broken front matter stay readable.

## Config (in the patch row)
```yaml
config:
  root: /absolute/path/to/vault   # default: <cwd>/.second-brain
  maxSearchResults: 5             # 1-20
  injectPrompt: true
```

## Tests
`node --test test/` -> 19 tests (slugging, links, backlinks, append, search ranking, traversal, atomic writes, plugin contract with a fake `ctx`).

## Not verified
Loading inside a running DSH was **not** done by the author (see `../NOTES.md`). Hand-written config validation is used instead
of an exported Schemastery `Config` to stay dependency-free.
