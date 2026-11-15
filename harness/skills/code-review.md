---
name: code-review
description: Checklist for reviewing a Python file for bugs, edge cases and clarity
---
Review in this order and report only what you verified by reading the code:
1. Correctness: off-by-one errors, unhandled None/empty input, wrong exception types.
2. Edge cases: empty collections, very large inputs, unicode, negative numbers.
3. Security: file paths from input, shell/eval use, secrets in source.
4. Clarity: names, dead code, functions doing more than one job.
Output a numbered list: `file:line - problem - suggested fix`. If nothing is wrong, say so plainly.
