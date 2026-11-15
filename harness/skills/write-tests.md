---
name: write-tests
description: How to write small, deterministic pytest tests for a function
---
1. One behaviour per test; name it `test_<function>_<situation>`.
2. Cover: a normal case, an empty/zero case, and one invalid input that should raise.
3. No network, no sleeping, no randomness without a fixed seed.
4. Use `pytest.mark.parametrize` for tables of inputs and expected outputs.
Write the tests to a `test_*.py` file next to the code, then run them with run_python.
