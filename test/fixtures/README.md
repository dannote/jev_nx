# Fixtures

`laya_golden.json` is what Laya's reference implementation answers for the
cases in `golden.py`: the token sequence and marker positions per question, and
the probabilities per answer. `Jev.Nx.LayaTest` replays those cases through
every runtime and compares.

The reference is PyTorch, shipped inside the checkpoint repository, so the
generator is a Python script. It declares its own dependencies and runs on its
own:

```sh
uv run test/fixtures/golden.py
```

Rerun it when the checkpoint changes or a case is added, and commit the JSON
with it. Nothing in the Elixir suite runs the script, and neither file is in
the published package.
