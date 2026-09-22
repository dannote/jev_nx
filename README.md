# Jev.Nx

Open decision models as a [Jev](https://hexdocs.pm/jev) backend, running
in-process on [Nx](https://hexdocs.pm/nx).

[TypeSafe Jev](https://docs.typesafe.ai) made typed decisions with calibrated
probabilities a model class of its own, and open models followed it. This
package runs them where your Elixir code runs and answers a `Jev.Server`
exactly as Jev does, so the same `handle_answer/3` clauses, thresholds, and
telemetry apply whichever model answered.

```elixir
children = [
  {Nx.Serving, serving: Jev.Nx.serving(Jev.Nx.Laya), name: MyApp.Laya, batch_size: 16}
]

config :jev, backend: Jev.Nx
config :jev_nx, serving: MyApp.Laya
```

Or per request, which is how a cascade asks the local model first and
escalates to Jev when it is unsure:

```elixir
def handle_call({:classify, text}, from, s),
  do: {:reply, {{:local, from, text}, text, [kind: @kinds], [backend: Jev.Nx]}, s}

def handle_answer(%{confidence: %{kind: c}}, {:local, from, text}, s) when c < 0.7,
  do: {:reply, {{:jev, from, text}, text, kind: @kinds}, s}

def handle_answer(%{kind: k}, {_stage, from, _text}, s), do: done(from, k, s)
```

## Models

| Model | Weights | Backbone | Status |
| --- | --- | --- | --- |
| [Laya](https://huggingface.co/convaiinnovations/laya) | Apache 2.0, 421M | ModernBERT-large | `Jev.Nx.Laya`; sequences and probabilities verified against the reference |

`Jev.Nx.Model` is the contract for the next one: encode a question into a
marked token sequence, batch, run, and read a distribution per question out of
the outputs. `Jev.Nx.Serving` does the padding, batching, and compilation for
every model that implements it.

## Installation

```elixir
def deps do
  [
    {:jev_nx, "~> 0.1"},
    {:exla, "~> 0.13"}           # or {:emily, "~> 1.0"} for Metal on Apple Silicon
  ]
end
```

```elixir
config :nx, default_backend: EXLA.Backend
config :nx, :default_defn_options, compiler: EXLA
```

This package does not choose an Nx backend. Parameters load onto
`Nx.default_backend/0`, and the serving compiles with `Nx.Defn.default_options/0`
unless given `defn_options`. The checkpoint downloads from the Hub on first
load, and `Jev.Nx.serving/3` loads it in the calling process, so in a child
spec the application waits for it; pass `repository: {:local, dir}` for a copy
on disk.

## Shapes and batching

Questions from concurrent callers are batched by shape. Sequences pad to the
smallest bucket in `sequence_length` that fits, options to the smallest in
`option_slots`, and each pair of buckets is one compiled program, built on
first use or at start with `compile: true`:

```elixir
Jev.Nx.serving(Jev.Nx.Laya, [checkpoint: :english],
  sequence_length: [128, 256, 512],
  option_slots: [4, 8, 16, 32, 64, 128],
  batch_size: 16,
  compile: true
)
```

## Testing

```sh
mix test                          # without the checkpoint
mix test --include model          # with it in ~/.cache/jev_nx/laya, or JEV_NX_LAYA_DIR
```

`test/fixtures/golden.py` records the reference implementation's sequences
and answers for the golden cases.
