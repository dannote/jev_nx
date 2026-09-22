# Changelog

## 0.1.1 (2026-09-22)

- `preallocate_params: true` copied the parameters to the backend once per batch key, eighteen
  times with the default buckets. They are now copied once per serving process and shared.
- Client-side tensors are built on the binary backend, so caller processes never allocate on the
  accelerator; the serving moves each batch.
- Batches are padded to `batch_size` only with `compile: true`. Without it, each shape compiles for
  the batch sizes that occur, as Bumblebee's servings do.
- `Jev.Nx.Laya.load/1` no longer rescues every exception into `{:error, _}`. Missing files and
  unreadable checkpoints are errors; bugs raise.

## 0.1.0 (2026-09-22)

- `Jev.Nx`: a `Jev.Backend` that answers from an in-process model behind an `Nx.Serving`.
- `Jev.Nx.Model`: the behaviour a model implements, encode, batch, forward, decode, so
  `Jev.Nx.Serving` can pad to shape buckets, batch across callers, and compile any of them the
  same way.
- `Jev.Nx.Laya`: Laya's English, multilingual, and typed-decisions checkpoints, the encoder through
  Bumblebee and the decision head in `Nx.Defn`. Sequences match the reference token for token
  and probabilities to about 1.0e-6.
