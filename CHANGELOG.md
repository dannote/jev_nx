# Changelog

## 0.1.0 (unreleased)

- `Jev.Nx`: a `Jev.Backend` that answers from an in-process model behind an `Nx.Serving`.
- `Jev.Nx.Model`: the behaviour a model implements, encode, batch, forward, decode, so
  `Jev.Nx.Serving` can pad to shape buckets, batch across callers, and compile any of them the
  same way.
- `Jev.Nx.Laya`: Laya's English, multilingual, and typed-decisions checkpoints, the encoder through
  Bumblebee and the decision head in `Nx.Defn`. Sequences match the reference token for token
  and probabilities to about 1.0e-6.
