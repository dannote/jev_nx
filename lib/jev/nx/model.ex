defmodule Jev.Nx.Model do
  @moduledoc """
  A decision model that runs on Nx.

  Every open Jev-class model does the same three things: turn a question over
  a state into a token sequence with marked option positions, run a batch of
  those through a network, and read a distribution per question out of the
  outputs at the marks. This behaviour names those steps so `Jev.Nx.Serving`
  can batch, pad, compile, and cache any model the same way, and so `Jev.Nx`
  can be one `Jev.Backend` for all of them.

  A model is a struct returned by `c:load/1`, holding whatever the callbacks
  need: an encoder, its parameters, a tokenizer, a calibration table.

  ## Items

  `c:encode/3` returns one item per question. An item is a map the model
  owns, with two required keys the serving uses to pick a padding bucket:

    * `:tokens` - the sequence length before padding, also reported as usage
    * `:options` - how many options the question has, so a batch can pad every
      item to the same number of option slots

  ## Shapes

  `c:batch/4` pads items to a sequence length and a number of option slots,
  both from the serving's bucket options, and `c:init/4` prepares one function
  per shape. A model that runs on `Nx.Defn` implements `c:init/4` with
  `Jev.Nx.Defn.runner/3`; anything else returns its own function.
  """

  @type model :: struct()

  @typedoc "One encoded question. Models add their own keys."
  @type item :: %{
          required(:tokens) => non_neg_integer(),
          required(:options) => pos_integer(),
          optional(atom()) => term()
        }

  @typedoc "Named input tensors, as an Axon model takes them."
  @type inputs :: %{String.t() => Nx.Tensor.t()}

  @typedoc "A padded batch shape: tokens per sequence and option slots per question."
  @type shape :: {:shape, pos_integer(), pos_integer()}

  @doc "Loads a model. Options are the model's own, such as a checkpoint name."
  @callback load(keyword()) :: {:ok, model()} | {:error, Exception.t()}

  @doc "The name reported as `model` in every reply, such as `\"laya-english\"`."
  @callback name(model()) :: String.t()

  @doc "The longest sequence the model accepts. Buckets are clipped to it."
  @callback max_sequence_length(model()) :: pos_integer()

  @doc "Encodes one question over the state into an item."
  @callback encode(model(), Jev.entry(), Jev.question()) :: item()

  @doc "Pads items into input tensors of the given sequence length and option slots."
  @callback batch(model(), [item()], pos_integer(), pos_integer()) :: inputs()

  @doc """
  Prepares the model to run one batch shape.

  Called in the serving process once per shape when it starts, and mirrors
  `Nx.Serving`'s own init: given the shape and the `defn_options` the serving
  was built with, return a function from input tensors to outputs. A model on
  `Nx.Defn` compiles or jits here, which `Jev.Nx.Defn.runner/3` does for it; a
  model on another runtime, such as an ONNX session, returns a function that
  calls it.

  The function receives an `Nx.Batch`, which a jitted function takes directly.
  A runtime that needs the tensors themselves calls `inputs/1` on it.

  `batch_size` is the serving's maximum, or `nil` when it has none. Inputs
  arrive with exactly that many rows only when the serving pads them.
  """
  @callback init(model(), shape(), batch_size :: pos_integer() | nil, defn_options :: keyword()) ::
              (Nx.Batch.t() -> Nx.Container.t())

  @doc "Reads one `Jev.Wire.Answer` per item out of the batch outputs, in item order."
  @callback decode(model(), [item()], Nx.Container.t()) :: [Jev.Wire.Answer.t()]

  @doc """
  The input tensors of a batch, for a runtime that is not `Nx.Defn`.

  A batch is a description of tensors to stack, concatenate, and pad, which
  the compiler would otherwise fuse into the computation. This materializes
  it, so the tensors exist before the model runs.
  """
  @spec inputs(Nx.Batch.t()) :: inputs()
  def inputs(%Nx.Batch{} = batch), do: Nx.Defn.jit_apply(&Function.identity/1, [batch])
end
