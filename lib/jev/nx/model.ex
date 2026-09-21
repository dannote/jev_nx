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

  `c:batch/4` pads items to a `sequence_length` and a number of
  `option_slots`, and `c:template/4` describes the same tensors for ahead-of-time
  compilation. Both come from the serving's bucket options. The same shapes
  reach `c:forward/1` however the batch was assembled.
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

  @doc "Input templates for `batch_size` items of the given shape, for compilation."
  @callback template(model(), pos_integer(), pos_integer(), pos_integer()) :: inputs()

  @doc "The parameters `c:forward/1` takes, as one term the serving can move to a backend."
  @callback params(model()) :: Nx.Container.t()

  @doc "The network as a function of parameters and inputs, suitable for `Nx.Defn.jit/2`."
  @callback forward(model()) :: (Nx.Container.t(), inputs() -> Nx.Container.t())

  @doc "Reads one `Jev.Wire.Answer` per item out of the batch outputs, in item order."
  @callback decode(model(), [item()], Nx.Container.t()) :: [Jev.Wire.Answer.t()]
end
