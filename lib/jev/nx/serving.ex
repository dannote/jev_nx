defmodule Jev.Nx.Serving do
  @moduledoc """
  An `Nx.Serving` for any `Jev.Nx.Model`.

  The serving takes `{state, questions}`, with the questions as
  `Jev.questions/1` returns them, and gives back a `Jev.Wire.Response`, the
  same shape a `/v1/systemone` server sends. Questions from concurrent callers
  are batched together by shape, so a `Jev.Server` with many requests in
  flight fills the accelerator instead of queueing forward passes.

  Put it in a supervision tree and give it a name:

      children = [
        {Nx.Serving,
         serving: Jev.Nx.Serving.new(Jev.Nx.Laya, [checkpoint: :english], sequence_length: [128, 512]),
         name: MyApp.Laya,
         batch_size: 16,
         batch_timeout: 20}
      ]

  ## Shapes and compilation

  Items are padded to the smallest bucket that fits, so the compiler sees a
  fixed set of shapes. `:sequence_length` and `:option_slots` are the bucket
  lists; each pair is one batch key, compiled on first use, or at start with
  `compile: true` when `:batch_size` is also given. Bucket lists that are too
  fine compile more, too coarse pad more; the defaults suit Laya.

  ## Options

    * `:sequence_length` - token buckets, default `[128, 256, model max]`,
      clipped to the model's maximum
    * `:option_slots` - option-count buckets, default `[4, 8, 16, 32, 64, 128]`
    * `:batch_size` - the maximum batch the serving runs; needed for `:compile`
    * `:compile` - compile every shape when the serving starts, default `false`
    * `:defn_options` - passed to `Nx.Defn.jit/2` and `Nx.Defn.compile/3`
    * `:preallocate_params` - move parameters to the compiler's backend at
      start, default `false`
  """

  alias Jev.Wire

  @option_slots [4, 8, 16, 32, 64, 128]

  @doc """
  Loads `module` with `model_opts` and builds its serving; see the module docs
  for `opts`. Raises if the model fails to load.
  """
  @spec new(module(), keyword(), keyword()) :: Nx.Serving.t()
  def new(module, model_opts \\ [], opts \\ []) when is_atom(module) do
    case module.load(model_opts) do
      {:ok, model} -> build(model, opts)
      {:error, exception} -> raise exception
    end
  end

  @doc "Builds the serving for an already loaded model; see the module docs for `opts`."
  @spec build(Jev.Nx.Model.model(), keyword()) :: Nx.Serving.t()
  def build(%module{} = model, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :sequence_length,
        :batch_size,
        option_slots: @option_slots,
        compile: false,
        defn_options: [],
        preallocate_params: false
      ])

    max = module.max_sequence_length(model)
    lengths = buckets(opts[:sequence_length] || [128, 256, max], max)
    slots = buckets(opts[:option_slots], nil)
    keys = for length <- lengths, slot <- slots, do: {:shape, length, slot}
    batch_size = opts[:batch_size]

    if opts[:compile] and is_nil(batch_size) do
      raise ArgumentError, "compile: true needs :batch_size"
    end

    Nx.Serving.new(&runner(model, &1, &2, batch_size, opts), opts[:defn_options])
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: keys)
    |> Nx.Serving.client_preprocessing(&preprocess(model, &1, lengths, slots))
    |> Nx.Serving.client_postprocessing(&postprocess(model, &1, &2))
  end

  # One compiled program per batch key, built when the serving first sees it.
  defp runner(model, {:shape, length, slot}, defn_options, batch_size, opts) do
    params = params(model, opts[:preallocate_params], defn_options)
    forward = compile(model, batch_size, length, slot, opts[:compile], defn_options)

    fn batch ->
      batch = if batch_size, do: Nx.Batch.pad(batch, batch_size - batch.size), else: batch
      forward.(params, batch) |> Nx.backend_transfer(Nx.BinaryBackend)
    end
  end

  defp params(%module{} = model, preallocate?, defn_options) do
    params = module.params(model)

    if preallocate? do
      Nx.backend_copy(params, Nx.Defn.to_backend(defn_options))
    else
      params
    end
  end

  defp compile(%module{} = model, batch_size, length, slot, true, defn_options) do
    template = module.template(model, batch_size, length, slot)
    params = Nx.Defn.Composite.traverse(module.params(model), &Nx.to_template/1)
    Nx.Defn.compile(module.forward(model), [params, template], defn_options)
  end

  defp compile(%module{} = model, _batch_size, _length, _slot, false, defn_options) do
    Nx.Defn.jit(module.forward(model), defn_options)
  end

  defp preprocess(%module{} = model, {state, questions}, lengths, slots) do
    {names, items} =
      questions
      |> Enum.map(fn {name, question} -> {name, module.encode(model, state, question)} end)
      |> Enum.unzip()

    length = bucket(lengths, items |> Enum.map(& &1.tokens) |> Enum.max())
    slot = bucket(slots, items |> Enum.map(& &1.options) |> Enum.max())

    batch =
      [module.batch(model, items, length, slot)]
      |> Nx.Batch.concatenate()
      |> Nx.Batch.key({:shape, length, slot})

    {batch, {names, items}}
  end

  defp postprocess(%module{} = model, {outputs, _metadata}, {names, items}) do
    answers = module.decode(model, items, outputs)

    %Wire.Response{
      model: module.name(model),
      answers: names |> Enum.map(&Atom.to_string/1) |> Enum.zip(answers) |> Map.new(),
      usage: %Wire.Usage{input_tokens: items |> Enum.map(& &1.tokens) |> Enum.sum()}
    }
  end

  defp buckets(list, max) do
    list
    |> Enum.map(fn n -> if max, do: min(n, max), else: n end)
    |> Enum.uniq()
    |> Enum.sort()
  end

  # The smallest bucket that fits, or the largest when nothing does; the model
  # is expected to have truncated to its maximum already.
  defp bucket(buckets, size) do
    Enum.find(buckets, List.last(buckets), &(&1 >= size))
  end
end
