defmodule Jev.Nx.Laya do
  @moduledoc """
  [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations,
  an Apache 2.0 decision model: a fine-tuned ModernBERT encoder and a decision
  head that scores one `[MASK]` per option. 421M parameters, 512 tokens of
  context, and among the best open scores on JevBench.

  The encoder loads through Bumblebee, the head is `Jev.Nx.Laya.Head`, and
  the input layout is `Jev.Nx.Laya.Sequence`. The checkpoint downloads from
  the Hub on first load.

      serving = Jev.Nx.serving(Jev.Nx.Laya, checkpoint: :english)

  ## Options

    * `:checkpoint` - `:english` (default), `:multilingual`, or `:typed_decisions`
    * `:repository` - a Bumblebee repository, default `{:hf, "convaiinnovations/laya"}`;
      `{:local, dir}` for a downloaded copy
    * `:type` - encoder parameter type, such as `:bf16`; default the checkpoint's

  ## What comes back

  Answers are calibrated with the checkpoint's temperature table, keyed by
  question type and option count, as the reference does. A choice is the
  label with the highest probability, a score the probability-weighted level,
  a yes/no the probability of the `true` option. Confidence is left to
  `Jev.reply/3`, so it is TypeSafe's definition here as everywhere.

  The context is short: after the question and options, roughly 300 tokens of
  state fit, and the rest is cut. A question whose options overflow the
  192-token head budget raises `ArgumentError`.
  """

  @behaviour Jev.Nx.Model

  alias Bumblebee.HuggingFace.Hub
  alias Jev.Nx.Laya.{Head, Sequence}
  alias Jev.Wire

  @repository {:hf, "convaiinnovations/laya"}
  @checkpoints %{english: "", multilingual: "multilingual", typed_decisions: "typed-decisions"}

  defstruct [:checkpoint, :encoder, :encoder_params, :encode, :special, :head_params, :config]

  @type t :: %__MODULE__{}

  @impl Jev.Nx.Model
  def load(opts \\ []) do
    opts = Keyword.validate!(opts, checkpoint: :english, repository: @repository, type: nil)
    checkpoint = opts[:checkpoint]

    subdir =
      Map.get(@checkpoints, checkpoint) ||
        raise ArgumentError, "unknown checkpoint #{inspect(checkpoint)}"

    repository = opts[:repository]

    with {:ok, config} <- config(repository, subdir),
         {:ok, spec} <- Bumblebee.load_spec(at(repository, [subdir, "encoder"])),
         spec = Bumblebee.configure(spec, architecture: :base),
         {:ok, encoder} <- encoder(repository, subdir, spec, opts[:type]),
         {:ok, tokenizer} <-
           Bumblebee.load_tokenizer(at(repository, [subdir, "tokenizer"]), type: :modernbert),
         {:ok, head_params} <- head_params(repository, subdir) do
      {:ok,
       %__MODULE__{
         checkpoint: checkpoint,
         encoder: encoder.model,
         encoder_params: encoder.params,
         encode: Sequence.encoder(tokenizer),
         special: Sequence.special(tokenizer),
         head_params: head_params,
         config: config
       }}
    end
  rescue
    exception -> {:error, exception}
  end

  @impl Jev.Nx.Model
  def name(%__MODULE__{checkpoint: checkpoint}) do
    "laya-" <> (checkpoint |> Atom.to_string() |> String.replace("_", "-"))
  end

  @impl Jev.Nx.Model
  def max_sequence_length(%__MODULE__{config: config}), do: config["max_len"]

  @impl Jev.Nx.Model
  def encode(%__MODULE__{} = model, state, question) do
    limits = [max_len: model.config["max_len"], head_max_len: model.config["head_max_len"]]
    sequence = Sequence.build(model.encode, model.special, state, question, limits)

    if length(sequence.markers) != length(sequence.labels) do
      raise ArgumentError,
            "options of #{inspect(question)} do not fit in Laya's #{limits[:head_max_len]} head tokens"
    end

    Map.merge(sequence, %{
      tokens: length(sequence.ids),
      options: length(sequence.labels),
      question: question
    })
  end

  @impl Jev.Nx.Model
  def batch(%__MODULE__{special: %{pad: pad}}, items, length, slots) do
    %{
      "input_ids" => tensor(items, & &1.ids, length, pad, :u32),
      "attention_mask" => tensor(items, &List.duplicate(1, length(&1.ids)), length, 0, :u32),
      "marker_pos" => tensor(items, & &1.markers, slots, 0, :s32),
      "marker_mask" => tensor(items, &List.duplicate(1, length(&1.markers)), slots, 0, :u8),
      "qtype" => Nx.tensor(Enum.map(items, & &1.qtype), type: :s32)
    }
  end

  defp tensor(items, fun, width, fill, type) do
    items
    |> Enum.map(fn item ->
      fun.(item) |> then(&(&1 ++ List.duplicate(fill, width - length(&1))))
    end)
    |> Nx.tensor(type: type)
  end

  @impl Jev.Nx.Model
  def template(%__MODULE__{}, batch_size, length, slots) do
    %{
      "input_ids" => Nx.template({batch_size, length}, :u32),
      "attention_mask" => Nx.template({batch_size, length}, :u32),
      "marker_pos" => Nx.template({batch_size, slots}, :s32),
      "marker_mask" => Nx.template({batch_size, slots}, :u8),
      "qtype" => Nx.template({batch_size}, :s32)
    }
  end

  @impl Jev.Nx.Model
  def params(%__MODULE__{encoder_params: encoder, head_params: head}),
    do: %{encoder: encoder, head: head}

  @impl Jev.Nx.Model
  def forward(%__MODULE__{encoder: encoder}) do
    {_init, predict} = Axon.build(encoder)

    fn params, inputs ->
      encoder_inputs = Map.take(inputs, ["input_ids", "attention_mask"])
      hidden_state = predict.(params.encoder, encoder_inputs).hidden_state

      Head.forward(
        hidden_state,
        inputs["attention_mask"],
        inputs["marker_pos"],
        inputs["marker_mask"],
        inputs["qtype"],
        params.head
      )
    end
  end

  @impl Jev.Nx.Model
  def decode(%__MODULE__{config: config}, items, logits) do
    logits
    |> Nx.to_list()
    |> Enum.zip_with(items, fn row, item ->
      probabilities = row |> Enum.take(item.options) |> calibrate(item, config) |> softmax()
      answer(item.question, item.labels, probabilities)
    end)
  end

  defp answer(%Jev.Choice{}, labels, probabilities) do
    {label, _} = labels |> Enum.zip(probabilities) |> Enum.max_by(&elem(&1, 1))

    %Wire.Answer{
      type: :choice,
      choice: label,
      probabilities: labels |> Enum.zip(probabilities) |> Map.new()
    }
  end

  defp answer(%Jev.Score{criteria: levels}, labels, probabilities) do
    %Wire.Answer{
      type: :score,
      score: probabilities |> Enum.with_index() |> Enum.map(fn {p, i} -> p * i end) |> Enum.sum(),
      legend: labels |> Enum.zip(Enum.map(levels, &Sequence.render/1)) |> Map.new(),
      probabilities: labels |> Enum.zip(probabilities) |> Map.new()
    }
  end

  defp answer(%Jev.Noul{}, _labels, [_no, yes]), do: %Wire.Answer{type: :noul, noul: yes}

  # The checkpoint's temperature for this question type and option count.
  defp calibrate(logits, %{question: %{type: type}, qtype: qtype, options: k}, config) do
    size =
      cond do
        k <= 2 -> "2"
        k <= 5 -> "3-5"
        k <= 10 -> "6-10"
        true -> "11+"
      end

    temperature =
      config["temperature_by_options"]["#{type}:#{size}"] || Enum.at(config["temperature"], qtype) ||
        1.0

    Enum.map(logits, &(&1 / temperature))
  end

  defp softmax(logits) do
    top = Enum.max(logits)
    exps = Enum.map(logits, &:math.exp(&1 - top))
    total = Enum.sum(exps)
    Enum.map(exps, &(&1 / total))
  end

  # Loading

  defp config(repository, subdir) do
    with {:ok, path} <- file(repository, [subdir, "rl_agent_config.json"]),
         {:ok, json} <- File.read(path) do
      JSON.decode(json)
    end
  end

  defp encoder(repository, subdir, spec, type) do
    opts = [
      spec: spec,
      params_filename: "model.safetensors",
      safetensors_reader: &encoder_tensors/1
    ]

    opts = if type, do: Keyword.put(opts, :type, type), else: opts
    Bumblebee.load_model(at(repository, [subdir]), opts)
  end

  # The checkpoint is one safetensors file with the encoder under "encoder."
  # and the head beside it.
  defp encoder_tensors(path) do
    for {"encoder." <> name, tensor} <- Safetensors.read!(path, lazy: true),
        into: %{},
        do: {name, tensor}
  end

  defp head_params(repository, subdir) do
    with {:ok, path} <- file(repository, [subdir, "model.safetensors"]) do
      tensors = Safetensors.read!(path, lazy: true)

      pair = fn prefix ->
        %{
          weight: tensor!(tensors, prefix <> ".weight"),
          bias: tensor!(tensors, prefix <> ".bias")
        }
      end

      layer = fn index ->
        prefix = "head.layers.#{index}"

        %{
          in_proj: %{
            weight: tensor!(tensors, prefix <> ".self_attn.in_proj_weight"),
            bias: tensor!(tensors, prefix <> ".self_attn.in_proj_bias")
          },
          out_proj: pair.(prefix <> ".self_attn.out_proj"),
          norm_1: pair.(prefix <> ".norm1"),
          norm_2: pair.(prefix <> ".norm2"),
          linear_1: pair.(prefix <> ".linear1"),
          linear_2: pair.(prefix <> ".linear2")
        }
      end

      params = %{
        type_embedding: tensor!(tensors, "type_emb.weight"),
        layer_0: layer.(0),
        layer_1: layer.(1),
        scorer: %{norm: pair.("scorer.0"), dense: pair.("scorer.1"), output: pair.("scorer.3")}
      }

      {:ok, Nx.backend_transfer(params, Nx.default_backend())}
    end
  end

  defp tensor!(tensors, name) do
    tensors |> Map.fetch!(name) |> Nx.to_tensor() |> Nx.as_type(:f32)
  end

  # Repositories

  defp at({:hf, repo}, parts), do: at({:hf, repo, []}, parts)

  defp at({:hf, repo, opts}, parts) do
    case Path.join([opts[:subdir] || "" | parts]) do
      "" -> {:hf, repo, Keyword.delete(opts, :subdir)}
      subdir -> {:hf, repo, Keyword.put(opts, :subdir, subdir)}
    end
  end

  defp at({:local, dir}, parts), do: {:local, Path.join([dir | parts])}

  defp file({:local, dir}, parts), do: {:ok, Path.join([dir | parts])}

  defp file({:hf, repo}, parts), do: file({:hf, repo, []}, parts)

  defp file({:hf, repo, opts}, parts) do
    filename = Path.join([opts[:subdir] || "" | parts])

    Hub.cached_download(
      Hub.file_url(repo, filename, opts[:revision]),
      Keyword.take(opts, [:cache_dir, :auth_token])
    )
  end
end
