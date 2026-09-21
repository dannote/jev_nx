defmodule Jev.Nx.Laya.Sequence do
  @moduledoc """
  Laya's input layout, token for token as the reference implementation builds it.

      [CLS] <type> question: <instructions> [SEP] [MASK] <option> [MASK] <option> ... [SEP] <state> [SEP]

  Every option starts with a `[MASK]` whose position is a marker; the head
  scores the hidden state at each marker. Options are capped at 48 tokens
  each, the question and options together at `head_max_len`, and the whole
  sequence at `max_len`, with the state truncated from the right. When the
  options alone overflow the head budget, each is shrunk evenly.

  Option order is the label order: a choice's labels sorted as strings, which
  is also how Jev encodes criteria on the wire; a score's levels in order; and
  `false` before `true` for a yes/no question, so the second marker is the
  probability of yes.

  Structured data is rendered the way the reference does in Python: a state
  or instructions with `json.dumps`, a criterion with `str()`. `json/1` and
  `render/1` reproduce those for JSON-shaped data, with map keys sorted, which
  is the order a Python server sees them in after Jev has encoded a request.
  """

  alias Tokenizers.{Encoding, Tokenizer}

  @qtypes %{choice: 0, score: 1, noul: 2}
  @option_tokens 48
  @min_head_tokens 8
  @min_option_budget 16
  @min_option_tokens 4

  @typedoc "Token ids to encode, or any of the tokenizer's other callables."
  @type encode :: (String.t() -> [non_neg_integer()])

  @type special :: %{cls: non_neg_integer(), sep: non_neg_integer(), mask: non_neg_integer()}

  @doc """
  Builds the sequence for one question over `state`.

  Returns the token ids, the marker positions in option order, the rendered
  option labels, and the question type index.
  """
  @spec build(encode(), special(), Jev.entry(), Jev.question(), keyword()) :: %{
          ids: [non_neg_integer()],
          markers: [non_neg_integer()],
          labels: [String.t()],
          qtype: 0..2
        }
  def build(encode, special, state, question, opts) do
    max_len = Keyword.fetch!(opts, :max_len)
    head_max_len = Keyword.fetch!(opts, :head_max_len)
    type = question.type

    head_ids = encode.("#{type} question: #{clean(instructions(question))}")

    option_ids =
      for option <- options(question) do
        [special.mask | encode.(" " <> clean(option)) |> Enum.take(@option_tokens)]
      end

    budget = head_max_len - total(option_ids)

    {option_ids, budget} =
      if budget < @min_option_budget do
        per =
          max(
            @min_option_tokens,
            div(head_max_len - @min_option_budget, max(1, length(option_ids)))
          )

        option_ids = Enum.map(option_ids, &Enum.take(&1, per))
        {option_ids, head_max_len - total(option_ids)}
      else
        {option_ids, budget}
      end

    prefix = [special.cls | Enum.take(head_ids, max(@min_head_tokens, budget))] ++ [special.sep]

    {markers, options_ids} =
      Enum.map_reduce(option_ids, length(prefix), fn option, position ->
        {position, position + length(option)}
      end)
      |> then(fn {markers, _} -> {markers, List.flatten(option_ids)} end)

    head = prefix ++ options_ids ++ [special.sep]
    room = max(0, max_len - length(head) - 1)
    state_ids = state |> serialize() |> clean() |> encode.() |> Enum.take(room)
    ids = Enum.take(head ++ state_ids ++ [special.sep], max_len)

    %{
      ids: ids,
      markers: Enum.filter(markers, &(&1 < max_len)),
      labels: labels(question),
      qtype: Map.fetch!(@qtypes, type)
    }
  end

  @doc "Wraps a Bumblebee tokenizer as the `encode` function `build/5` takes."
  @spec encoder(Bumblebee.Tokenizer.t()) :: encode()
  def encoder(%{native_tokenizer: native}) do
    fn text ->
      {:ok, encoding} = Tokenizer.encode(native, text, add_special_tokens: false)
      Encoding.get_ids(encoding)
    end
  end

  @doc "The `[CLS]`, `[SEP]`, `[MASK]`, and `[PAD]` ids of a Bumblebee tokenizer."
  @spec special(Bumblebee.Tokenizer.t()) :: %{
          cls: integer(),
          sep: integer(),
          mask: integer(),
          pad: integer()
        }
  def special(%{native_tokenizer: native}) do
    vocab = Tokenizer.get_vocab(native)

    for {name, token} <- [cls: "[CLS]", sep: "[SEP]", mask: "[MASK]", pad: "[PAD]"], into: %{} do
      {name, Map.fetch!(vocab, token)}
    end
  end

  @doc "The option labels in the order their markers appear."
  @spec labels(Jev.question()) :: [String.t()]
  def labels(%Jev.Choice{criteria: criteria}) do
    criteria |> Map.keys() |> Enum.map(&Atom.to_string/1) |> Enum.sort()
  end

  def labels(%Jev.Score{criteria: levels}),
    do: Enum.map(0..(length(levels) - 1), &Integer.to_string/1)

  def labels(%Jev.Noul{}), do: ["false", "true"]

  @doc "The option texts in marker order, as the reference renders them."
  @spec options(Jev.question()) :: [String.t()]
  def options(%Jev.Choice{criteria: criteria} = question) do
    for label <- labels(question) do
      case Map.fetch!(criteria, String.to_existing_atom(label)) do
        empty when empty in [nil, ""] -> label
        description -> "#{label}: #{render(description)}"
      end
    end
  end

  def options(%Jev.Score{criteria: levels}) do
    Enum.with_index(levels, fn level, index -> "level #{index}: #{render(level)}" end)
  end

  def options(%Jev.Noul{criteria: criteria}) do
    criteria = criteria || %{}

    [
      "false: " <> (criterion(criteria[false]) || "no, the statement does not hold"),
      "true: " <> (criterion(criteria[true]) || "yes, the statement holds")
    ]
  end

  defp criterion(empty) when empty in [nil, ""], do: nil
  defp criterion(value), do: render(value)

  @doc """
  Python's `json.dumps(value, ensure_ascii=False)` of JSON-shaped data, which
  is how the reference serializes a state or structured instructions: a space
  after every comma and colon.

      iex> Jev.Nx.Laya.Sequence.json(%{title: "Crash", tags: ["ios", 17]})
      ~s({"tags": ["ios", 17], "title": "Crash"})
  """
  @spec json(term()) :: String.t()
  def json(value) when is_binary(value) or is_number(value) or is_boolean(value) or is_nil(value),
    do: JSON.encode!(value)

  def json(value) when is_atom(value), do: JSON.encode!(Atom.to_string(value))
  def json(value) when is_list(value), do: "[" <> Enum.map_join(value, ", ", &json/1) <> "]"

  def json(%{} = value) do
    "{" <>
      Enum.map_join(sorted(value), ", ", fn {k, v} ->
        JSON.encode!(to_string(k)) <> ": " <> json(v)
      end) <>
      "}"
  end

  @doc """
  Python's `str()` of JSON-shaped data: strings as they are, everything else
  as Python's repr, which is what the reference puts in an option text.

      iex> Jev.Nx.Laya.Sequence.render(%{what: "Charges, refunds", examples: ["Charged twice"]})
      "{'examples': ['Charged twice'], 'what': 'Charges, refunds'}"
  """
  @spec render(term()) :: String.t()
  def render(value) when is_binary(value), do: value
  def render(value), do: repr(value)

  defp repr(value) when is_binary(value) do
    if String.contains?(value, "'") and not String.contains?(value, "\""),
      do: "\"" <> escape(value) <> "\"",
      else: "'" <> String.replace(escape(value), "'", "\\'") <> "'"
  end

  defp repr(true), do: "True"
  defp repr(false), do: "False"
  defp repr(nil), do: "None"
  defp repr(value) when is_atom(value), do: repr(Atom.to_string(value))
  defp repr(value) when is_integer(value), do: Integer.to_string(value)
  defp repr(value) when is_float(value), do: Float.to_string(value)
  defp repr(value) when is_list(value), do: "[" <> Enum.map_join(value, ", ", &repr/1) <> "]"

  defp repr(%{} = value) do
    "{" <>
      Enum.map_join(sorted(value), ", ", fn {k, v} -> repr(to_string(k)) <> ": " <> repr(v) end) <>
      "}"
  end

  defp sorted(%_{} = struct), do: struct |> Map.from_struct() |> sorted()
  defp sorted(map), do: Enum.sort_by(map, fn {k, _} -> to_string(k) end)

  defp escape(string) do
    string
    |> String.replace("\\", "\\\\")
    |> String.replace("\n", "\\n")
    |> String.replace("\t", "\\t")
    |> String.replace("\r", "\\r")
  end

  defp instructions(%{instructions: text}) when is_binary(text), do: text
  defp instructions(%{instructions: structured}), do: json(structured)

  defp serialize(state) when is_binary(state), do: state
  defp serialize(state), do: json(state)

  # A [MASK] in user text would become a marker.
  defp clean(text), do: String.replace(text, "[MASK]", " ")

  defp total(option_ids), do: option_ids |> Enum.map(&length/1) |> Enum.sum()
end
