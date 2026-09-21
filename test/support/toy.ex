defmodule Jev.Nx.Toy do
  @moduledoc false
  # A Jev.Nx.Model with no network: each option's logit is the option's index,
  # unless the state names a winner. Exercises the serving and the backend.

  @behaviour Jev.Nx.Model

  defstruct max_len: 64

  @impl true
  def load(opts), do: {:ok, struct!(__MODULE__, opts)}

  @impl true
  def name(_), do: "toy"

  @impl true
  def max_sequence_length(%{max_len: max}), do: max

  @impl true
  def encode(%{max_len: max}, state, question) do
    labels = Jev.Nx.Laya.Sequence.labels(question)
    text = if is_binary(state), do: state, else: JSON.encode!(state)
    tokens = min(max, text |> String.split() |> length())
    winner = Enum.find_index(labels, &String.contains?(text, "winner=#{&1}"))
    %{tokens: tokens, options: length(labels), labels: labels, question: question, winner: winner}
  end

  @impl true
  def batch(_model, items, length, slots) do
    %{
      "logits" => Nx.tensor(Enum.map(items, &logits(&1, slots)), type: :f32),
      "length" => Nx.tensor(List.duplicate(length, length(items)))
    }
  end

  defp logits(item, slots) do
    base = Enum.map(0..(slots - 1), fn i -> if i < item.options, do: i / 1, else: -100.0 end)
    if item.winner, do: List.replace_at(base, item.winner, 50.0), else: base
  end

  @impl true
  def template(_model, batch_size, _length, slots) do
    %{
      "logits" => Nx.template({batch_size, slots}, :f32),
      "length" => Nx.template({batch_size}, :s32)
    }
  end

  @impl true
  def params(_model), do: %{scale: Nx.tensor(1.0)}

  @impl true
  def forward(_model),
    do: fn params, inputs ->
      %{logits: Nx.multiply(inputs["logits"], params.scale), length: inputs["length"]}
    end

  @impl true
  def decode(_model, items, outputs) do
    Enum.zip_with(Nx.to_list(outputs.logits), items, fn row, item ->
      probabilities = row |> Enum.take(item.options) |> softmax()
      answer(item.question, item.labels, probabilities)
    end)
  end

  defp answer(%Jev.Choice{}, labels, ps) do
    {label, _} = labels |> Enum.zip(ps) |> Enum.max_by(&elem(&1, 1))
    %Jev.Wire.Answer{type: :choice, choice: label, probabilities: Map.new(Enum.zip(labels, ps))}
  end

  defp answer(%Jev.Score{}, labels, ps) do
    score = ps |> Enum.with_index() |> Enum.map(fn {p, i} -> p * i end) |> Enum.sum()
    %Jev.Wire.Answer{type: :score, score: score, probabilities: Map.new(Enum.zip(labels, ps))}
  end

  defp answer(%Jev.Noul{}, _labels, [_no, yes]), do: %Jev.Wire.Answer{type: :noul, noul: yes}

  defp softmax(xs) do
    top = Enum.max(xs)
    es = Enum.map(xs, &:math.exp(&1 - top))
    Enum.map(es, &(&1 / Enum.sum(es)))
  end
end
