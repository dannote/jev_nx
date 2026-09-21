defmodule Jev.Nx.Laya.Head do
  @moduledoc """
  Laya's decision head, from the published weights.

  On top of the encoder's hidden states: a learned embedding of the question
  type is added, two pre-norm transformer layers run over the sequence with
  the padding mask, the hidden state at each option's `[MASK]` marker is
  gathered, and a scorer turns each into one logit. Softmax over a question's
  markers is its distribution; calibration by temperature happens in
  `Jev.Nx.Laya` after the batch, where the option count is known.

  The layers follow PyTorch's `TransformerEncoderLayer` with `norm_first`,
  which means ReLU in the feed-forward and 64-wide attention heads. The
  scorer is `LayerNorm`, `Linear`, GELU, `Linear`.
  """

  import Nx.Defn

  @epsilon 1.0e-5
  @head_size 64
  @masked -1.0e4

  @doc """
  Logits at the marker positions, `{batch, option_slots}`, with unused slots
  at a large negative value.
  """
  defn forward(hidden_state, attention_mask, marker_pos, marker_mask, qtype, params) do
    hidden_state = Nx.as_type(hidden_state, :f32)
    hidden_state = hidden_state + Nx.new_axis(Nx.take(params.type_embedding, qtype), 1)

    hidden_state = layer(hidden_state, attention_mask, params.layer_0)
    hidden_state = layer(hidden_state, attention_mask, params.layer_1)

    {batch_size, sequence_length, hidden_size} = Nx.shape(hidden_state)
    option_slots = Nx.axis_size(marker_pos, 1)

    indices =
      marker_pos
      |> Nx.clip(0, sequence_length - 1)
      |> Nx.new_axis(-1)
      |> Nx.broadcast({batch_size, option_slots, hidden_size})

    logits =
      hidden_state
      |> Nx.take_along_axis(indices, axis: 1)
      |> layer_norm(params.scorer.norm)
      |> dense(params.scorer.dense)
      |> Axon.Activations.gelu()
      |> dense(params.scorer.output)
      |> Nx.squeeze(axes: [-1])

    Nx.select(marker_mask == 1, logits, @masked)
  end

  defnp layer(hidden_state, attention_mask, params) do
    hidden_state =
      hidden_state + attention(layer_norm(hidden_state, params.norm_1), attention_mask, params)

    feed_forward =
      hidden_state
      |> layer_norm(params.norm_2)
      |> dense(params.linear_1)
      |> Axon.Activations.relu()
      |> dense(params.linear_2)

    hidden_state + feed_forward
  end

  defnp attention(hidden_state, attention_mask, params) do
    {batch_size, sequence_length, hidden_size} = Nx.shape(hidden_state)
    heads = div(hidden_size, @head_size)

    qkv =
      hidden_state
      |> dense(params.in_proj)
      |> Nx.reshape({batch_size, sequence_length, 3, heads, @head_size})

    query =
      qkv
      |> Nx.slice_along_axis(0, 1, axis: 2)
      |> Nx.squeeze(axes: [2])
      |> Nx.transpose(axes: [0, 2, 1, 3])

    key =
      qkv
      |> Nx.slice_along_axis(1, 1, axis: 2)
      |> Nx.squeeze(axes: [2])
      |> Nx.transpose(axes: [0, 2, 1, 3])

    value =
      qkv
      |> Nx.slice_along_axis(2, 1, axis: 2)
      |> Nx.squeeze(axes: [2])
      |> Nx.transpose(axes: [0, 2, 1, 3])

    scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1]) / Nx.sqrt(@head_size)

    padding =
      attention_mask
      |> Nx.equal(1)
      |> Nx.reshape({batch_size, 1, 1, sequence_length})
      |> Nx.broadcast({batch_size, heads, sequence_length, sequence_length})

    weights = padding |> Nx.select(scores, -1.0e9) |> Axon.Activations.softmax()

    weights
    |> Nx.dot([3], [0, 1], value, [2], [0, 1])
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, sequence_length, hidden_size})
    |> dense(params.out_proj)
  end

  # PyTorch layouts: weight is {out, in}.
  defnp(dense(input, %{weight: weight, bias: bias}), do: Nx.dot(input, [-1], weight, [1]) + bias)

  defnp layer_norm(input, %{weight: weight, bias: bias}) do
    mean = Nx.mean(input, axes: [-1], keep_axes: true)
    variance = Nx.mean((input - mean) ** 2, axes: [-1], keep_axes: true)
    (input - mean) / Nx.sqrt(variance + @epsilon) * weight + bias
  end
end
