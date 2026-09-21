defmodule Jev.Nx.ServingTest do
  use ExUnit.Case, async: true

  alias Jev.Nx.{Serving, Toy}
  alias Jev.Wire

  @questions Jev.questions(
               kind: {"Kind?", %{bug: nil, feature: nil, other: nil}},
               severity: {"Severity?", ["low", "mid", "high"]},
               urgent: "Urgent?"
             )

  test "answers every question with a wire response" do
    serving = Serving.new(Toy)

    assert %Wire.Response{model: "toy", answers: answers, usage: %Wire.Usage{input_tokens: 9}} =
             Nx.Serving.run(serving, {"one two three", @questions})

    # Logit = option index, so the last option wins everywhere.
    assert %Wire.Answer{type: :choice, choice: "other"} = answers["kind"]
    assert %Wire.Answer{type: :score, score: score} = answers["severity"]
    assert score > 1.5
    assert %Wire.Answer{type: :noul, noul: yes} = answers["urgent"]
    assert yes > 0.5
  end

  test "pads to the smallest bucket that fits and clips buckets to the model" do
    serving = Serving.new(Toy, [max_len: 20], sequence_length: [4, 8, 100], option_slots: [4, 8])

    long = Enum.map_join(1..30, " ", &"w#{&1}")

    assert %Wire.Response{usage: %{input_tokens: 60}} =
             Nx.Serving.run(serving, {long, @questions})

    many = {"Which?", Map.new(1..7, fn i -> {:"opt#{i}", nil} end)}

    assert %Wire.Response{answers: %{"which" => %{choice: "opt7"}}} =
             Nx.Serving.run(serving, {"winner=opt7", Jev.questions(which: many)})
  end

  test "batches concurrent callers and keeps their answers apart" do
    serving = Serving.new(Toy, [], sequence_length: [8])

    start_supervised!(
      {Nx.Serving, serving: serving, name: __MODULE__, batch_size: 8, batch_timeout: 50}
    )

    tasks =
      for label <- ~w(bug feature other bug)a do
        Task.async(fn ->
          questions = Jev.questions(kind: {"Kind?", %{bug: nil, feature: nil, other: nil}})
          {label, Nx.Serving.batched_run(__MODULE__, {"winner=#{label}", questions})}
        end)
      end

    for {label, %Wire.Response{answers: %{"kind" => answer}}} <- Task.await_many(tasks) do
      assert answer.choice == Atom.to_string(label)
    end
  end

  test "compile: true needs a batch size" do
    assert_raise ArgumentError, ~r/batch_size/, fn -> Serving.new(Toy, [], compile: true) end
    assert %Nx.Serving{} = Serving.new(Toy, [], compile: true, batch_size: 4)
  end
end
