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

  test "builds client tensors on the binary backend whatever the default is" do
    serving = Serving.new(Toy)

    # A backend that does not exist: any tensor built on the process default
    # raises, so preprocessing passes only by choosing the binary backend itself.
    previous = Nx.default_backend()
    Nx.default_backend(Jev.Nx.NoBackend)
    on_exit(fn -> Nx.default_backend(previous) end)

    assert_raise UndefinedFunctionError, fn -> Nx.tensor([1]) end
    assert {%Nx.Batch{size: 3}, _info} = serving.client_preprocessing.({"state", @questions})
  end

  test "preallocates the parameters once for every shape" do
    serving =
      Serving.new(Toy, [],
        sequence_length: [8, 16],
        option_slots: [4, 8],
        preallocate_params: true
      )

    start_supervised!({Nx.Serving, serving: serving, name: __MODULE__.Prealloc, batch_size: 4})

    # Two shapes, one copy: the second key finds the first key's parameters.
    for state <- ["one", Enum.map_join(1..12, " ", &"w#{&1}")] do
      assert %Wire.Response{} = Nx.Serving.batched_run(__MODULE__.Prealloc, {state, @questions})
    end

    pid = Process.whereis(__MODULE__.Prealloc)
    {:dictionary, dictionary} = Process.info(pid, :dictionary)
    assert [_one] = for({{Jev.Nx.Defn, _}, _} <- dictionary, do: :copy)
  end

  test "compile: true pads every batch to the batch size" do
    serving =
      Serving.new(Toy, [], sequence_length: [8], option_slots: [4], compile: true, batch_size: 4)

    assert %Wire.Response{answers: %{"kind" => %{choice: "other"}}} =
             Nx.Serving.run(
               serving,
               {"one", Jev.questions(kind: {"Kind?", %{bug: nil, feature: nil, other: nil}})}
             )
  end

  test "compile: true needs a batch size" do
    assert_raise ArgumentError, ~r/batch_size/, fn -> Serving.new(Toy, [], compile: true) end
    assert %Nx.Serving{} = Serving.new(Toy, [], compile: true, batch_size: 4)
  end
end
