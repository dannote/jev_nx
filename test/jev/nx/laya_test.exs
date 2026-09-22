defmodule Jev.Nx.LayaTest do
  # The reference implementation's sequences and answers for a handful of
  # cases, recorded by test/fixtures/golden.py. Sequences must match token for
  # token; probabilities agree with PyTorch to about 1.0e-6 on the CPU, so the
  # tolerance is 1.0e-4.
  use ExUnit.Case, async: false

  alias Jev.Nx.Laya

  @moduletag :model
  @moduletag timeout: 600_000

  @golden "test/fixtures/laya_golden.json" |> File.read!() |> JSON.decode!()

  setup_all do
    dir = System.get_env("JEV_NX_LAYA_DIR", Path.expand("~/.cache/jev_nx/laya"))
    {:ok, model} = Laya.load(repository: {:local, dir})

    onnx_dir = System.get_env("JEV_NX_LAYA_ONNX_DIR", Path.expand("~/.cache/jev_nx/laya-onnx"))

    onnx =
      if File.exists?(Path.join(onnx_dir, "laya.onnx")) do
        {:ok, model} = Laya.load(runtime: :onnx, repository: {:local, onnx_dir})
        Jev.Nx.Serving.build(model, sequence_length: [128, 512])
      end

    %{
      laya: model,
      serving: Jev.Nx.Serving.build(model, sequence_length: [128, 512]),
      onnx: onnx
    }
  end

  for %{"name" => name} = case <- @golden do
    @case case

    test "#{name}: sequences match the reference", %{laya: model} do
      state = @case["state"]

      for {qid, question} <- questions(@case["questions"]) do
        item = Laya.encode(model, state, question)
        expected = @case["sequences"][Atom.to_string(qid)]
        assert item.ids == expected["ids"], "ids differ for #{qid}"
        assert item.markers == expected["markers"], "markers differ for #{qid}"
      end
    end

    test "#{name}: answers match the reference", %{serving: serving} do
      assert_matches_reference(serving, @case)
    end

    test "#{name}: the ONNX export answers the same", %{onnx: onnx} do
      if onnx, do: assert_matches_reference(onnx, @case, 2.0e-3)
    end
  end

  defp assert_matches_reference(serving, %{} = case_, tolerance \\ 1.0e-4) do
    questions = questions(case_["questions"])
    wire = Nx.Serving.run(serving, {case_["state"], questions})

    assert wire.model == "laya-english"
    assert wire.usage.input_tokens == case_["input_tokens"]

    for {qid, expected} <- case_["answers"] do
      answer = wire.answers[qid]
      assert answer.type == String.to_existing_atom(expected["type"])

      case expected["type"] do
        "choice" ->
          assert answer.choice == expected["choice"]
          assert_close(answer.probabilities, expected["probabilities"], qid, tolerance)

        "score" ->
          assert_in_delta answer.score, expected["score"], tolerance
          assert_close(answer.probabilities, expected["probabilities"], qid, tolerance)

        "noul" ->
          assert_in_delta answer.noul, expected["noul"], tolerance
      end
    end

    reply = Jev.reply(wire, questions)
    assert is_map(reply.confidence)
  end

  test "rejects a question whose options overflow the head", %{laya: model} do
    huge =
      {"Which?",
       Map.new(1..200, fn i -> {:"option_#{i}", String.duplicate("long words ", 20)} end)}

    [which: question] = Jev.questions(which: huge) |> Map.to_list()

    assert_raise ArgumentError, ~r/do not fit/, fn -> Laya.encode(model, "state", question) end
  end

  defp questions(wire) do
    Map.new(wire, fn {name, q} -> {String.to_atom(name), question(q)} end)
  end

  defp question(%{"type" => "noul"} = q) do
    criteria =
      q["criteria"] && Map.new(q["criteria"], fn {k, v} -> {String.to_existing_atom(k), v} end)

    %Jev.Noul{instructions: instructions(q), criteria: criteria}
  end

  defp question(%{"type" => "choice"} = q) do
    %Jev.Choice{
      instructions: instructions(q),
      criteria: Map.new(q["criteria"], fn {k, v} -> {String.to_atom(k), v} end)
    }
  end

  defp question(%{"type" => "score"} = q),
    do: %Jev.Score{instructions: instructions(q), criteria: q["criteria"]}

  defp instructions(%{"instructions" => text}) when is_binary(text), do: text
  defp instructions(%{"instructions" => map}), do: map

  defp assert_close(actual, expected, qid, tolerance) do
    assert Map.keys(actual) |> Enum.sort() == Map.keys(expected) |> Enum.sort()

    for {label, p} <- expected do
      assert_in_delta actual[label], p, tolerance, "#{qid}/#{label}: #{actual[label]} vs #{p}"
    end
  end
end
