# Largest probability difference from the Python reference, per runtime, over
# the golden cases. Run with the checkpoints in ~/.cache/jev_nx.
laya = System.get_env("JEV_NX_LAYA_DIR", Path.expand("~/.cache/jev_nx/laya"))
onnx = System.get_env("JEV_NX_LAYA_ONNX_DIR", Path.expand("~/.cache/jev_nx/laya-onnx"))
golden = "test/fixtures/laya_golden.json" |> File.read!() |> JSON.decode!()

question = fn
  %{"type" => "noul"} = q ->
    criteria = q["criteria"] && Map.new(q["criteria"], fn {k, v} -> {String.to_existing_atom(k), v} end)
    %Jev.Noul{instructions: q["instructions"], criteria: criteria}

  %{"type" => "choice"} = q ->
    %Jev.Choice{instructions: q["instructions"], criteria: Map.new(q["criteria"], fn {k, v} -> {String.to_atom(k), v} end)}

  %{"type" => "score"} = q ->
    %Jev.Score{instructions: q["instructions"], criteria: q["criteria"]}
end

runtimes =
  [
    {"bumblebee/exla-cpu", [repository: {:local, laya}], [defn_options: [compiler: EXLA]]},
    {"onnx/cpu", [runtime: :onnx, repository: {:local, onnx}], []}
  ] ++
    if Code.ensure_loaded?(EMLX) do
      [{"bumblebee/emlx-metal", [repository: {:local, laya}], [defn_options: [compiler: EMLX], preallocate_params: true]}]
    else
      []
    end

probability_deltas = fn answer, expected ->
  Enum.map(expected["probabilities"], fn {label, p} -> abs(answer.probabilities[label] - p) end)
end

IO.puts("\n#{String.pad_trailing("runtime", 22)} max |Δp|   labels")

for {name, load_opts, serving_opts} <- runtimes do
  {:ok, model} = Jev.Nx.Laya.load(load_opts)
  serving = Jev.Nx.Serving.build(model, [sequence_length: [128, 512]] ++ serving_opts)

  {deltas, agree} =
    Enum.reduce(golden, {[], true}, fn case_, {deltas, agree} ->
      questions = Map.new(case_["questions"], fn {n, q} -> {String.to_atom(n), question.(q)} end)
      wire = Nx.Serving.run(serving, {case_["state"], questions})

      Enum.reduce(case_["answers"], {deltas, agree}, fn {qid, expected}, {deltas, agree} ->
        answer = wire.answers[qid]

        case expected["type"] do
          "noul" -> {[abs(answer.noul - expected["noul"]) | deltas], agree}
          "choice" -> {probability_deltas.(answer, expected) ++ deltas, agree and answer.choice == expected["choice"]}
          "score" -> {probability_deltas.(answer, expected) ++ deltas, agree}
        end
      end)
    end)

  IO.puts(
    String.pad_trailing(name, 22) <>
      String.pad_trailing(Float.to_string(Float.round(Enum.max(deltas), 6)), 11) <>
      if(agree, do: "all match", else: "DIFFER")
  )
end

IO.puts("")
