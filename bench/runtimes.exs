# Laya's runtimes on this machine, with Benchee. Not a JevBench score: one
# process, one short state, servings warm before the first measurement.
#
#     mix run bench/runtimes.exs
laya = System.get_env("JEV_NX_LAYA_DIR", Path.expand("~/.cache/jev_nx/laya"))
onnx = System.get_env("JEV_NX_LAYA_ONNX_DIR", Path.expand("~/.cache/jev_nx/laya-onnx"))

questions =
  Jev.questions(
    kind: {"What kind of issue?", %{bug: "Broken", feature: "New behavior", other: nil}},
    severity: {"How severe?", ["Cosmetic", "Workaround", "Blocks", "Data loss"]},
    security: "Is this a vulnerability?"
  )

state = %{title: "App crashes on launch", body: "Since 2.3.1 the app closes immediately on iOS 17."}

runtimes =
  [
    {"bumblebee/exla-cpu", [repository: {:local, laya}], [defn_options: [compiler: EXLA]]},
    {"onnx/cpu", [runtime: :onnx, repository: {:local, onnx}], []}
  ] ++
    if Code.ensure_loaded?(EMLX) do
      [
        {"bumblebee/emlx-metal", [repository: {:local, laya}],
         [defn_options: [compiler: EMLX], preallocate_params: true]}
      ]
    else
      []
    end

servings =
  Map.new(runtimes, fn {name, load_opts, serving_opts} ->
    {:ok, model} = Jev.Nx.Laya.load(load_opts)
    serving = Jev.Nx.Serving.build(model, [sequence_length: [128, 512]] ++ serving_opts)
    # Compile every shape the benchmark uses before anything is measured.
    Nx.Serving.run(serving, {state, Map.take(questions, [:kind])})
    Nx.Serving.run(serving, {state, questions})
    {name, serving}
  end)

for {label, asked} <- [{"1 question", Map.take(questions, [:kind])}, {"3 questions", questions}] do
  IO.puts("\n== #{label}")

  servings
  |> Map.new(fn {name, serving} -> {name, fn -> Nx.Serving.run(serving, {state, asked}) end} end)
  |> Benchee.run(warmup: 2, time: 10, print: [configuration: false])
end
