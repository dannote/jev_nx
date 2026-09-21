defmodule Jev.Nx do
  @moduledoc """
  Open decision models as a `Jev.Backend`, running in-process on Nx.

  A `Jev.Server` that asks a local model instead of TypeSafe:

      children = [
        {Nx.Serving, serving: Jev.Nx.serving(Jev.Nx.Laya), name: MyApp.Laya, batch_size: 16}
      ]

      config :jev, backend: Jev.Nx
      config :jev_nx, serving: MyApp.Laya

      # or per request
      {:reply, {tag, state, [kind: @kinds], [backend: Jev.Nx, serving: MyApp.Laya]}, s}

  Replies have exactly the shape `Jev.HTTP` produces, built through
  `Jev.Wire` and `Jev.reply/3`, so `handle_answer/3` clauses, confidence
  thresholds, and telemetry dashboards do not know which answered. That makes
  the cascade in the Jev guides one line: ask `Jev.Nx` first, escalate to
  `Jev.HTTP` when confidence is low.

  `Jev.Nx.Laya` is the first model. `Jev.Nx.Model` is how to add another.

  ## Backends

  This package does not pick an Nx backend or compiler. Configure one in the
  application: `EXLA` on CPU or CUDA, `Emily` or `EMLX` for Metal on Apple
  Silicon. Parameters load onto `Nx.default_backend/0`.
  """

  @behaviour Jev.Backend

  @typedoc "A running `Nx.Serving` process, by name or pid, or a bare `Nx.Serving` for scripts."
  @type serving :: GenServer.server() | Nx.Serving.t()

  @doc "Builds the serving for `model`; see `Jev.Nx.Serving.new/3`."
  @spec serving(module(), keyword(), keyword()) :: Nx.Serving.t()
  defdelegate serving(model, model_opts \\ [], opts \\ []), to: Jev.Nx.Serving, as: :new

  @doc """
  Answers `questions` about `state` with the model behind `opts[:serving]`,
  or `config :jev_nx, serving:`.

  Cost is zero. Usage counts the tokens the model read.
  """
  @impl Jev.Backend
  @spec post(Jev.entry(), Jev.questions() | keyword() | map(), keyword()) ::
          {:ok, Jev.reply()} | {:error, term()}
  def post(state, questions, opts \\ []) do
    questions = Jev.questions(questions)
    serving = serving_from(opts)
    metadata = %{backend: __MODULE__, serving: serving, tag: opts[:tag]}

    Jev.Telemetry.span(state, questions, metadata, fn ->
      wire = run(serving, {state, questions})
      {:ok, Jev.reply(wire, questions, usd_per_million_input: 0), %{model: wire.model}}
    end)
  end

  defp serving_from(opts) do
    opts[:serving] || Application.get_env(:jev_nx, :serving) ||
      raise ArgumentError, "pass serving: MyApp.Laya or set config :jev_nx, serving: MyApp.Laya"
  end

  defp run(%Nx.Serving{} = serving, input), do: Nx.Serving.run(serving, input)
  defp run(serving, input), do: Nx.Serving.batched_run(serving, input)
end
