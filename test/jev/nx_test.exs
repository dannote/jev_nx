defmodule Jev.NxTest do
  use ExUnit.Case, async: true

  alias Jev.Nx.{Serving, Toy}

  setup do
    ref = make_ref()
    events = [[:jev, :request, :start], [:jev, :request, :stop], [:jev, :answer]]
    :telemetry.attach_many({__MODULE__, ref}, events, &__MODULE__.forward/4, {self(), ref})
    on_exit(fn -> :telemetry.detach({__MODULE__, ref}) end)
    %{ref: ref, serving: Serving.new(Toy)}
  end

  def forward(event, measurements, metadata, {parent, ref}) do
    send(parent, {ref, event, measurements, metadata})
  end

  test "is a Jev.Backend that returns the reply map", %{serving: serving} do
    assert {:ok, reply} =
             Jev.Nx.post(
               "winner=feature",
               [kind: {"Kind?", %{bug: nil, feature: nil}}, urgent: "Urgent?"],
               serving: serving
             )

    assert %{kind: :feature, urgent: urgent, model: "toy", usage: %{cost: cost}} = reply
    assert cost == 0
    assert is_float(urgent)
    assert reply.confidence.kind > 0.99
  end

  test "emits the shared telemetry", %{serving: serving, ref: ref} do
    {:ok, _} = Jev.Nx.post("s", [kind: {"Kind?", %{a: nil, b: nil}}], serving: serving, tag: ref)

    assert_receive {^ref, [:jev, :request, :start], _, %{backend: Jev.Nx, tag: ^ref}}
    assert_receive {^ref, [:jev, :request, :stop], %{input_tokens: 1}, %{model: "toy"}}
    assert_receive {^ref, [:jev, :answer], _, %{name: :kind, tag: ^ref}}
  end

  test "a Jev.Server can pick it per request", %{serving: serving} do
    defmodule Picky do
      use Jev.Server
      def init(_), do: {:ok, %{}}

      def handle_call({:ask, opts}, from, s),
        do: {:reply, {from, "winner=b", [kind: {"Kind?", %{a: nil, b: nil}}], opts}, s}

      def handle_answer(reply, from, s) do
        GenServer.reply(from, reply)
        {:noreply, s}
      end
    end

    pid = start_supervised!({Picky, []})

    assert %{kind: :b, model: "toy"} =
             GenServer.call(pid, {:ask, [backend: Jev.Nx, serving: serving]})
  end

  test "needs a serving" do
    assert_raise ArgumentError, ~r/serving/, fn ->
      Jev.Nx.post("s", kind: {"Kind?", %{a: nil, b: nil}})
    end
  end
end
