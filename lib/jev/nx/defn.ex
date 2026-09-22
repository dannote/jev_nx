defmodule Jev.Nx.Defn do
  @moduledoc """
  The `c:Jev.Nx.Model.init/4` of a model that runs on `Nx.Defn`.

  A model built from Axon or `defn` implements the callback with `runner/3`,
  which compiles or jits the forward pass, moves the parameters to the
  compiler's backend once per serving process when asked, and hands back the
  function the serving calls:

      @impl Jev.Nx.Model
      def init(%__MODULE__{} = model, shape, batch_size, defn_options) do
        Jev.Nx.Defn.runner(
          fn -> params(model) end,
          forward(model),
          defn_options ++ [template: template(model, batch_size, shape)]
        )
      end

  Without a `:template` the forward pass is jitted and compiles on the shapes
  it is given. With one it is compiled ahead of time, which is what a serving
  built with `compile: true` passes.
  """

  @doc """
  Builds the run function for one batch shape.

  `params` is a function so nothing is loaded or copied when the parameters
  are already prepared. `forward` takes the parameters and the input tensors.

  ## Options

    * `:template` - input templates to compile against, otherwise jit
    * `:preallocate_params` - copy the parameters to the compiler's backend
      once per process, default `false`
    * every other option goes to `Nx.Defn.compile/3` or `Nx.Defn.jit/2`
  """
  @spec runner((-> Nx.Container.t()), (Nx.Container.t(), map() -> Nx.Container.t()), keyword()) ::
          (map() -> Nx.Container.t())
  def runner(params, forward, opts) do
    {template, opts} = Keyword.pop(opts, :template)
    {preallocate?, defn_options} = Keyword.pop(opts, :preallocate_params, false)

    params = params(params, preallocate?, defn_options)
    forward = compile(forward, template, params, defn_options)

    fn inputs -> forward.(params, inputs) end
  end

  defp params(params, false, _defn_options), do: params.()

  # The serving calls init once per batch shape in one process, so the copy is
  # cached there and every shape's program shares it.
  defp params(params, true, defn_options) do
    key = {__MODULE__, defn_options}

    with nil <- Process.get(key) do
      copied = Nx.backend_copy(params.(), Nx.Defn.to_backend(defn_options))
      Process.put(key, copied)
      copied
    end
  end

  defp compile(forward, nil, _params, defn_options), do: Nx.Defn.jit(forward, defn_options)

  defp compile(forward, template, params, defn_options) do
    params = Nx.Defn.Composite.traverse(params, &Nx.to_template/1)
    Nx.Defn.compile(forward, [params, template], defn_options)
  end
end
