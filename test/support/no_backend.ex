defmodule Jev.Nx.NoBackend do
  @moduledoc false
  # An Nx backend that can be selected but cannot hold a tensor. A test sets
  # it as the process default to prove code chose its own backend explicitly.
  def init(opts), do: opts
end
