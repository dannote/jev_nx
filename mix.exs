defmodule Jev.Nx.MixProject do
  use Mix.Project

  @version "0.1.1"
  @source_url "https://github.com/dannote/jev_nx"

  def project do
    [
      app: :jev_nx,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      name: "Jev.Nx",
      description: description(),
      source_url: @source_url,
      homepage_url: @source_url,
      package: package(),
      docs: docs(),
      deps: deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      dialyzer: [plt_add_apps: [:ex_unit]],
      aliases: aliases()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  def cli do
    [preferred_envs: [ci: :test]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp description do
    "Open decision models as a Jev backend: Laya and friends running in-process on Nx."
  end

  defp deps do
    [
      {:jev, "~> 0.2.0"},
      {:bumblebee, "~> 0.7.1"},
      {:nx, "~> 0.12 or ~> 0.13"},
      {:exla, ">= 0.0.0", only: [:dev, :test]},
      {:plug, "~> 1.14", only: :test},
      {:ex_doc, "~> 0.40", only: :dev, runtime: false},
      {:ex_slop, "~> 0.4", only: [:dev, :test], runtime: false},
      {:reach, "~> 2.0", only: [:dev, :test], runtime: false},
      {:ex_dna, "~> 1.0", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.0", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.0", only: [:dev, :test], runtime: false},
      {:vibe_kit, "~> 0.1", only: [:dev, :test], runtime: false},
      {:igniter, "~> 0.6", only: [:dev, :test]}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url, "Jev" => "https://hexdocs.pm/jev"},
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE NOTICE CHANGELOG.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      source_url: @source_url,
      source_ref: "v#{@version}",
      extras: ["README.md", "CHANGELOG.md", "LICENSE", "NOTICE"],
      groups_for_modules: [
        Backend: [Jev.Nx, Jev.Nx.Model, Jev.Nx.Serving],
        Laya: [Jev.Nx.Laya, Jev.Nx.Laya.Sequence, Jev.Nx.Laya.Head]
      ]
    ]
  end

  defp aliases do
    [
      ci: [
        "compile --warnings-as-errors",
        "format --check-formatted",
        "test",
        "credo --strict",
        "dialyzer",
        "ex_dna --max-clones 0",
        "reach.check --arch --smells"
      ]
    ]
  end
end
