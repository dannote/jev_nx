# Jev.Nx.Model is the contract and knows nothing else. Jev.Nx.Serving batches
# any model through the contract, Jev.Nx is the Jev.Backend on top of a
# serving, and the models implement the contract without seeing either.
[
  layers: [
    contract: "Jev.Nx.Model",
    serving: "Jev.Nx.Serving",
    backend: "Jev.Nx",
    models: ["Jev.Nx.Laya", "Jev.Nx.Laya.Sequence", "Jev.Nx.Laya.Head"]
  ],
  deps: [
    forbidden: [
      {:contract, :serving},
      {:contract, :backend},
      {:contract, :models},
      {:serving, :backend},
      {:serving, :models},
      {:backend, :models},
      {:models, :serving},
      {:models, :backend}
    ]
  ]
]
