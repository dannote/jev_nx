# Tests tagged :model load the Laya checkpoint from JEV_NX_LAYA_DIR, or
# ~/.cache/jev_nx/laya, and are skipped when it is not there.
laya = System.get_env("JEV_NX_LAYA_DIR", Path.expand("~/.cache/jev_nx/laya"))
exclude = if File.exists?(Path.join(laya, "model.safetensors")), do: [], else: [model: true]
ExUnit.start(exclude: exclude)
