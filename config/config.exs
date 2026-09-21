import Config

if config_env() == :test do
  config :nx, default_backend: EXLA.Backend
  config :nx, :default_defn_options, compiler: EXLA
end
