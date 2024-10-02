import os

DEFAULT_TWBENCH_CACHE_HOME = os.path.expanduser("~/.cache/tw-bench")
TWBENCH_CACHE_HOME = os.getenv("TWBENCH_CACHE_HOME", DEFAULT_TWBENCH_CACHE_HOME)
os.environ["TWBENCH_CACHE_HOME"] = TWBENCH_CACHE_HOME  # Set the environment variable, in case it wasn't.
os.makedirs(TWBENCH_CACHE_HOME, exist_ok=True)

# Check if cache is flag is set to force download
TWBENCH_FORCE_DOWNLOAD = os.getenv("TWBENCH_FORCE_DOWNLOAD", "false").lower() in ("yes", "true", "t", "1")
