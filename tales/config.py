import os

DEFAULT_TALES_CACHE_HOME = os.path.expanduser("~/.cache/tales")
TALES_CACHE_HOME = os.getenv("TALES_CACHE_HOME", DEFAULT_TALES_CACHE_HOME)
os.environ["TALES_CACHE_HOME"] = (
    TALES_CACHE_HOME  # Set the environment variable, in case it wasn't.
)
os.makedirs(TALES_CACHE_HOME, exist_ok=True)

# Check if cache is flag is set to force download
TALES_FORCE_DOWNLOAD = os.getenv("TALES_FORCE_DOWNLOAD", "false").lower() in (
    "yes",
    "true",
    "t",
    "1",
)
