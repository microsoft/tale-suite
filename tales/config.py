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

# The Jericho walkthrough catalog (tales/jericho/walkthroughs.jsonl) is kept private
# and is not part of the repository. Without it, loading Jericho fails unless this
# flag is set, in which case Jericho loads with no walkthroughs.
TALES_NO_WALKTHROUGHS = os.getenv("TALES_NO_WALKTHROUGHS", "false").lower() in (
    "yes",
    "true",
    "t",
    "1",
)
