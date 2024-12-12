import textworld_express as twx

# TASK_NAMES = list(twx.GAME_NAMES)

TASKS = [
    (
        "CookingWorld",
        "cookingworld",
        "numLocations=1, numIngredients=2, numDistractorItems=5, includeDoors=0, limitInventorySize=0",
    ),
    (
        "TextWorldCommonsense",
        "twc",
        "numLocations=1,numItemsToPutAway=1,includeDoors=0,limitInventorySize=0",
    ),
    (
        "CoinCollector",
        "coin",
        "numLocations=1, numDistractorItems=5, limitInventorySize=0",
    ),
    ("Arithmetic", "arithmetic", ""),
    (
        "MapReader",
        "mapreader",
        "numLocations=2, maxDistanceApart=1, maxDistractorItemsPerLocation=2, includeDoors=0, limitInventorySize=0",
    ),
    ("Sorting", "sorting", ""),
    ("SimonSays10", "simonsays", "gameLength=10, numDistractors=4, memorization=0"),
    ("SimonSays20", "simonsays", "gameLength=20, numDistractors=4, memorization=0"),
    ("SimonSays50", "simonsays", "gameLength=50, numDistractors=4, memorization=0"),
    ("SimonSays100", "simonsays", "gameLength=100, numDistractors=4, memorization=0"),
    (
        "SimonSaysWithMemory10",
        "simonsays",
        "gameLength=10, numDistractors=4, memorization=1",
    ),
    (
        "SimonSaysWithMemory20",
        "simonsays",
        "gameLength=20, numDistractors=4, memorization=1",
    ),
    (
        "SimonSaysWithMemory50",
        "simonsays",
        "gameLength=50, numDistractors=4, memorization=1",
    ),
    (
        "SimonSaysWithMemory100",
        "simonsays",
        "gameLength=100, numDistractors=4, memorization=1",
    ),
    ("PeckingOrder", "peckingorder", ""),
]


def get_seeds(split, env=None):
    env = env or twx.TextWorldExpressEnv()
    if split == "train":
        return env.getValidSeedsTrain()
    elif split == "valid":
        return env.getValidSeedsDev()
    elif split == "test":
        return env.getValidSeedsTest()
    else:
        raise NotImplementedError("Only plan to support train, dev, and test splits.")
