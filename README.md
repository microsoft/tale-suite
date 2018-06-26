# TextWorld - Baselines
This repository contains the files needed to benchmark RL agents on text-based games.

## Getting Started
1.	Make sure TextWorld is installed (see [TextWorld's installation](https://github.com/Microsoft/TextWorld)).
2.	Download the curated list of games: `python download.py`.
3.	Run benchmark evaluation on all the games for the specified agent:
```
python benchmark --games games/* --agent agent_template.py:CustomAgent
```

## Requirements
- Python 3.5+
