# Text-games - Benchmark
This repository contains the files needed to benchmark language agents on text-based games.

## Getting Started
1.	Make sure dependencies are installed.
    pip install -r requirements.txt

2.	Download the curated list of games.

    python download.py

3.	Run benchmark evaluation on all the games for the specified agent:

    python benchmark.py --games games/* --agent agent_template.py:CustomAgent
