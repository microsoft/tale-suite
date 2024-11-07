# Text-games Benchmark
This repository contains the files needed to benchmark language agents on text-based games.

## Getting Started
1.	Make a dedicted conda virtual environment.

        conda create -n tw-bench python=3.10
        conda activate tw-bench

2.	Install the required packages (Jericho, TextWorld, TextWorld-Express, ScienceWorld, DiscoveryWorld, ALFWorld).

        pip install -r requirements.txt

3.	Run benchmark evaluation on all the games for the specified random agent:

        python benchmark.py --agent agents/random.py random

4.	Run benchmark evaluation on a subset of the games:

        python benchmark.py --env tw-cooking --agent agents/random.py:RandomAgent

5.	Run benchmark evaluation on specific games:

        python benchmark.py --agent agents/random.py random --envs JerichoEnvZork1 JerichoEnvDetective

6.	Run benchmark evaluation using as a HumanAgent (Not working atm):

        python benchmark.py --agent agents/human.py human --envs TWCookingLevel1

7.	Run benchmark evaluation where the LLM is provided the ground-truth walkthrough:

        python benchmark.py --agent agents/llm_walkthrough.py walkthrough --envs JerichoEnvZork1


You will need java 1.8+ installed to run the environments TextWorld-Express and ScienceWorld.

## Benchmarking LLMs

In order to benchmark a given LLM acting as language agent playing text-based games, you will need to first configure it. `tw-bench` is leveraging the [`llm`](https://llm.datasette.io/en/stable/) library to handle communication with different LLMs.

    python benchmark.py --envs TWCookingLevel1 --agent agents/llm.py:LLMAgent

### API-based LLMS

`llm` natively supports OpenAI models.

### Adding support to other LLMS

    llm install llm-azure


## Custom Agents

To build a custom agent, you need to create a new file (e.g., `custom.py`) in the agents folder and implement the `Agent` class.

```python
from typing import Dict, Any
import twbench

class CustomAgent(twbench.Agent):

    def act(self, obs: str, reward: float, done: bool, infos: Dict[str, Any]) -> str:
        # ...
        return "help"
```

You can then use this agent by specifying the path to the file and the class name in the `--agent` argument.

        python benchmark.py --agent agents/custom.py:CustomAgent


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Privacy

[Microsoft Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement)
