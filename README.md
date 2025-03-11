# Text-games Benchmark
This repository contains the files needed to benchmark language agents on text-based games.

## Getting Started
1.	Make a dedicated conda virtual environment.

        conda create -n tw-bench python=3.12
        conda activate tw-bench

2.	Install the required packages (Jericho, TextWorld, TextWorld-Express, ScienceWorld, ALFWorld).

        pip install -r requirements.txt

> [!WARNING]
> You will need Java 1.8+ installed to run the environments TextWorld-Express and ScienceWorld.
>
>     sudo apt update && apt install openjdk-8-jre-headless -y

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

*Note: The walkthrough agent does not add ">" to its action history as this causes the llm to then generate ">action" which is not accepted by the game engine. For example, the llm will first generate "action" but this will be appended to the context as ">action". Thus, the llm will then generate ">action1" which is not accepted by the game engine.

## Examples
We provide a pre-built docker image at

        docker pull czcui/twb:prebuilt

[Please see the following this docs page for more details on how to set up a local vllm for use with the text world benchmark.](https://docs.google.com/document/d/1Q5FtcNpYDpMLbyraJ1dSKxJLwOgLvWCECiPsnDkEq2Y/edit?usp=sharing)

An example script can be found in the scripts folder.

## Benchmarking LLMs

In order to benchmark a given LLM acting as language agent playing text-based games, you will need to first configure it. `tw-bench` is leveraging the [`llm`](https://llm.datasette.io/en/stable/) library to handle communication with different LLMs.

    python benchmark.py --envs TWCookingLevel1 --agent agents/llm.py zero-shot

### API-based LLMS

`llm` natively supports OpenAI models and self-hosted models that offer an OpenAI-compatible API (e.g. like vLLM does - more on this below).

### Adding support to other LLMS

`llm` offers different plugins to include other LLMs. E.g.

    llm install llm-claude

See the `llm`plugins [page](https://llm.datasette.io/en/stable/plugins/directory.html) for more information.

### Deploying a model locally using vLLM

To serve a custom HugginFace model with vLLM, one can use the vllm docker image like this:

    docker run --runtime nvidia --gpus all --restart unless-stopped --name vllm-Llama-3.1-8B-Instruct --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 --host 0.0.0.0

Then, add the following entrypoint in `~/.config/io.datasette.llm/extra-openai-models.yaml`

```
- model_id: meta-llama/Llama-3.1-8B-Instruct
  model_name: meta-llama/Llama-3.1-8B-Instruct
  api_base: "http://0.0.0.0:8000/v1"
```

You can check that everything is working properly with this simple command:

    llm -m meta-llama/Llama-3.1-8B-Instruct "Hi. What's your name?"

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
