# TALES: Text-Adventure Learning Environment Suite
This repository contains the files needed to benchmark language agents on a curated list of text-based games from the following frameworks: [Jericho](https://github.com/microsoft/jericho), [TextWorld](https://github.com/microsoft/textworld), [TextWorld-Express](https://github.com/cognitiveailab/TextWorldExpress), [ScienceWorld](https://github.com/allenai/ScienceWorld), [ALFWorld](https://github.com/alfworld/alfworld)).

[[Technical Report](https://arxiv.org/abs/2504.14128)] [[Project Page](https://t.co/rFPMRoqO9y)]

## 1. Installation

It is recommended to create and activate a conda or virtual environment. `tales` requires `Python>=3.12`:

    conda create -n tales python=3.12
    conda activate tales

Then, install `tales` directly from PyPI:

    pip install tale-suite

> [!WARNING]
> The name of the Python package on PyPI is `tale-suite` and not `tales`.

Alternatively, clone the repository and install locally:

    git clone https://github.com/microsoft/tale-suite
    cd tale-suite
    pip install -e .

> [!WARNING]
> You will need Java 1.8+ installed to run the environments TextWorld-Express and ScienceWorld.
>
>     sudo apt update && apt install openjdk-8-jre-headless -y

Alternatively, if the above isn't working:

>      sudo apt-get update && apt-get install default-jre default-jdk

### Using Docker
We provide a pre-built docker image at

    docker pull czcui/twb:prebuilt

[Please see the following docs page for more details on how to set up a local vllm for use with the text world benchmark.](https://docs.google.com/document/d/1Q5FtcNpYDpMLbyraJ1dSKxJLwOgLvWCECiPsnDkEq2Y/edit?usp=sharing)

An example script can be found in the scripts folder.

## 2. Getting Started

1.	Run benchmark evaluation on all the games for the specified random agent:

    ```python
    python benchmark.py --agent agents/random.py random

2.	Run benchmark evaluation on a subset of the games:

    ```python
    python benchmark.py --agent agents/random.py random --env textworld

3.	Run benchmark evaluation on specific games:

    ```python
    python benchmark.py --agent agents/random.py random --envs JerichoEnvZork1 JerichoEnvDetective

4.	Run benchmark evaluation using as a HumanAgent:

    ```python
    python benchmark.py --agent agents/human.py human --envs TWCookingLevel1

5.	Run benchmark evaluation where the ground-truth walkthrough is being followed:

    ```python
    python benchmark.py --agent agents/walkthrough.py walkthrough --envs JerichoEnvZork1


## 3. Benchmarking LLMs

In order to benchmark a given LLM acting as language agent playing text-based games, you will need to first configure it. `tales` is leveraging the [`llm`](https://llm.datasette.io/en/stable/) library to handle communication with different LLMs.

    python benchmark.py --agent agents/llm.py zero-shot --envs TWCookingLevel1

### API-based LLMs

`llm` natively supports OpenAI models and self-hosted models that offer an OpenAI-compatible API (e.g. like vLLM does - more on this below).

### Adding support to other LLMs

`llm` offers different plugins to include other LLMs. E.g.

    llm install llm-anthropic

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

## 4. Building Custom Agents

To build a custom agent, you need to create a new file (e.g., `custom.py`) in the agents folder and implement the `Agent` class and implement the proper arguments parser.

```python
from typing import Dict, Any
import tales

class CustomAgent(tales.Agent):

    def act(self, obs: str, reward: float, done: bool, infos: Dict[str, Any]) -> str:
        # ...
        return "help"


def build_argparser(parser=None):
    return parser or argparse.ArgumentParser()


register(
    name="my-agent",
    desc=(
        "This is a custom agent that always output 'help' as a text action."
    ),
    klass=CustomAgent,
    add_arguments=build_argparser,
)
```

You can then use this agent by specifying the path to the file and the class name in the `--agent` argument.

        python benchmark.py --agent agents/custom.py my-agent

> [!NOTE]
> See the [agents folder](https://github.com/microsoft/tale-suite/tree/main/agents) for more concrete examples.

## Citation
```
@article{cui2025tales,
  title={TALES: Text-Adventure Learning Environment Suite},
  author={Christopher Cui, Xingdi Yuan, Ziang Xiao, Prithviraj Ammanabrolu, Marc-Alexandre C\^ot\'e},
  journal={arXiv preprint arXiv:2504.14128},
  year={2025},
  url={https://arxiv.org/abs/2504.14128}
}
```

If you use this benchmark, please consider citing the original frameworks as well.
```
@article{cote18textworld,
  author = {Marc-Alexandre C\^ot\'e and \'Akos K\'ad\'ar and Xingdi Yuan and Ben Kybartas and Tavian Barnes and Emery Fine and James Moore and Ruo Yu Tao and Matthew Hausknecht and Layla El Asri and Mahmoud Adada and Wendy Tay and Adam Trischler},
  title = {TextWorld: A Learning Environment for Text-based Games},
  journal = {CoRR},
  volume = {abs/1806.11532},
  year = {2018}
}
@article{jansen2022textworldexpress,
  url = {https://arxiv.org/abs/2208.01174},
  author = {Jansen, Peter A. and Côté, Marc-Alexandre},
  title = {TextWorldExpress: Simulating Text Games at One Million Steps Per Second},
  journal = {arXiv},
  year = {2022},
}
@inproceedings{hausknecht2020interactive,
  title={Interactive fiction games: A colossal adventure},
  author={Hausknecht, Matthew and Ammanabrolu, Prithviraj and C{\^o}t{\'e}, Marc-Alexandre and Yuan, Xingdi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  year={2020}
}
@inproceedings{ALFWorld20,
               title ={{ALFWorld: Aligning Text and Embodied Environments for Interactive Learning}},
               author={Mohit Shridhar and Xingdi Yuan and Marc-Alexandre C\^ot\'e and Yonatan Bisk and Adam Trischler and Matthew Hausknecht},
               booktitle = {Proceedings of the International
               Conference on Learning Representations (ICLR)},
               year = {2021},
               url = {https://arxiv.org/abs/2010.03768}}
@misc{scienceworld2022,
    title={ScienceWorld: Is your Agent Smarter than a 5th Grader?},
    author={Ruoyao Wang and Peter Jansen and Marc-Alexandre C{\^o}t{\'e} and Prithviraj Ammanabrolu},
    year={2022},
    eprint={2203.07540},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2203.07540}
}
```

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
This framework does not collect user's personal data. For more information about Microsoft's privacy policies. Please see [Microsoft Privacy Statement](https://www.microsoft.com/en-ca/privacy/privacystatement).

## Responsible AI
Please see our [Responsible AI Statement](https://github.com/microsoft/tale-suite/blob/main/RESPONSIBLE_AI.md).