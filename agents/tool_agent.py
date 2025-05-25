import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)
from typing import List, Dict, Optional, Union

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, make a choice between the following tools to decide what to do next. You should only invoke one tool at a time."
    " \nAct: Perform an action in the game. When doing this, make sure to provide a *single* short phrase to interact with the game, e.g. `Act: get lamp` (without the backticks)."
    " \nReason: Think reflectively over your past experience in the game and consider the best next action. Unless you record your thoughts, you will only have access to them until your next action. When doing this, declare the tool and follow with your step-by-step reasoning, e.g. `Reason: It's dark, so I probably need to find a lamp. I saw a lamp...` (without the backticks)."
    " \nRecord: Add the previous 'Reason' output to a scratchpad. This scratchpad will only be appended to your most recent observation, however will also keep all previous recorded reasoning outputs. When doing this, declare the tool without any additional text, e.g. `Record:` (without the backticks)."
    " \nWhen stuck, try using the `help` command to see what commands are available.")


class LLMToolAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.conversation = kwargs["conversation"]
        self.scratchpad = "I haven't recorded anything yet."
        self.llm_self_obs = ""
        self.tool_mode = ""
        self.llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 100,  # Text actions are short phrases.
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
        ]:
            # For these models, we cannot set the seed.
            self.llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            self.llm_kwargs.pop("seed")
            self.llm_kwargs["max_output_tokens"] = self.llm_kwargs.pop("max_tokens")

    @property
    def uid(self):
        return (
            f"LLMToolAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "tool",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces the response to be computed.
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        # Input from act:
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def llm_query_and_record_stats(self, latest_obs, tool_use):
        # Wrapper function since we need to run this a couple times. Take in a latest 'assistant' message and send it up to the LLM.
        # Returns stripped text (response_text) and relevant info (dictionary)
        messages = self.build_messages(latest_obs)
        # Record metrics:
        response = self._llm_call_from_messages(messages, **self.llm_kwargs)
        response_text = response.text().strip()
        info = {
            "prompt": format_messages_to_markdown(messages),
            "thinking": self.llm_self_obs,
            "response": response.text(),
            "nb_tokens": self.token_counter(messages=messages, text=response.text()),
            "notepad": self.scratchpad,
            "tool": tool_use,
            "last_turn": latest_obs
        }

        return response_text, info


    def handle_tool_use(self, raw_obs: str, obs_rep:str, response_text:str):
        # Should be immediately called after LLM is queried. 
        # 
        # Behaviors:
        # Act: Pass it directly to the environment
        # Reason: The LLM's reasoning output should be in the response, so directly store it and query the model again
        # Record: Save reasoning output to the notepad and then query the model again
        # 
        # Unless the models fails to follow instructions, this should (theoretically) never require more than two calls. We still track stats just in case.
        # If the command does not match 'act:', 'reason:' or 'record:', then we assume the agent is just trying to act
        # 
        # input: Response object from LLM call: Response, obs_rep: str (for throwing out reasoning output)
        # output: next action in environment: str, infos about tool use to be added to stats: array of dictionaries (one info for each step)
        infos = []
        if 'Act:' in response_text:
            # Just extract the act, no need to calculate usage statistics, since its done in the main act method. This case is basically same as regular LLMAgent.
            return response_text.split("Act:")[-1].strip(), []
        elif 'Reason:' in response_text:
            # For reasoning, we need to calculate the reasoning trace, any failed parses and then return the action for calculation through the original 'act' function.
            # Get the reasoning trace (should be immediate output)
            reasoning_trace = response_text.split("Reason:")[-1].strip()

            # Return the message back to the llm to get the actual action
            llm_self_obs = raw_obs + "\n" + f"<SCRATCHPAD>{self.scratchpad}<SCRATCHPAD>" + "\n<THINKING>" + reasoning_trace + f"<THINKING>\nNow provide your next action in a single short phrase, e.g. `get lamp` (without the backticks).\nOr record your thoughts if you want to save them for later.\n> "
            self.llm_self_obs = llm_self_obs

            response_text, info = self.llm_query_and_record_stats(llm_self_obs, "Reason")
            infos.append(info)

            # Being nice here:
            if 'record' in response_text.lower():
                post_record_self_obs = f"{raw_obs}\n\n Your latest thought is: {reasoning_trace}\n In your scratchpad, you have recorded: {self.scratchpad}\n You rewrite your scratchpad. What is recorded in your scratchpad now?"
                response_text, info = self.llm_query_and_record_stats(post_record_self_obs, "Record")
                infos.append(info)
                self.scratchpad = response_text.replace("Record: ", "")

                # Extract the observation from the original and rebuild the latest observation
                new_obs_rep = raw_obs + "\n\n<SCRATCHPAD>" + self.scratchpad + "<SCRATCHPAD>\nYou've rewritten your scratch pad. Now provide your next action in a single short phrase, e.g. `get lamp` (without the backticks).\n>"
                response_text, info = self.llm_query_and_record_stats(new_obs_rep, "Act")
                infos.append(info)

            action = response_text.split("Act: ")[-1].strip()
            self.history.append((f"{obs_rep}\n> ", f"{action}\n"))

            return action, infos
        else:
            return response_text, []


    def act(self, obs, reward, done, infos):
        # Add the notepad to the observation
        obs_rep = f"{obs}\n<SCRATCHPAD>{self.scratchpad}<SCRATCHPAD>\n> "

        # Reset the self observation (reflective observation?) at each step
        self.llm_self_obs = ""
        
        response_text, info = self.llm_query_and_record_stats(obs_rep, "NA")
        
        action, infos = self.handle_tool_use(obs, obs_rep, response_text)
        self.history.append((f"{obs}\n> ", f"{action}\n"))
        infos.insert(0, info)

        # Build tool use stat:
        # Potential cases:
        # Case 1: just action -> No thinking trace, no changes to notepad
        # Case 2: Think then action -> Thinking trace, no changes to notepad
        # Case 3: Think, record, then act -> Thinking trace and new notepad.
        # 
        # prompt: Only use the last one
        # response: Let this be the final action
        # thinking: if a thinking trace is generated
        # nb_tokens: Same as usual
        # Tool use: string with all tool uses
        # Notepad: Current notepad
        # All_responses: Reconstructs the conversation turns otherwise thrown away. This is the thinking and record trace
        act_info = infos[-1]
        stats = {
            "prompt": act_info["prompt"],
            "response": act_info["response"],
            "thinking": "",
            "nb_tokens": 0,
            "tool_use": [],
            "notepad": self.scratchpad,
            "all_responses": ""
        }
        for info in infos:
            stats['nb_tokens'] += info['nb_tokens']
            stats['tool_use'].append(info['tool'])
            stats['all_responses'] += "\nInput: " + info['last_turn'] + "\nOutput: " + info['response']

        stats['tool_use'] = ",".join(stats['tool_use'])

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="tool-agent",
    desc=(
        "This agent uses thinking and a scratchpad to progress through the game."
    ),
    klass=LLMToolAgent,
    add_arguments=build_argparser,
)
