"""Optional Copilot-backed observation rewriting for TALES agents."""

from __future__ import annotations

import atexit
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import llm
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from tales.utils import is_recoverable_error

DEFAULT_OBSERVATION_REWRITE_INSTRUCTION = (
    "Please rewrite this observation in a more serious tone. "
    "Preserve all information and return only the rewritten observation."
)
ZORK_KEYWORD_ALIAS_PROFILE = "zork-keywords-v1"
ZORK_KEYWORD_ALIASES = (
    ("great underground empire", "velorian-underrealm"),
    ("serial number 840726", "archive code 530194"),
    ("revision 88", "edition 41"),
    ("flood control dam #3", "narvex-complex-seven"),
    ("flood control dam", "narvex-complex"),
    ("white house", "ivorak-dwelling"),
    ("frigid river", "selmora-current"),
    ("aragain falls", "oravain-cascade"),
    ("land of the dead", "mortessa-realm"),
    ("entrance to hades", "nethros-gate"),
    ("atlantis room", "nerath-chamber"),
    ("end of rainbow", "prismarch-terminus"),
    ("on the rainbow", "prismarch-span"),
    ("troll room", "dravik-chamber"),
    ("cyclops room", "monoptor-chamber"),
    ("coal mine", "keldran-excavation"),
    ("living room", "solivar-chamber"),
    ("east-west passage", "traversan-corridor"),
    ("round room", "orbivar-chamber"),
    ("strange passage", "velstran-corridor"),
    ("treasure room", "reliquar-chamber"),
    ("dam lobby", "narvex-vestibule"),
    ("maintenance room", "servitor-chamber"),
    ("mirror room", "specular-chamber"),
    ("loud room", "resonara-chamber"),
    ("skeleton key", "ossikar-key"),
    ("platinum bar", "argentor-ingot"),
    ("pot of gold", "aurikon-cache"),
    ("jewel-encrusted egg", "gemovrix"),
    ("gold coffin", "aurelsarc"),
    ("crystal skull", "vitrecrux"),
    ("crystal trident", "crystavarn"),
    ("jade figurine", "jadovar"),
    ("sapphire-encrusted bracelet", "azurband"),
    ("trunk of jewels", "gemcask"),
    ("leather bag of coins", "delvar-pouch"),
    ("brass lantern", "bronzar-luminar"),
    ("brass bell", "bronzar-chimek"),
    ("magic boat", "naviron-enchanted"),
    ("trophy case", "vitrinox"),
    ("display case", "vitrinox-display"),
    ("nasty knife", "malvex-blade"),
    ("rusty knife", "corvex-blade"),
    ("plastic pile", "foldryn"),
    ("pile of plastic", "foldryn-mass"),
    ("red button", "crimson-pressel"),
    ("yellow button", "amber-pressel"),
    ("front door", "entry-porten"),
    ("trap door", "subport-spaced"),
    ("zorkmids", "delvars"),
    ("zorkmid", "delvar"),
    ("grues", "grunsts"),
    ("grue", "grunst"),
    ("thieves", "skulvens"),
    ("thief", "skulven"),
    ("trolls", "draviks"),
    ("troll", "dravik"),
    ("skeletons", "ossikars"),
    ("skeleton", "ossikar"),
    ("bats", "vaskirs"),
    ("bat", "vaskir"),
    ("mailboxes", "posternoxes"),
    ("mailbox", "posternox"),
    ("leaflets", "vellumets"),
    ("leaflet", "vellumet"),
    ("lanterns", "luminars"),
    ("lantern", "luminar"),
    ("swords", "varkblades"),
    ("sword", "varkblade"),
    ("boats", "navirons"),
    ("boat", "naviron"),
    ("pumps", "aerovexes"),
    ("pump", "aerovex"),
    ("sceptres", "regaliths"),
    ("sceptre", "regalith"),
    ("tridents", "trivarns"),
    ("trident", "trivarn"),
    ("chalices", "caldrises"),
    ("chalice", "caldris"),
    ("coffins", "sarcovens"),
    ("coffin", "sarcoven"),
    ("eggs", "ovrixes"),
    ("egg", "ovrix"),
    ("canaries", "chirrens"),
    ("canary", "chirren"),
    ("scarabs", "kheprans"),
    ("scarab", "khepran"),
    ("emeralds", "verdrixes"),
    ("emerald", "verdrix"),
    ("paintings", "muravals"),
    ("painting", "muraval"),
    ("bells", "chimeks"),
    ("bell", "chimek"),
    ("books", "codrixes"),
    ("book", "codrix"),
    ("candles", "wickars"),
    ("candle", "wickar"),
    ("wrenches", "torqens"),
    ("wrench", "torqen"),
    ("screwdrivers", "turnexes"),
    ("screwdriver", "turnex"),
    ("axes", "cleavorns"),
    ("axe", "cleavorn"),
    ("knives", "edgrixes"),
    ("knife", "edgrix"),
    ("torches", "flamars"),
    ("torch", "flamar"),
    ("skulls", "cranions"),
    ("skull", "cranion"),
    ("figurines", "effigixes"),
    ("figurine", "effigix"),
    ("bracelets", "wristrels"),
    ("bracelet", "wristrel"),
    ("bottles", "flaskens"),
    ("bottle", "flasken"),
    ("matches", "sparkets"),
    ("match", "sparket"),
    ("ropes", "cordaxes"),
    ("rope", "cordax"),
    ("cases", "vaultens"),
    ("case", "vaulten"),
    ("baskets", "weaverns"),
    ("basket", "weavern"),
    ("lamps", "glowrixes"),
    ("lamp", "glowrix"),
    ("piles", "foldryns"),
    ("pile", "foldryn-single"),
    ("diamonds", "clarions"),
    ("diamond", "clarion"),
    ("trapdoors", "subports"),
    ("trapdoor", "subport"),
    ("matchbooks", "sparkcodexes"),
    ("matchbook", "sparkcodex"),
    ("bags", "pouchrens"),
    ("bag", "pouchren"),
    ("buttons", "pressels"),
    ("button", "pressel"),
    ("mirrors", "speculons"),
    ("mirror", "speculon"),
    ("shovels", "spadrixes"),
    ("shovel", "spadrix"),
    ("baubles", "trinkalons"),
    ("bauble", "trinkalon"),
    ("trunks", "caskors"),
    ("trunk", "caskor"),
    ("maps", "chartixes"),
    ("map", "chartix"),
    ("windows", "glassens"),
    ("window", "glassen"),
    ("sacks", "carryns"),
    ("sack", "carryn"),
    ("rugs", "tressels"),
    ("rug", "tressel"),
    ("railings", "balustrans"),
    ("railing", "balustran"),
    ("prayers", "litanyxes"),
    ("prayer", "litanyx"),
    ("bars", "rodrixes"),
    ("bar", "rodrix"),
    ("lids", "coverns"),
    ("lid", "covern"),
    ("doors", "portens"),
    ("door", "porten"),
    ("bolts", "fastenors"),
    ("bolt", "fastenor"),
    ("machines", "mecharas"),
    ("machine", "mechara"),
    ("switches", "actuatorixes"),
    ("switch", "actuatorix"),
    ("coal", "cindrax"),
    ("sand", "siltren"),
    ("odysseus", "ithakor"),
    ("brass", "bronzar"),
    ("zork", "alok"),
    ("frobozz", "varnax"),
    ("flathead", "peldoor"),
    ("aragain", "oravain"),
    ("infocom", "novacore"),
    ("gue", "vur"),
    ("fcd", "nrb"),
    ("hades", "nethros"),
    ("atlantis", "nerathis"),
    ("rainbow", "prismarch"),
    ("cyclops", "monoptor"),
    ("house", "dwellix"),
    ("dam", "narvex"),
    ("river", "fluvex"),
    ("forest", "sylvarn"),
    ("maze", "labyrn"),
    ("temple", "sanctovar"),
    ("altar", "altaryn"),
    ("reservoir", "cisternax"),
    ("canyon", "ravinox"),
    ("chasm", "gulfrax"),
    ("garlic", "zarnel"),
    ("platinum", "argentor"),
    ("sapphire", "azuren"),
    ("crystal", "vitren"),
    ("jade", "jadren"),
    ("gold", "aurel"),
    ("water", "aquen"),
    ("buoy", "floaten"),
    ("label", "tagrix"),
)
KEYWORD_ALIAS_PROFILES = {
    ZORK_KEYWORD_ALIAS_PROFILE: ZORK_KEYWORD_ALIASES,
}


class ObservationRewriteError(RuntimeError):
    """Raised when an observation cannot be rewritten exactly as configured."""


class RewriteBridge(Protocol):
    def call(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class KeywordAliasReplacement:
    source: str
    target: str
    count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "count": self.count,
        }


@dataclass(frozen=True)
class KeywordAliasApplication:
    text: str
    replacements: tuple[KeywordAliasReplacement, ...]

    @property
    def replacement_count(self) -> int:
        return sum(item.count for item in self.replacements)


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.title()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+")
    return re.compile(
        rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def apply_keyword_aliases(
    text: str,
    profile: str | None,
    *,
    reverse: bool = False,
) -> KeywordAliasApplication:
    if profile is None:
        return KeywordAliasApplication(text=text, replacements=())
    try:
        aliases = KEYWORD_ALIAS_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown keyword alias profile: {profile!r}.") from exc

    source_index = 1 if reverse else 0
    ordered = sorted(aliases, key=lambda item: len(item[source_index]), reverse=True)
    output = text
    replacements = []
    for canonical, alias in ordered:
        source, target = (alias, canonical) if reverse else (canonical, alias)
        pattern = _term_pattern(source)
        count = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return _match_case(match.group(0), target)

        output = pattern.sub(replace, output)
        if count:
            replacements.append(
                KeywordAliasReplacement(
                    source=source,
                    target=target,
                    count=count,
                )
            )

    return KeywordAliasApplication(
        text=output,
        replacements=tuple(replacements),
    )


def keyword_alias_profile_sha256(profile: str | None) -> str | None:
    if profile is None:
        return None
    try:
        aliases = KEYWORD_ALIAS_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown keyword alias profile: {profile!r}.") from exc
    payload = json.dumps(
        {"profile": profile, "aliases": aliases},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def observation_rewrite_condition_sha256(
    model: str,
    reasoning_effort: str,
    instruction: str,
    keyword_alias_profile: str | None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "instruction": instruction,
            "keyword_alias_profile": keyword_alias_profile,
            "keyword_alias_profile_sha256": keyword_alias_profile_sha256(
                keyword_alias_profile
            ),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ObservationRewrite:
    text: str
    prompt: str
    model: str
    nb_tokens: int
    nb_tokens_prompt: int
    nb_tokens_response: int
    nb_tokens_thinking: int
    cache_read_tokens: int
    cache_write_tokens: int
    duration: float
    pre_alias_text: str | None = None
    keyword_alias_profile: str | None = None
    keyword_alias_replacements: tuple[KeywordAliasReplacement, ...] = ()

    def as_stats(self) -> dict[str, Any]:
        return {
            "rewritten_observation": self.text,
            "observation_rewrite_prompt": self.prompt,
            "observation_rewrite_model": self.model,
            "observation_rewrite_nb_tokens": self.nb_tokens,
            "observation_rewrite_nb_tokens_prompt": self.nb_tokens_prompt,
            "observation_rewrite_nb_tokens_response": self.nb_tokens_response,
            "observation_rewrite_nb_tokens_thinking": self.nb_tokens_thinking,
            "observation_rewrite_cache_read_tokens": self.cache_read_tokens,
            "observation_rewrite_cache_write_tokens": self.cache_write_tokens,
            "observation_rewrite_duration": self.duration,
            "pre_alias_rewritten_observation": self.pre_alias_text,
            "observation_keyword_alias_profile": self.keyword_alias_profile,
            "observation_keyword_alias_replacements": [
                item.as_dict() for item in self.keyword_alias_replacements
            ],
            "observation_keyword_alias_replacement_count": sum(
                item.count for item in self.keyword_alias_replacements
            ),
        }


class CopilotObservationRewriter:
    """Rewrite each observation in an independent, memory-free model session."""

    def __init__(
        self,
        model: str,
        reasoning_effort: str,
        instruction: str = DEFAULT_OBSERVATION_REWRITE_INSTRUCTION,
        *,
        bridge: RewriteBridge | None = None,
        timeout: float = 600.0,
        keyword_alias_profile: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Observation rewriter model cannot be empty.")
        if not reasoning_effort.strip():
            raise ValueError("Observation rewriter reasoning effort cannot be empty.")
        if not instruction.strip():
            raise ValueError("Observation rewrite instruction cannot be empty.")
        if (
            keyword_alias_profile is not None
            and keyword_alias_profile not in KEYWORD_ALIAS_PROFILES
        ):
            raise ValueError(
                f"Unknown keyword alias profile: {keyword_alias_profile!r}."
            )

        from agents.copilot_capi import CopilotBridge, ExactnessError

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.instruction = instruction
        self.timeout = timeout
        self.keyword_alias_profile = keyword_alias_profile
        self.bridge = bridge or CopilotBridge()
        self._bridge_error = ExactnessError
        self._closed = False
        if bridge is None:
            atexit.register(self.close)

    def build_prompt(self, observation: str) -> str:
        return f"{self.instruction}\n\n{observation}"

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _send(self, prompt: str) -> dict[str, Any]:
        try:
            return self.bridge.call(
                "rewrite",
                {
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "prompt": prompt,
                    "timeout_ms": int(self.timeout * 1000),
                },
                timeout=self.timeout + 60.0,
            )
        except self._bridge_error as exc:
            raise llm.ModelError(str(exc)) from exc

    def rewrite(self, observation: str) -> ObservationRewrite:
        prompt = self.build_prompt(observation)
        started = time.monotonic()
        result = self._send(prompt)
        duration = time.monotonic() - started

        pre_alias_text = str(result.get("text") or "").strip()
        if not pre_alias_text:
            raise ObservationRewriteError(
                "Observation rewriter returned an empty response."
            )
        alias_result = apply_keyword_aliases(
            pre_alias_text,
            self.keyword_alias_profile,
        )
        text = alias_result.text

        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise ObservationRewriteError(
                "Observation rewriter returned no usage telemetry."
            )
        prompt_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        thinking_tokens = int(usage.get("reasoning_tokens") or 0)
        if min(prompt_tokens, output_tokens, thinking_tokens) < 0:
            raise ObservationRewriteError(
                "Observation rewrite token counts must be non-negative."
            )
        if thinking_tokens > output_tokens:
            raise ObservationRewriteError(
                "Observation rewrite reasoning tokens cannot exceed output tokens."
            )

        return ObservationRewrite(
            text=text,
            prompt=prompt,
            model=str(result.get("model") or self.model),
            nb_tokens=prompt_tokens + output_tokens,
            nb_tokens_prompt=prompt_tokens,
            nb_tokens_response=output_tokens - thinking_tokens,
            nb_tokens_thinking=thinking_tokens,
            cache_read_tokens=int(usage.get("cache_read_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
            duration=duration,
            pre_alias_text=pre_alias_text,
            keyword_alias_profile=self.keyword_alias_profile,
            keyword_alias_replacements=alias_result.replacements,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.bridge.close()
