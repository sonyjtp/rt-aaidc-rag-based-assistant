from pathlib import Path
import re
import yaml
from typing import List, Optional, Tuple

from config import METAQUESTIONS_FPATH, PROMPT_CONFIG_FPATH, DEFAULT_NOT_KNOWN_MSG


class MetaPattern:
    def __init__(self, pattern: str, kind: str, response: str):
        self.pattern = pattern
        try:
            self.rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # fallback to literal match
            self.rx = re.compile(re.escape(pattern), re.IGNORECASE)
        self.kind = kind
        self.response = response


class PersonaHandler:
    def __init__(self, config_path: Optional[str] = None):
        cfg_path = Path(config_path) if config_path else None
        meta_cfg_primary = Path(METAQUESTIONS_FPATH)
        prompt_cfg = Path(PROMPT_CONFIG_FPATH)
        if cfg_path and cfg_path.exists():
            self.cfg_path = cfg_path
        elif meta_cfg_primary.exists():
            self.cfg_path = meta_cfg_primary
        else:
            self.cfg_path = prompt_cfg

        self.patterns: List[MetaPattern] = []
        self.allow_self_description = True
        self.default_meta_refusal = DEFAULT_NOT_KNOWN_MSG
        self._load()

    def _load(self):
        if not self.cfg_path.exists():
            return
        try:
            data = yaml.safe_load(self.cfg_path.read_text()) or {}
        except Exception:
            return
        meta_questions = data.get("meta_questions", [])
        for item in meta_questions:
            pattern = item.get("pattern", "")
            kind = item.get("kind", "refuse")
            response = item.get("response", "")
            self.patterns.append(MetaPattern(pattern, kind, response))
        self.allow_self_description = bool(data.get("allow_self_description", True))
        self.default_meta_refusal = data.get("default_meta_refusal", self.default_meta_refusal)

    def is_meta_question(self, query: str) -> Optional[Tuple[str, str]]:
        """Return (kind, response) if query matches a meta pattern, else None."""
        if not query:
            return None
        for p in self.patterns:
            if p.rx.search(query):
                return p.kind, p.response or self.default_meta_refusal
        # simple heuristic: if mentions 'your' or 'you' and 'know' / 'can' / 'capability'
        q = query.lower()
        has_you_reference = "you" in q or "your" in q
        has_capability_words = any(k in q for k in ("know", "can", "capab", "limit", "what do you", "capabilities"))
        if has_you_reference and has_capability_words:
            return "describe", self.patterns[0].response if self.patterns else self.default_meta_refusal
        return None

    def handle_meta_question(self, query: str) -> Optional[str]:
        m = self.is_meta_question(query)
        if not m:
            return None
        kind, response = m
        if kind == "sensitive":
            return self.default_meta_refusal
        if kind == "describe":
            if self.allow_self_description:
                return response
            return self.default_meta_refusal
        # default: refusal
        return self.default_meta_refusal
