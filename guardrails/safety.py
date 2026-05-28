"""Input/output safety guardrails."""

import re
from typing import Tuple

# Patterns that signal jailbreak attempts or harmful requests
_INPUT_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|constraints?)",
    r"\bDAN\b.*?(do anything now|no restrictions?)",
    r"pretend\s+(you\s+)?(have\s+no|there\s+are\s+no)\s+(restrictions?|rules?|guidelines?|limits?)",
    r"you\s+are\s+now\s+(a\s+)?(?:uncensored|unrestricted|jailbroken|evil|unfiltered)",
    r"act\s+as\s+(if\s+you\s+were\s+)?(an?\s+)?(unrestricted|uncensored|evil|malicious)",
    r"(how\s+to|steps?\s+to|instructions?\s+(for|to))\s+(make|build|create|synthesize|produce)\s+(a\s+)?(bomb|explosive|nerve\s+agent|bioweapon|meth|fentanyl|ricin|sarin|anthrax)",
    r"(hack|exploit|crack|bypass)\s+(into\s+)?(bank|hospital|power\s+grid|government|school)",
    r"\b(csam|child\s+porn|child\s+sexual\s+abuse)\b",
    r"(provide|give\s+me|write)\s+(a\s+)?(malware|ransomware|keylogger|trojan|virus)\s+(code|script|program)",
    r"(suicide|self.harm)\s+(method|instruction|how\s+to)",
]

# Compiled once at import time
_INPUT_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INPUT_PATTERNS]

# Patterns in output that indicate the model produced dangerous content
_OUTPUT_PATTERNS = [
    r"(step\s+\d+|first,?\s+|then,?\s+).{0,80}(detonate|synthesize\s+the|mix\s+the\s+(chemicals?|substances?))",
    r"(to\s+make|to\s+build|to\s+create)\s+(a\s+)?(bomb|explosive|bioweapon|nerve\s+agent)",
    r"(download|install|run)\s+.{0,60}(malware|ransomware|payload|keylogger)",
]
_OUTPUT_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _OUTPUT_PATTERNS]

REFUSAL_MESSAGE = (
    "I'm sorry, but I can't help with that request. "
    "Please ask me something else — I'm happy to assist with a wide range of topics."
)

OUTPUT_REFUSAL = (
    "I'm not able to provide that information as it may be harmful. "
    "Please feel free to ask me something else."
)


class SafetyChecker:
    """Stateless safety checker. Wrap assistant calls with check_input/check_output."""

    def check_input(self, text: str) -> Tuple[bool, str]:
        """Returns (is_safe, reason). is_safe=False means block the input."""
        for pattern in _INPUT_RE:
            if pattern.search(text):
                return False, f"Matched harmful input pattern: {pattern.pattern[:60]}"
        return True, ""

    def check_output(self, text: str) -> Tuple[bool, str]:
        """Returns (is_safe, reason). is_safe=False means replace the output."""
        for pattern in _OUTPUT_RE:
            if pattern.search(text):
                return False, f"Matched harmful output pattern: {pattern.pattern[:60]}"
        return True, ""

    def safe_response_for_blocked_input(self) -> str:
        return REFUSAL_MESSAGE

    def safe_response_for_blocked_output(self) -> str:
        return OUTPUT_REFUSAL
