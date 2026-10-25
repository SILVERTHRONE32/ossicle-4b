"""Question layouts for multiple-choice benchmarks."""
import re

LETTERS = "ABCDEFGHIJKL"
_strip = lambda c: re.sub(r"^\(?[A-Da-d]\)\s*", "", str(c)).strip()


def options_prompt(question, choices):
    """Option text layout (default)."""
    opts = "\n".join(f"- {_strip(c)}" for c in choices)
    return f"{question}\nOptions:\n{opts}\nReply with the exact text of the one correct option, and nothing else."


def letter_prompt(question, choices):
    """Lettered layout; map the reply with `letter_to_option`."""
    n = len(choices)
    names = ", ".join(LETTERS[: n - 1]) + ", and " + LETTERS[n - 1]
    opts = "\n".join(f"{LETTERS[i]}. {_strip(c)}" for i, c in enumerate(choices))
    return (f"Choose the most suitable answer from options {names} to respond to the question in the next line. "
            f"You should only choose {', '.join(LETTERS[: n - 1])}, or {LETTERS[n - 1]}. "
            f"Do not provide any additional explanations or content.\nQuestion: {question}\n{opts}")


def letter_to_option(reply, choices):
    m = re.match(r"^\W*(?:the\s+(?:correct\s+)?(?:answer|option)\s+is\s+)?\(?([A-L])\)?(?:[.:)\s]|$)", reply, re.I)
    if m and LETTERS.index(m.group(1).upper()) < len(choices):
        return str(choices[LETTERS.index(m.group(1).upper())])
    return reply
