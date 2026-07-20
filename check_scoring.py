"""Test the REAL scoring prompts at old vs new token budgets.

    python check_scoring.py

Six small calls (~cents). Answers one question definitively: are the scoring-call
failures caused by max_tokens being too small for a reply that arrives wrapped in
markdown fences?

Background: this OpenRouter route does not honour response_format, so every reply
carries ```json fences costing ~8 tokens that were supposed to be absent. The old
budgets (prior=120, relatedness=60, lift=200) were sized assuming bare JSON.
"""
import sys

from counterpoint.config import Settings
from counterpoint.scoring import PRIOR_PROMPT, REL_PROMPT, LIFT_PROMPT
from counterpoint.llm import extract_json

H = "The new supplier's flour is defective."
D = "Only loaves baked with the new supplier's flour came out overbaked."

# (label, prompt, old_budget, new_budget)
CASES = [
    ("prior", PRIOR_PROMPT.replace("{stmt}", D), 120, 400),
    ("relatedness", REL_PROMPT.replace("{h}", H).replace("{d}", D), 60, 400),
    ("lift", LIFT_PROMPT.replace("{h}", H).replace("{d}", D), 200, 600),
]


def call(client, model, prompt, max_tokens):
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system",
                       "content": "Reply with a single JSON object. "
                                  "No prose, no markdown fences."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"})
        c = r.choices[0]
        content = c.message.content
        used = r.usage.completion_tokens if getattr(r, "usage", None) else None
        if not content:
            return False, c.finish_reason, used, "EMPTY content"
        try:
            extract_json(content)
            return True, c.finish_reason, used, "parsed ok"
        except Exception as e:
            return False, c.finish_reason, used, f"parse failed: {e}"
    except Exception as e:
        return False, "-", None, f"{type(e).__name__}: {e}"


def main():
    s = Settings()
    if not s.openrouter_api_key:
        print("FAIL: no OPENROUTER_API_KEY")
        return 1
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=s.openrouter_api_key, timeout=60.0, max_retries=0,
                    default_headers={"X-Title": "counterpoint-budget-check"})
    print(f"model: {s.model}\n")
    print(f"{'call':14s}{'budget':>8s}{'ok':>5s}{'finish':>10s}{'used':>6s}  note")
    print("-" * 74)
    old_fail = new_fail = 0
    for label, prompt, old, new in CASES:
        for tag, budget in (("OLD", old), ("NEW", new)):
            ok, finish, used, note = call(client, s.model, prompt, budget)
            if not ok:
                if tag == "OLD":
                    old_fail += 1
                else:
                    new_fail += 1
            print(f"{label+' '+tag:14s}{budget:>8d}{'Y' if ok else 'N':>5s}"
                  f"{str(finish):>10s}{str(used):>6s}  {note[:28]}")
    print("-" * 74)
    print(f"old budgets failed: {old_fail}/3   new budgets failed: {new_fail}/3")
    print()
    if old_fail and not new_fail:
        print("CONFIRMED: the small budgets were the cause. Apply the new scoring.py.")
    elif not old_fail and not new_fail:
        print("NOT CONFIRMED: both budgets work here. The failures come from somewhere")
        print("else — capture a failing call's raw content before changing anything.")
    else:
        print("Mixed/unclear: see the finish_reason and used-token columns above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
