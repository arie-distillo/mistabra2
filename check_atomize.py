"""Isolate ONE atomization call and print exactly what the model returns.

    python check_atomize.py
    python check_atomize.py --model openai/gpt-5 --max-tokens 4000

Why this exists: atomization fails inside the server, and the server's traceback has
not been reaching the client log. This bypasses the whole stack — no FastAPI, no
threads, no fail-fast — and shows the raw HTTP response, so the cause is visible
rather than inferred.
"""
import argparse
import sys

from counterpoint.config import Settings
from counterpoint.atomize import ATOMIZE_PROMPT

PASSAGE = ("The oven temperature sensors reported stable readings all night. "
           "No fault codes were raised by the controller. "
           "The bakery switched to a new flour supplier in November. "
           "Deliveries from that supplier arrived late three times.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--no-json-format", action="store_true",
                    help="omit response_format (some models/routes reject it)")
    args = ap.parse_args()

    s = Settings()
    model = args.model or s.model
    print(f"model      : {model}")
    print(f"max_tokens : {args.max_tokens}")
    print(f"key present: {'yes' if s.openrouter_api_key else 'NO'}")
    print(f"prompt size: {len(ATOMIZE_PROMPT)} chars")
    print()
    if not s.openrouter_api_key:
        print("FAIL: no OPENROUTER_API_KEY")
        return 1

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=s.openrouter_api_key,
                    default_headers={"X-Title": "counterpoint-atomize-check"})

    kw = dict(model=model, temperature=0, max_tokens=args.max_tokens,
              messages=[{"role": "system",
                         "content": "Reply with a single JSON object. "
                                    "No prose, no markdown fences."},
                        {"role": "user",
                         "content": ATOMIZE_PROMPT.replace("{passage}", PASSAGE)}])
    if not args.no_json_format:
        kw["response_format"] = {"type": "json_object"}

    print("calling ...", flush=True)
    try:
        r = client.chat.completions.create(**kw)
    except Exception as e:
        print(f"\nEXCEPTION: {type(e).__name__}: {e}")
        return 1

    choice = r.choices[0]
    content = choice.message.content
    print(f"finish_reason : {choice.finish_reason}")
    print(f"content is None: {content is None}")
    print(f"content length : {0 if content is None else len(content)}")
    usage = getattr(r, "usage", None)
    if usage:
        # If completion_tokens ~= max_tokens while content is empty, the budget went
        # to reasoning rather than output — that is the empty-content failure mode.
        print(f"usage          : {usage}")
    print()
    print("RAW CONTENT:")
    print(repr(content))
    print()
    if not content:
        print("DIAGNOSIS: empty content.")
        print("  - if finish_reason is 'length', the budget was exhausted -> raise "
              "--max-tokens")
        print("  - if usage shows reasoning tokens, the model spent the budget "
              "thinking")
        print("  - try --no-json-format: some routes return empty with "
              "response_format")
        return 1
    try:
        # Use the SAME parser the system uses. json.loads() alone reports a false
        # failure on fenced output, which the pipeline handles fine.
        from counterpoint.llm import extract_json
        data = extract_json(content)
        dps = data.get("data_points", [])
        print(f"PARSED OK: {len(dps)} data points from 4 sentences")
        for d in dps:
            print("  -", d)
        if len(dps) > 8:
            print("\nNOTE: more than 2 per sentence — still over-splitting.")
        return 0
    except Exception as e:
        print(f"JSON PARSE FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
