"""Standalone LLM connectivity check — run this before any real-model run.

    python check_llm.py
    python check_llm.py --model openai/gpt-5

Makes ONE call and prints the real error if it fails. This exists because the
scorer degrades gracefully on failure (returns {}), which is right for an isolated
bad response but hides a systemic fault — a wrong key or model id would otherwise
show up only as an empty corpus and a page of zeros.
"""
import argparse
import sys

from counterpoint.config import Settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    s = Settings()
    model = args.model or s.model

    print(f"key present : {'yes' if s.openrouter_api_key else 'NO'}", flush=True)
    if s.openrouter_api_key:
        k = s.openrouter_api_key
        print(f"key looks like: {k[:8]}...{k[-4:]}  (len {len(k)})", flush=True)
    print(f"force_mock  : {s.force_mock}", flush=True)
    print(f"model       : {model}", flush=True)
    print()

    if not s.openrouter_api_key:
        print("FAIL: no OPENROUTER_API_KEY found in environment or .env")
        return 1
    if s.force_mock:
        print("NOTE: COUNTERPOINT_FORCE_MOCK is set — the app would use the mock LLM.")

    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: the 'openai' package is not installed.  pip install openai")
        return 1

    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=s.openrouter_api_key,
                    default_headers={"X-Title": "counterpoint-check"})
    print("calling the model (one short request) ...", flush=True)
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=50,
            messages=[{"role": "system", "content": "Reply with a single JSON object."},
                      {"role": "user", "content": 'Return exactly: {"ok": true}'}])
        content = r.choices[0].message.content
        print(f"RAW RESPONSE: {content!r}")
        if not content:
            print("\nFAIL: the model returned EMPTY content. Try a different model id.")
            return 1
        print("\nOK: the model answered. Real-model runs should work.")
        return 0
    except Exception as e:
        print(f"\nFAIL: {type(e).__name__}: {e}")
        msg = str(e).lower()
        if "401" in msg or "auth" in msg or "credential" in msg:
            print("HINT: the API key is rejected — check OPENROUTER_API_KEY in .env")
        elif "404" in msg or "not found" in msg or "model" in msg:
            print(f"HINT: the model id '{model}' may be wrong. Check "
                  "https://openrouter.ai/models for the exact string.")
        elif "402" in msg or "credit" in msg or "quota" in msg or "insufficient" in msg:
            print("HINT: the account appears to be out of credit.")
        elif "connect" in msg or "timeout" in msg or "ssl" in msg or "proxy" in msg:
            print("HINT: network/proxy problem reaching openrouter.ai")
        return 1


if __name__ == "__main__":
    sys.exit(main())
