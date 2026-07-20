"""Load the YAML test scenarios and (optionally) inject noise.

The scenarios ship with an ACH `ground_truth` block (worst_cell, elimination,
verdicts). We IGNORE that apparatus for scoring — the product is not an ACH tool.
We keep only:
  * episodes  -> corpus documents (each carries credibility + metadata)
  * claims    -> hypotheses
  * ground_truth.claims -> a light sanity oracle for tests (e.g. "H3 should be
    among the most-contradicted"), used only to assert directional sanity, never
    fed to the scorer.

Noise injection stress-tests the matrix: does it keep irrelevant data points
neutral (lift ~1, low diagnosticity) instead of letting them pollute verdicts?
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Generic off-topic sentences — should stay neutral under every hypothesis.
NOISE_SENTENCES = [
    "The office cafeteria introduced a new seasonal menu featuring pumpkin soup.",
    "Quarterly fire-drill training was completed by all staff on the third floor.",
    "The parking garage will be repainted over the coming bank holiday weekend.",
    "A new espresso machine was installed in the fourth-floor break room.",
    "The company softball team lost its semi-final match on Saturday afternoon.",
    "IT reminded everyone to update their passwords before the end of the month.",
    "The reception area now displays a rotating exhibit of local student artwork.",
    "Recycling bins were relocated from the lobby to the loading dock corridor.",
]


@dataclass
class Scenario:
    name: str
    domain: str
    episodes: list[dict]      # {content, credibility, metadata, source_type}
    hypotheses: list[dict]    # {text, context_summary}
    ground_truth: dict        # kept only as a sanity oracle for tests


def load_scenario(path: str | Path) -> Scenario:
    d = yaml.safe_load(Path(path).read_text())
    episodes = []
    for e in d.get("episodes", []):
        episodes.append({
            "content": e["content"].strip(),
            "credibility": e.get("credibility", 1.0),
            "source_type": e.get("source_type", "CONTEXT"),
            "metadata": e.get("metadata", {}),
        })
    hyps = [{"text": c["text"].strip(),
             "context_summary": c.get("context_summary", "").strip()}
            for c in d.get("claims", [])]
    return Scenario(name=d.get("name", ""), domain=d.get("domain", ""),
                    episodes=episodes, hypotheses=hyps,
                    ground_truth=d.get("ground_truth", {}))


def all_scenarios() -> list[Scenario]:
    """Files that are not scenario-shaped (e.g. capability-test case files, which are
    lists) are skipped rather than crashing the loader — data/ is a shared directory."""
    out = []
    for p in sorted(DATA_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict) or "episodes" not in raw:
            continue
        out.append(load_scenario(p))
    return out

def inject_noise(scenario: Scenario, n: int = 4, seed: int = 0,
                 duplicate: int = 1) -> Scenario:
    """Return a copy of the scenario with noise episodes added:
      * `n` off-topic sentences (credibility 0.3), tagged noise=irrelevant
      * `duplicate` near-duplicate of an existing episode, tagged noise=duplicate
    """
    rng = random.Random(seed)
    eps = list(scenario.episodes)
    for i in range(n):
        eps.append({
            "content": rng.choice(NOISE_SENTENCES),
            "credibility": 0.3,
            "source_type": "OTHER",
            "metadata": {"noise": "irrelevant", "idx": i},
        })
    for i in range(duplicate):
        if scenario.episodes:
            src = rng.choice(scenario.episodes)
            eps.append({
                "content": src["content"] + " (duplicate record from a second system).",
                "credibility": src["credibility"],
                "source_type": src["source_type"],
                "metadata": {"noise": "duplicate", "idx": i},
            })
    return Scenario(name=scenario.name + " +noise", domain=scenario.domain,
                    episodes=eps, hypotheses=scenario.hypotheses,
                    ground_truth=scenario.ground_truth)


def load_into_pipeline(pipeline, scenario: Scenario):
    """Add a scenario's episodes and hypotheses into a Pipeline."""
    for e in scenario.episodes:
        meta = dict(e["metadata"])
        meta["credibility"] = e["credibility"]
        meta["source_type"] = e["source_type"]
        pipeline.add_episode(e["content"], meta=meta)
    hs = [pipeline.add_hypothesis(h["text"], h["context_summary"])
          for h in scenario.hypotheses]
    return hs
