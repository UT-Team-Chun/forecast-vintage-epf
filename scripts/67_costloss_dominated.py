"""Phase 108: the cost--loss increment against a fallback an actor would actually use.

The defect
----------
Section 5.4 converts the floor result into Murphy's cost--loss value score and
reports the weather channels' contribution as V(with) - V(without), headlining the
cost--loss ratio at which that difference is largest (alpha = 0.64, +0.361).

Above alpha = 0.42 the without-weather forecast has a NEGATIVE value score: it is
worse than always or never protecting, so no actor would use it. They would fall
back to climatology, whose value score is zero by construction. Differencing
against a strictly dominated alternative therefore credits the weather channels
with the dominated forecast's losses, and the headline alpha is precisely where
that credit is largest.

Two consequences, both of which the article got wrong:
  1. Against max(0, V_without) -- the best an actor can do without weather -- the
     increment peaks at alpha = 0.42, not 0.64.
  2. V_with falls steeply in alpha, so the vintage is worth MOST to actors whose
     protection is cheap relative to their exposure. The article concluded the
     opposite, that it serves actors with expensive protection, which inverted the
     population its own curve identifies.

A referee found this. Read-only over phase97's committed value curves.
"""

from __future__ import annotations

import json
import subprocess

from alg.config import EpfConfig

CFG = EpfConfig()
RUN = CFG.project_root / "data" / "runs"
P97 = RUN / "phase97_cost_loss.json"
OUT = RUN / "phase108_costloss_dominated_benchmark.json"


def main() -> None:
    d = json.loads(P97.read_text())
    with_wx = d["models"]["m2"]["value_curve"]
    without = d["models"]["m2p"]["value_curve"]
    raw = d["weather_increment_in_value"]["difference"]

    alphas = sorted((a for a in with_wx if with_wx[a] is not None), key=float)
    rows = []
    for a in alphas:
        vw, vo = float(with_wx[a]), float(without[a])
        rows.append({
            "alpha": float(a),
            "v_with": vw,
            "v_without": vo,
            "increment_raw": (None if raw.get(a) is None else float(raw[a])),
            # what an actor can actually achieve without weather: climatology is
            # always available and scores zero, so a negative score is never chosen
            "increment_vs_usable_fallback": vw - max(0.0, vo),
        })

    dominated = [r["alpha"] for r in rows if r["v_without"] < 0.0]
    peak_raw = max(rows, key=lambda r: (r["increment_raw"] is not None, r["increment_raw"] or -9))
    peak_fix = max(rows, key=lambda r: r["increment_vs_usable_fallback"])
    v_first, v_last = rows[0]["v_with"], rows[-1]["v_with"]

    out = {
        "phase": 108,
        "title": "Cost--loss increment against a fallback an actor would use",
        "defect": ("the published increment differences against a forecast whose "
                   "value score is negative above alpha = "
                   f"{min(dominated):.2f}, i.e. strictly dominated by climatology"),
        "n_alpha": len(rows),
        "alpha_where_without_weather_turns_negative": min(dominated) if dominated else None,
        "n_alpha_dominated": len(dominated),
        "peak_raw_difference": {"alpha": peak_raw["alpha"],
                                "value": peak_raw["increment_raw"]},
        "peak_vs_usable_fallback": {"alpha": peak_fix["alpha"],
                                    "value": peak_fix["increment_vs_usable_fallback"]},
        "v_with_at_lowest_alpha": {"alpha": rows[0]["alpha"], "value": v_first},
        "v_with_at_highest_alpha": {"alpha": rows[-1]["alpha"], "value": v_last},
        "v_with_fold_decline_low_to_high": v_first / v_last if v_last else None,
        "reading": ("V_with declines steeply in alpha, so the vintage is worth most "
                    "to actors whose protection is CHEAP relative to exposure. The "
                    "article concluded the opposite"),
        "curve": rows,
        "source": P97.name,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=CFG.project_root).stdout.strip(),
        "script": "scripts/67_costloss_dominated.py",
    }
    OUT.write_text(json.dumps(out, indent=2))

    print(f"without-weather score turns negative at alpha = {min(dominated):.2f} "
          f"({len(dominated)} of {len(rows)} grid points dominated)")
    print(f"peak of raw difference        : alpha {peak_raw['alpha']:.2f}  "
          f"{peak_raw['increment_raw']:+.3f}   <- what the article headlines")
    print(f"peak vs usable fallback       : alpha {peak_fix['alpha']:.2f}  "
          f"{peak_fix['increment_vs_usable_fallback']:+.3f}")
    print(f"V_with: {v_first:.3f} at alpha {rows[0]['alpha']:.2f} -> "
          f"{v_last:.3f} at alpha {rows[-1]['alpha']:.2f} "
          f"({v_first/v_last:.0f}x decline)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
