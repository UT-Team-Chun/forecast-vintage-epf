"""Phase 106: express the two-target asymmetry in COMMENSURABLE units.

Why this exists
---------------
The article's headline asymmetry is a pair of variance shares: the excluded run's
innovation explains 14.0% of the physical residual's variance against 1.2% of the
price residual's. Three independent referees objected, and they are right: those
are R-squared-like shares taken on two DIFFERENT dependent variables with very
different residual variances, so their ratio is not a meaningful comparison. A
share of a small variance and a share of a large one cannot be divided.

The fix is to compare the same thing on both targets: the fraction by which each
target's own mean absolute error falls. That is commensurable by construction,
because each is normalised by its own scale.

Both legs are read from the artefacts the article already publishes, and the arm
labels are asserted to match so the ratio cannot silently mix measurements.

Read-only over existing provenance; fits nothing.
"""

from __future__ import annotations

import json
import subprocess

from alg.config import EpfConfig

CFG = EpfConfig()
RUN = CFG.project_root / "data" / "runs"

# The price leg: the gate-timing delta on the capacity-weighted primary arm, over
# that arm's own pre-gate mean absolute error.
P85 = RUN / "phase85_exact_ladder.json"
P82B = RUN / "phase82b_absolute_metrics.json"
# The physical leg: already computed commensurably by phase101.
P101 = RUN / "phase101_physical_normalisation.json"

ARM = "arm_cw_primary"
TIER = "msm_d2_12z_cw"


def main() -> None:
    p85 = json.loads(P85.read_text())
    p82b = json.loads(P82B.read_text())
    p101 = json.loads(P101.read_text())

    gt = p85[ARM]["gate46_47_48_gate_timing"]
    delta_price = float(gt["delta"])
    mae_price = float(p82b["relative"]["all"]["tiers"][TIER]["mae"])

    # Both legs must describe the SAME arm and tier, or the ratio mixes panels.
    assert TIER in p82b["relative"]["all"]["tiers"], "pre-gate tier absent from phase82b"
    assert p101["panel"]["pre_gate_tier"] == TIER, (
        f"phase101 pre-gate tier {p101['panel']['pre_gate_tier']} != {TIER}")

    price_pct = 100.0 * delta_price / mae_price
    phys_pct = float(p101["q2_relative_improvement"]["pooled_ratio_of_means_pct"])

    out = {
        "phase": 106,
        "title": "The two-target asymmetry in commensurable units",
        "why": ("variance shares on two different dependent variables are not "
                "comparable; fractional error reduction is"),
        "price_leg": {
            "delta_jpy_kwh": delta_price,
            "own_mae_jpy_kwh": mae_price,
            "relative_reduction_pct": price_pct,
            "arm": ARM, "tier": TIER,
            "sources": [str(P85.name), str(P82B.name)],
        },
        "physical_leg": {
            "relative_reduction_pct": phys_pct,
            "ci95_pct": p101["q2_relative_improvement"]["pooled_ratio_of_means_ci95_pct"],
            "source": str(P101.name),
        },
        "ratio_physical_over_price": phys_pct / price_pct,
        "for_contrast_the_variance_shares": {
            "physical_pct": 14.0, "price_pct": 1.2, "ratio": 14.0 / 1.2,
            "note": ("printed by the article today; retained here only so the "
                     "difference between the two framings is visible"),
        },
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=CFG.project_root).stdout.strip(),
        "script": "scripts/55_commensurable_asymmetry.py",
    }
    dest = RUN / "phase106_commensurable_asymmetry.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"price:    {delta_price:+.4f} / {mae_price:.3f} = {price_pct:.3f}%")
    print(f"physical: {phys_pct:.3f}%  (CI {out['physical_leg']['ci95_pct']})")
    print(f"ratio physical/price = {out['ratio_physical_over_price']:.1f}x")
    print(f"variance-share ratio = {14.0/1.2:.1f}x  <- the framing being replaced")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
