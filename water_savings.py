"""
water_savings.py
----------------
Simulate water usage and compute savings vs. a naive baseline.
"""

from typing import List

WATER_MAP = {
    "Irrigate Immediately": 10,
    "Irrigate (Check Conditions)": 10,
    "Moderate Irrigation": 5,
    "Delay Irrigation": 0,
}

BASELINE_PER_FIELD = 10  # mm applied every cycle in a naive system


def simulate_water_usage(decisions: List[str]) -> List[int]:
    """Return water applied (mm) for each decision."""
    return [WATER_MAP.get(d, 0) for d in decisions]


def compute_savings(decisions: List[str]) -> dict:
    """
    Compare smart irrigation water usage against a naive baseline.

    Returns
    -------
    dict with keys: total_water, baseline, savings_mm, savings_pct
    """
    usage = simulate_water_usage(decisions)
    total_water = sum(usage)
    baseline = len(decisions) * BASELINE_PER_FIELD
    savings_mm = baseline - total_water
    savings_pct = (savings_mm / baseline) * 100 if baseline > 0 else 0.0

    return {
        "total_water": total_water,
        "baseline": baseline,
        "savings_mm": savings_mm,
        "savings_pct": round(savings_pct, 2),
    }


def print_savings_report(decisions: List[str]) -> None:
    """Pretty-print a water savings summary."""
    result = compute_savings(decisions)
    print("=" * 40)
    print("       WATER SAVINGS REPORT")
    print("=" * 40)
    print(f"  Smart system total : {result['total_water']} mm")
    print(f"  Naive baseline     : {result['baseline']} mm")
    print(f"  Water saved        : {result['savings_mm']} mm")
    print(f"  Savings (%)        : {result['savings_pct']}%")
    print("=" * 40)
