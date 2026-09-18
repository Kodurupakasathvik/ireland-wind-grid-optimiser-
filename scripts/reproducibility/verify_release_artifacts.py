"""Read-only integrity checks for the public research release.

This verifier requires only the Python standard library. It intentionally does
not rerun the research models: it verifies the published result artifacts,
source checksums, release figures, and the counterfactual security-credit
invariant.

Run from the repository root:
    python scripts/reproducibility/verify_release_artifacts.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data_manifest.csv"
COUNTERFACTUAL = ROOT / "data" / "processed" / "counterfactual_analysis.csv"
FIGURE_DIRECTORY = ROOT / "reports" / "figures"

EXPECTED_FIGURES = {
    "forecast-validation-nmae.svg",
    "counterfactual-security-screen.svg",
    "production-ac-security.svg",
    "s2-controlled-q-continuation.svg",
    "intervention-comparison.svg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_manifest(*, require_source_inputs: bool) -> tuple[int, int]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("Data manifest has no rows.")

    verified = 0
    unavailable = 0
    for row in rows:
        path = ROOT / row["relative_path"]
        if not path.is_file():
            unavailable += 1
            if require_source_inputs:
                raise AssertionError(f"Manifest input is missing: {path}")
            continue
        if sha256(path) != row["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {path}")
        verified += 1
    return verified, unavailable


def verify_counterfactual_gate() -> tuple[int, int]:
    with COUNTERFACTUAL.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["sensitivity_case"] == "base"
        ]
    if not rows:
        raise AssertionError("Counterfactual output has no base-case rows.")

    noncreditable = 0
    secure = 0
    for row in rows:
        is_secure = row["physically_secure"] == "True"
        converged = row["ac_converged"] == "True"
        credited_energy = float(row["security_screened_energy_mwh"])
        credited_value = float(row["indicative_value_eur"])
        credited_co2 = float(row["indicative_co2_tonnes"])
        if converged:
            voltage = float(row["minimum_voltage_pu"])
            line_loading = float(row["maximum_line_loading_percent"])
            if not (
                math.isfinite(voltage)
                and 0.0 < voltage <= 2.0
                and math.isfinite(line_loading)
                and 0.0 <= line_loading <= 10_000.0
            ):
                raise AssertionError(
                    "Converged AC row has an implausible core metric: "
                    f"{row['scenario']}"
                )
        elif row["security_assessment"] != "AC_NOT_CONVERGED_NOT_CREDITED":
            raise AssertionError(
                "Non-converged AC row has the wrong release assessment: "
                f"{row['scenario']}"
            )
        if is_secure:
            if credited_energy < 0 or credited_value < 0 or credited_co2 < 0:
                raise AssertionError(f"Negative credit in secure row: {row['scenario']}")
            secure += 1
        else:
            if any(value != 0.0 for value in (credited_energy, credited_value, credited_co2)):
                raise AssertionError(
                    "Insecure AC result received a counterfactual credit: "
                    f"{row['scenario']}"
                )
            noncreditable += 1
    return secure, noncreditable


def verify_figures() -> int:
    actual = {path.name for path in FIGURE_DIRECTORY.glob("*.svg")}
    if actual != EXPECTED_FIGURES:
        raise AssertionError(
            "Release figure set differs from the documented set: "
            f"expected {sorted(EXPECTED_FIGURES)}, found {sorted(actual)}"
        )
    for name in sorted(actual):
        ElementTree.parse(FIGURE_DIRECTORY / name)
    return len(actual)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify published release artifacts and available source inputs."
    )
    parser.add_argument(
        "--require-source-inputs",
        action="store_true",
        help="Fail when a source input listed in data_manifest.csv is absent.",
    )
    args = parser.parse_args()

    manifest_count, unavailable_source_count = verify_manifest(
        require_source_inputs=args.require_source_inputs
    )
    secure_count, noncreditable_count = verify_counterfactual_gate()
    figure_count = verify_figures()
    source_summary = f"{manifest_count} manifest hashes"
    if unavailable_source_count:
        source_summary += (
            f", {unavailable_source_count} listed source inputs unavailable"
        )
    print(
        "Release verification passed: "
        f"{source_summary}, "
        f"{secure_count} AC-secure and {noncreditable_count} non-creditable base rows, "
        f"{figure_count} SVG figures."
    )


if __name__ == "__main__":
    main()
