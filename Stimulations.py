# MIMIR-X Scenario 2 — Python Simulation
# -----------------------------------------------------------
# This simulates a Mars trash-to-materials pipeline:
# 1) Sorting -> 2) Shredding -> 3) Mechanical Press (tiles/sheets)
# 4) Enzyme Depolymerization (PET/PA/PU) -> oligomer binders
# 5) Optional binder mixing to produce sealants/foams
#
# Inputs:
# - Mixed inorganic waste stream composition (kg) per batch
# - Process parameters (yields, energy, cycle times)
#
# Outputs:
# - Mass flow by product (tiles/sheets, binders, rejects)
# - Energy consumption, cycle times, throughput
# - CSV export of batch-by-batch results and a config JSON
#
# Notes:
# - No seaborn; charts are matplotlib only
# - One chart per figure

import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# -----------------------------
# Config & Materials
# -----------------------------

@dataclass
class Material:
    name: str
    mech_ok: bool           # eligible for mechanical hot-press
    enzyme_ok: bool         # eligible for enzyme depolymerization
    bulk_density: float     # kg/L (used for volume handling estimates)
    base_contam: float      # baseline contamination fraction (0..1)

# Representative materials in Scenario 2
MATERIALS: Dict[str, Material] = {
    "EVA_foam": Material("EVA_foam", mech_ok=True,  enzyme_ok=False, bulk_density=0.09,  base_contam=0.05),
    "LDPE_film": Material("LDPE_film", mech_ok=True,  enzyme_ok=False, bulk_density=0.40,  base_contam=0.04),
    "PP_hard":   Material("PP_hard",   mech_ok=True,  enzyme_ok=False, bulk_density=0.90,  base_contam=0.03),
    "PET":       Material("PET",       mech_ok=False, enzyme_ok=True,  bulk_density=1.38,  base_contam=0.04),
    "PA_nylon":  Material("PA_nylon",  mech_ok=False, enzyme_ok=True,  bulk_density=1.14,  base_contam=0.04),
    "PU_foam":   Material("PU_foam",   mech_ok=False, enzyme_ok=True,  bulk_density=0.03,  base_contam=0.07),
    "Textiles":  Material("Textiles",  mech_ok=True,  enzyme_ok=False, bulk_density=0.20,  base_contam=0.06),
    "Aluminum":  Material("Aluminum",  mech_ok=False, enzyme_ok=False, bulk_density=2.70,  base_contam=0.01),
    "Composites":Material("Composites",mech_ok=True,  enzyme_ok=False, bulk_density=1.60,  base_contam=0.08),
    "Misc":      Material("Misc",      mech_ok=False, enzyme_ok=False, bulk_density=0.50,  base_contam=0.10),
}

DEFAULT_INPUT_FRACTIONS = {
    # Fractions should sum ~1.0; small slack is handled by normalization
    "EVA_foam": 0.18,
    "LDPE_film": 0.14,
    "PP_hard": 0.10,
    "PET": 0.16,
    "PA_nylon": 0.06,
    "PU_foam": 0.05,
    "Textiles": 0.10,
    "Aluminum": 0.05,
    "Composites": 0.10,
    "Misc": 0.06,
}

# -----------------------------
# Process Parameters
# -----------------------------

@dataclass
class ProcessParams:
    # Sorting
    sorting_loss: float = 0.01            # fraction lost during sorting (dust/fines)
    sorting_energy_kwh_per_kg: float = 0.03

    # Shredding
    shred_energy_kwh_per_kg: float = 0.05

    # Mechanical press (tiles/sheets)
    mech_press_yield: float = 0.85        # fraction retained as product from mech_ok mass
    mech_press_energy_kwh_per_kg: float = 0.12
    mech_cycle_time_min: float = 20.0

    # Enzyme reactor (PET/PA/PU)
    enzyme_max_conversion: float = 0.80   # asymptotic conversion at long residence times
    enzyme_rate_constant_h: float = 0.12  # pseudo-first-order rate constant
    enzyme_residence_time_h: float = 6.0
    enzyme_energy_kwh_per_kg: float = 0.20

    # Binder blending
    binder_blend_loss: float = 0.05
    binder_energy_kwh_per_kg: float = 0.04

    # Contamination multiplier (adds stochastic losses)
    contam_sigma: float = 0.02

    # Batch handling
    batch_size_kg: float = 70.0           # mass per batch
    batches: int = 180                    # total batches (e.g., 3 years ~ 12,600 kg)

params = ProcessParams()

# -----------------------------
# Core Simulation
# -----------------------------

@dataclass
class BatchResult:
    batch_id: int
    input_mass_kg: float
    mech_feed_kg: float
    enz_feed_kg: float
    metals_kg: float
    rejects_kg: float

    tiles_sheets_kg: float
    binders_kg: float
    sealants_kg: float

    energy_kwh: float
    cycle_time_min: float

def normalize_fractions(fracs: Dict[str, float]) -> Dict[str, float]:
    s = sum(fracs.values())
    if s == 0:
        raise ValueError("All input fractions are zero.")
    return {k: v / s for k, v in fracs.items()}

def enzyme_conversion_fraction(p: ProcessParams) -> float:
    # Simple first-order approach to plateau
    return p.enzyme_max_conversion * (1 - math.exp(-p.enzyme_rate_constant_h * p.enzyme_residence_time_h))

def run_batch(batch_id: int, p: ProcessParams, input_mix_kg: Dict[str, float]) -> BatchResult:
    # Sorting
    sorting_losses = sum(input_mix_kg.values()) * p.sorting_loss
    sorting_energy = sum(input_mix_kg.values()) * p.sorting_energy_kwh_per_kg

    # Categorize streams
    mech_feed = 0.0
    enz_feed = 0.0
    metals = 0.0
    rejects = 0.0

    for name, mass in input_mix_kg.items():
        m = MATERIALS[name]
        contam = max(0.0, np.random.normal(m.base_contam, p.contam_sigma))
        clean_mass = mass * (1 - contam)  # contaminated portion becomes reject
        rejected = mass - clean_mass

        if m.mech_ok:
            mech_feed += clean_mass
        elif m.enzyme_ok:
            enz_feed += clean_mass
        elif name == "Aluminum":
            metals += clean_mass
        else:
            rejects += clean_mass

        rejects += rejected  # add contamination to rejects

    # Shredding
    shred_energy = (mech_feed + enz_feed) * p.shred_energy_kwh_per_kg

    # Mechanical press
    tiles_sheets = mech_feed * p.mech_press_yield
    mech_press_losses = mech_feed - tiles_sheets
    mech_energy = mech_feed * p.mech_press_energy_kwh_per_kg
    mech_time = 0 if mech_feed == 0 else p.mech_cycle_time_min

    # Enzyme reactor
    conv = enzyme_conversion_fraction(p)   # fraction to oligomers
    oligomers = enz_feed * conv
    enzyme_losses = enz_feed - oligomers
    enzyme_energy = enz_feed * p.enzyme_energy_kwh_per_kg

    # Binder blending (some oligomers become sealants/foam; the rest remain "binders")
    blend_loss = oligomers * p.binder_blend_loss
    binders_final = oligomers - blend_loss
    # Split: 60% stays as binders, 40% turned into sealants/foam
    sealants = binders_final * 0.40
    binders = binders_final - sealants
    binder_energy = oligomers * p.binder_energy_kwh_per_kg

    total_energy = sorting_energy + shred_energy + mech_energy + enzyme_energy + binder_energy
    total_time = mech_time  # simplification: enzyme is continuous and not rate-limiting for UI demo

    return BatchResult(
        batch_id=batch_id,
        input_mass_kg=sum(input_mix_kg.values()),
        mech_feed_kg=mech_feed,
        enz_feed_kg=enz_feed,
        metals_kg=metals,
        rejects_kg=rejects + mech_press_losses + enzyme_losses,  # all non-product
        tiles_sheets_kg=tiles_sheets,
        binders_kg=binders,
        sealants_kg=sealants,
        energy_kwh=total_energy,
        cycle_time_min=total_time,
    )

def build_input_mix(batch_size: float, fractions: Dict[str, float]) -> Dict[str, float]:
    f = normalize_fractions(fractions)
    return {k: batch_size * v for k, v in f.items()}

def run_simulation(p: ProcessParams, fractions: Dict[str, float]):
    all_rows: List[Dict] = []
    for b in range(1, p.batches + 1):
        mix = build_input_mix(p.batch_size_kg, fractions)
        res = run_batch(b, p, mix)
        all_rows.append(asdict(res))

    df = pd.DataFrame(all_rows)

    totals = {
        "total_input_kg": df["input_mass_kg"].sum(),
        "total_tiles_sheets_kg": df["tiles_sheets_kg"].sum(),
        "total_binders_kg": df["binders_kg"].sum(),
        "total_sealants_kg": df["sealants_kg"].sum(),
        "total_metals_kg": df["metals_kg"].sum(),
        "total_rejects_kg": df["rejects_kg"].sum(),
        "total_energy_kwh": df["energy_kwh"].sum(),
        "avg_cycle_time_min": df["cycle_time_min"].replace(0, np.nan).mean(skipna=True),
    }
    return df, totals

if __name__ == "__main__":
    fractions = DEFAULT_INPUT_FRACTIONS.copy()
    df, totals = run_simulation(params, fractions)

    # Save outputs
    results_csv = "mimirx_scenario2_batches.csv"
    summary_json = "mimirx_scenario2_summary.json"
    config_json = "mimirx_scenario2_config.json"

    df.to_csv(results_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(totals, f, indent=2)
    with open(config_json, "w") as f:
        json.dump({
            "materials": {k: asdict(v) for k, v in MATERIALS.items()},
            "fractions": fractions,
            "params": asdict(params),
        }, f, indent=2)

    # Charts (one per figure, no seaborn, no explicit colors)
    import matplotlib.pyplot as plt
    import numpy as np

    # 1) Product mass totals
    plt.figure()
    labels = ["Tiles/Sheets", "Binders", "Sealants", "Metals", "Rejects"]
    values = [
        totals["total_tiles_sheets_kg"],
        totals["total_binders_]()
