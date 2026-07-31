# wlp-decomposition

Water–lipid–protein (WLP) volumetric material decomposition from dual-energy CT — **pure physics and
math**. A single standalone Pluto notebook builds a QRM-thorax phantom in code, simulates virtual
monoenergetic images (VMI) at **70 and 150 keV** via
[BasisSimulator.jl](https://github.com/MolloiLab/BasisSimulator.jl) **(v0.8.0)**, then recovers
per-voxel and per-region volume fractions `(f_w, f_l, f_p)` from a calibration surface plus a
σ-weighted coupled Huber-TV.

The full derivation is [`wlp_decomposition_math.md`](wlp_decomposition_math.md); it is the
authoritative description of what the code does, and every number below comes from it.

## Result (129 held-out test ROIs)

Raw per-voxel decode, pooled over eroded cores:

| fraction | CCC | slope | RMSE |
|---|---|---|---|
| `f_w` | **0.996** | 1.009 | 0.023 |
| `f_l` | **0.997** | 1.014 | 0.019 |
| `f_p` | **0.998** | 0.995 | 0.005 |

The *delivered* estimator — the same decode after the Huber-TV and a single simplex projection —
scores CCC 0.994 / 0.995 / 0.997 and RMSE 0.028 / 0.024 / 0.007. It is slightly worse at the region
level by construction: any one-sided projection buys a feasible map with a little ROI accuracy.

**Detectability.** Healthy and diseased pericoronary fat differ by about 5 HU. Against this method's
own region-level error that is a margin of **1.5–2.6×**, depending on which way the composition
moves — worst in a mixed lipid/protein direction, best along pure protein. The noise floor is
0.24 HU, 14× below the total error, so what limits 5 HU detection is systematic accuracy, not photon
statistics. See §6.1 of the math document for the calculation.

## Method

1. **Forward** — mix water/lipid/protein by volume fraction into one effective attenuating material
   per insert; 80/140 kVp simulation with the **`:dd_fast` projector** → Cong water/iodine basis →
   FBP (3-slice, cone-usable z-band) → VMI at 70 and 150 keV.
2. **Phantom** — a stadium QRM-thorax generated in code (925 × 675 at 0.4 mm/px): fat ring, muscle,
   lungs, ribs, spine, and a 55 mm heart cavity holding three insert geometries — 13 hex-packed
   ø19.9 mm circular inserts, 16 held-out sectors, and a single-insert size series at r = 4, 6, 9,
   12 mm.
3. **Endpoints** — theoretical HU from NIST cross-sections, never from a measured rod. At 70/150 keV
   the recon matches theory to −2.3 / +0.5 HU with |max| < 5 HU.
4. **Noise** — convex `σ_E(HU)` per energy plus the inter-energy correlation ρ = 0.787, measured
   from paired within-ROI residuals.
5. **Inverse** — calibration surface `f = poly₂(HU₇₀, HU₁₅₀)` fit on 52 calibration cores, then a
   σ_f-weighted coupled Huber-TV (λ = 12.12 by golden-section on calibration only, ε = 0.04, 25
   sweeps) with the simplex projection applied **once** on the result.
6. **Validation** — recovered-vs-true CCC / slope / RMSE / R² on 129 ROIs disjoint from calibration,
   plus a conservation check on total lipid across the size series.

> **The recon must be quantitative before any decode is trustworthy.** Validate the simulator against
> pure-material `theoretical_hu` first. BasisSimulator v0.2.1's projector compressed non-water HU by
> ~50 HU, which caps `f_w` near 0.75 — a simulator artifact, not physics. Use v0.8.0 with
> `SimOptions(…; projector=:dd_fast)`.

## Scope — what this does not do

- **The Bayesian MAP is scaffolded but never executed.** The notebook builds the adipose prior
  object, but nothing consumes it: no Newton step, no posterior. Every number here comes from the
  empirical calibration surface cleaned by Huber-TV. Treat the estimator as calibration-based, not
  Bayesian.
- **The TV regulariser is a hand-set smoothness assumption**, not a learned or adipose-derived prior.
- **All accuracy claims are simulation-vs-simulation.** The truth is the composition painted into the
  phantom, recovered through a forward and inverse both written here. This validates the math and the
  recon's fidelity; it does not validate against physical phantoms or patients.
- **No scanner-specific number transfers.** Endpoints, noise lines and ρ are properties of this
  acquisition. The *recipe* transfers; the constants must be re-measured per system, as in the
  sibling [`calibration_comparison.md`](https://github.com/MolloiLab/wl-noise-aware-mmd).

## Run

```julia
julia --project=.        # dev-deps BasisSimulator v0.8.0, XrayAttenuation 0.3.x
```

Open `wlp_decomposition.jl` in Pluto. GPU (Metal/CUDA) is auto-detected and falls back to CPU. The
first run simulates and caches to `data/wlp_*_cache_70_150.jls`; delete those to re-simulate. Julia
**1.12 or newer** is required to deserialize the caches (they are serialization data-version 30).

## Data

The phantom is **generated in code** (no mask file) and there are **no input data files** — the
notebook is standalone. The Woodard & White 1986 adipose composition table (59 points, 7 studies),
used solely to draw physiological compositions, is inlined as the `ADIPOSE_CSV` constant. Derived
from copyrighted figures — private use.
