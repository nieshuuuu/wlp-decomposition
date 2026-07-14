# wlp-decomposition

Water–lipid–protein (WLP) volumetric material decomposition from dual-energy CT — **pure physics and
math**. A single standalone Pluto notebook generates a phantom (Ø280 mm water cylinder with 13 ø28 mm
rods), simulates virtual monoenergetic images (VMI) at 40 keV and 70 keV via
[BasisSimulator.jl](https://github.com/MolloiLab/BasisSimulator.jl) **(v0.8.0)**, then recovers per-voxel
and per-region volume fractions `(f_w, f_l, f_p)` by a calibration surface + adipose prior + coupled-TV.

## Result (97 test ROIs)

| fraction | CCC | slope |
|---|---|---|
| **f_water** | **0.99** | 1.02 |
| f_lipid | 1.00 | 1.01 |
| f_protein | 1.00 | 1.02 |

**100% of ROIs < 5 HU at 70 keV** (mean 0.8 HU). All three fractions ≫ CCC 0.9.

## Method

1. **Forward** — mix water/lipid/protein by volume fraction into one effective attenuating material per
   rod; dual-kVp (80/140) simulation with the **`:dd_fast` projector** → Cong water/iodine basis → FBP
   (3-slice, inside the cone-usable z-band) → VMI at 40 & 70 keV.
2. **Endpoints** — theoretical HU (`μ` via XrayAttenuation); on v0.8.0 the recon matches theory (pure
   lipid −205 vs −213, < 2%), so the mixture model is linear-additive.
3. **Noise** — convex `σ_E(HU) = a·HU² + b·HU + c` per energy + inter-energy correlation ρ, from the rod ROIs.
4. **Inverse** — calibration surface `f = poly₂(HU₄₀, HU₇₀)` fit from known mixtures; applied per region
   (√N-pooled, with SEM error bars) and per voxel. Adipose `𝒩(f_w)·𝒩(f_l)·Γ(f_p)` prior for Bayesian refinement.
5. **Validation** — recovered-vs-true CCC/slope/RMSE/R²; detectability within 5 HU; ROI-area curve (fig 9).

> **Note.** The recon must be quantitative — validate the simulator against pure-material `theoretical_hu`
> first. The old BasisSimulator v0.2.1 projector compressed non-water HU by ~50 HU, which would cap f_water
> near 0.75 (a simulator artifact, not physics). Use v0.8.0 with `SimOptions(…; projector=:dd_fast)`.

## Run

```julia
julia --project=.        # env dev-deps BasisSimulator v0.8.0 (~/Developer/BasisSimulator.jl), XrayAttenuation 0.3.x
```

Open `wlp_decomposition.jl` in Pluto. GPU (Metal/CUDA) is auto-detected; falls back to CPU. The first run
simulates (~6 min) and caches to `wlp_sim_cache.jls`; delete that file to re-simulate.

## Data

The phantom is **generated in code** (no mask file), and there are **no input data files** — the
notebook is standalone. The Woodard & White 1986 adipose composition table (59 points, 7 studies),
used solely to fit the material prior, is inlined as the `ADIPOSE_CSV` constant in the notebook's
composition-generation cell and plotted there as `assets/fig1_adipose_composition.png`. Derived from
copyrighted figures — private use.
