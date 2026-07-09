# wlp-decomposition

Water–lipid–protein (WLP) volumetric material decomposition from dual-energy CT — **pure physics and
math**. A single standalone Pluto notebook simulates virtual monoenergetic images (VMI) at 40 keV and
70 keV of WLP-mixture rods in a QRM-thorax phantom (via [BasisSimulator.jl](https://github.com/MolloiLab/BasisSimulator.jl)),
then recovers per-voxel and per-region volume fractions `(f_w, f_l, f_p)` by an explicit Bayesian MAP
(Normal water · Normal lipid · Gamma protein) + coupled total-variation.

## Method

1. **Forward** — mix water/lipid/protein by volume fraction into one effective attenuating material per
   rod; dual-kVp (80/140) simulation → water/iodine basis → FBP → VMI at 40 & 70 keV.
2. **Endpoints** — lipid/protein HU from theory (`μ` via XrayAttenuation); only water is measured; a
   β-debias reconciles recon-vs-theory beam hardening.
3. **Noise** — convex `σ_E(HU) = a·HU² + b·HU + c` per energy + inter-energy correlation ρ, from
   replicate scans.
4. **Inverse** — water-referenced `y = Gθ + ε`; `−ln` posterior = GLS data term + Normal/Normal/Gamma
   prior; MAP solved per region (√N-pooled, with error bars) and per voxel (+ coupled-TV maps).
5. **Validation** — recovered-vs-true CCC/slope/RMSE/R²; detectability within 5 HU (ROI accuracy +
   contrast separability).

## Run

```julia
julia --project=.        # env pins BasisSimulator @ MolloiLab main, XrayAttenuation 0.3.x
```

Open `wlp_decomposition.jl` in Pluto. GPU (Metal/CUDA) is auto-detected; falls back to CPU.

## Data

- `data/qrm_thorax_wlplat_1850x1350_uint8.{raw,toml}` — 21-rod QRM-thorax phantom mask (provenance:
  `MolloiLab` PCATSim generator).
- `data/adipose_composition_distribution.csv`, `adipose_verified_compositions.csv` — Woodard & White 1986
  adipose composition, used only to fit the material prior. Derived from copyrighted figures — private use.
