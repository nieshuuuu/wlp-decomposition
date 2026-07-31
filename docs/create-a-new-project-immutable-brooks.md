# Water–Lipid–Protein Material Decomposition — Standalone Physics Notebook

## Context

Build a **new, self-contained project** whose single deliverable is one standalone Pluto notebook doing
water/lipid/protein (WLP) volumetric-fraction decomposition from dual-energy CT — **pure physics and math,
nothing clinical**. Pipeline: simulate VMI at 40 keV and 70 keV of WLP-mixture cylinder rods in a
QRM-thorax phantom (BasisSimulator.jl), then recover per-voxel and per-region volume fractions
`(f_w, f_l, f_p)` by an explicit Bayesian MAP (Normal water · Normal lipid · Gamma protein) + coupled-TV,
and validate against ground truth.

**New, not an addition to `wl-noise-aware-mmd`:** that repo is mature but clinical-framed and split across
~30 `src/` modules. The user wants a clean, physics-only, *standalone* notebook that contains everything —
no local module scripts. So we **copy** the proven math (solver, noise, TV, metrics, adipose prior) into
notebook cells rather than `include` them, and **copy** the phantom masks + adipose CSV into the project's
`data/`. `BasisSimulator` stays an ordinary package dependency (not a "module script").

**Success criteria (user):**
- `f_water` recovered-vs-true **CCC > 0.9** (report CCC/slope/RMSE/R² for all three fractions).
- Detectability, **both** senses: (1) ROI-mean accuracy — recovered composition mapped back to HU within
  **5 HU** of truth, with ROI error bars (SEM); (2) minimum detectable contrast — two ROIs whose true HU
  differ by ≥5 HU are statistically separable (error bars don't overlap).
- Every pairwise relationship written as an equation **and** plotted; delivered maps vs true maps; accuracy table.

**Decisions locked with the user:**
- Phantom = **reproducible QRM-thorax** (reuse `wl-noise-aware-mmd`'s own masks, not example 07's proprietary lab `.raw`).
- **Minimal comments/markdown** in the notebook.
- Calibration scale = **Moderate** (~100–200 distinct calibration compositions across ~5–8 sims + noise replicates).
- Solver = **both** per-ROI/cluster-pooled *and* per-voxel, **plus** an unsupervised-clustering path.
- **Update BasisSimulator to latest v0.8.0** — `/Users/shunie/Developer/BasisSimulator.jl` is already on `main`; `git fetch origin --tags && git pull --ff-only` → v0.8.0. API is **signature-stable** v0.2.1→v0.8.0 (only `SimOptions` gained additive kwargs), so no call sites change. Resolve `XrayAttenuation` to 0.3.x (has `basis_lipid`/`basis_collagen`). **Do NOT use `-phantomloading.jl`** (divergent branch: no Cong/VMI/`resolve_source_spectrum_full`, XA 0.2).
- Recon **512×512×8**; there is **no library slice-completeness helper**, so compute `z_usable = c·(1−R_body/STI)` in-notebook (as nb07 does) and keep only slices inside the fully-sampled band; discard cone-truncated first/last slices.
- Noise model is **nonlinear (convex, ~quadratic)** in HU, not linear.
- **Endpoints theory-based** (below): lipid/protein HU from theory (no pure-material rods exist in real life), only water is measured.

---

## Project layout

```
/Users/shunie/Developer/wlp-decomposition/
├── Project.toml                # deps: BasisSimulator (dev, latest), CairoMakie, Statistics,
│                               #        LinearAlgebra, Random, Unitful, DelimitedFiles, Distributions,
│                               #        Clustering (k-means), TOML
├── data/
│   ├── qrm_thorax_wlplat_1850x1350_uint8.raw + .toml   # copied 21-rod mask (circular/star + calibration)
│   ├── adipose_composition_distribution.csv            # copied from ~/Developer/adipose_composition
│   └── adipose_verified_compositions.csv               # cachectic point (lipid 4.24/water 83.86/protein 12.76)
├── assets/                     # saved PNGs (safe_save, ≤1920 px/side)
└── wlp_decomposition.jl        # THE standalone Pluto notebook — everything inline
```

`Pkg.develop(path="/Users/shunie/Developer/BasisSimulator.jl")` after updating it to latest.

---

## The math (step-by-step — the notebook's spine)

**Unknowns/voxel:** `f_w,f_l,f_p ≥ 0`, `f_w+f_l+f_p=1`. Measurements `m=(HU_40,HU_70)`. Endpoints
`p_w,p_l,p_p` at each energy (calibrated).

**Step 1 — three explicit equations (user's form):**
```
f_w·μ_{w,E} + f_l·μ_{l,E} + f_p·μ_{p,E} + ε_E = m_E,  E∈{40,70};   f_w+f_l+f_p = 1
```

**Step 2 — reconcile the "σ per material per energy" (physics-first).** A voxel has one noise per energy,
not three. The user's `σ_{i,E}` are the ordinates of the single heteroscedastic noise curve `σ_E(HU)` read
off at each pure endpoint. If `σ_E` were affine they'd be a fraction-weighted average; since it is
**convex/quadratic** (below), the mixture noise is `σ_E(HU_mix)` with `HU_mix = Σ_i f_i·μ_{i,E}`. Either
way the three σ's collapse into one effective `ε_E` per energy with variance `σ_E(HU_mix)²`. (Optional
errors-in-variables endpoint term `Σ_i f_i²·Cov(σ_{i,E},σ_{i,E'})` is second-order; off by default.)

**Step 3 — water-reference → square 2×2.** Substitute `f_w = 1 − f_l − f_p`:
```
m − p_w = G·θ + ε,  θ=(f_l,f_p),  G=[p_l−p_w | p_p−p_w],  ε~N(0,Σ)
Σ = [σ_40²  ρσ_40σ_70 ; ρσ_40σ_70  σ_70²]
```
Exactly determined (per-voxel GLS=OLS=barycentric); noiseless locus = triangle `(p_w,p_l,p_p)`. Soft
tissue makes it a sliver (cond(G)≈5 at 40/70) → `f_p` fragile → prior + spatial pooling supply the missing
information (Alvarez–Macovski: CT attenuation is intrinsically 2-D).

**Step 4 — the −ln posterior (user's loss).** With `W=GᵀΣ⁻¹G`, `θ̂=W⁻¹GᵀΣ⁻¹(m−p_w)`:
```
L = ½(θ−θ̂)ᵀW(θ−θ̂)                          # both energies, one 2×2
  + ½(f_w−μ_w)²/s_w² + ½(f_l−μ_l)²/s_l²        # Normal(water)·Normal(lipid)
  − (α−1)·ln f_p + f_p/θ_p,  f_w=1−f_l−f_p     # Gamma(protein) = −ln Γ(f_p;α,θ_p)
```
**Gamma vs the user's Poisson:** `f_p` continuous, ≥0, low, right-skewed → Gamma (moment-matched:
`α=μ_p²/σ_p²`, `θ_p=σ_p²/μ_p`). The user's linearized-Poisson `β(λ−f_p ln λ)` is exactly the Gamma rate
pull `f_p/θ_p` (`1/θ_p ↔ −β ln λ`); Gamma adds the convex barrier `−(α−1)ln f_p` enforcing `f_p>0` +
strict convexity (unique MAP). Both offered; Gamma is the rigorous default.

**Step 5 — MAP solve (Newton), done PER CLUSTER and PER VOXEL (see Solver).**
```
g_l = [W(θ−θ̂)]₁ − (f_w−μ_w)/s_w² + (f_l−μ_l)/s_l²
g_p = [W(θ−θ̂)]₂ − (f_w−μ_w)/s_w² − (α−1)/f_p + 1/θ_p
H   = W + [1/s_w²+1/s_l²  1/s_w² ; 1/s_w²  1/s_w²+(α−1)/f_p²]
```
`θ ← θ − t·H⁻¹g` with `f_p,f_w>0` backtracking; strictly convex ⇒ unique optimum. (This is the **global
optimum of each voxel/cluster's own convex problem — NOT a whole-image joint solve.** Clarifies the user's
"global" concern.) A closed-form `(f_w,f_l)` variant with the full −0.97-correlated Gaussian and linear
protein term reproduces the user's literal written model without Newton — included as an alternative.

**Step 6 — spatial coupled Huber-TV (boundary detector, per-voxel map only).**
```
f_i ← (W_i+λΣ_n c_n M)⁻¹(W_i y_i+λΣ_n c_n M f_n),  c_n=1/max(‖f_n−f_i‖_M,ε);  λ=0.06, iters=80, ε=0.04
```
simplex-project each sweep. Controls `f_p` variance *spatially* (√N) instead of over-tightening the prior.

**n-choose-3 note:** no independent 3-way fit — the single 3-material relation is closure `f_w+f_l+f_p=1`
(the 2-simplex). Three pairwise relations + closure are the complete set.

**Guardrail (surface prominently):** any per-voxel `f_p` prior tight enough to denoise pins the `f_p` slope
toward the prior mean; a full-covariance *location* prior can collapse `f_w/f_l` slope. Keep priors
**broad** (`s_wl≈0.15`, Gamma shape≈1.2) and do variance control via pooling/TV. Expose
`PRIOR_MODE ∈ {:free,:broad,:gamma,:fwl_closed}`; `:free`→pooled/TV gives f_w slope ≈1.0 (meets CCC>0.9),
tight priors shown as the counter-example.

---

## Endpoints — theory-based (user: only water is realistically calibratable)

Real life can't supply pure-lipid or pure-collagen rods, so the lipid/protein endpoints come from **theory**,
and only water is measured:
- `p_l(E), p_p(E)` = **theoretical HU** from XA: `p_i(E) = 1000·(μ_i(E) − μ_w(E))/μ_w(E)`, with `μ_i` from
  `compute_μ_at_energy(basis_lipid | protein_material, E)`, `E∈{40,70}`. (Physics, no calibration rod.)
- `p_w(E)` = **measured** from a water anchor region (label 6) — the one endpoint you can realistically
  calibrate; expect ≈0 with a small cupping offset.
- **Recon↔theory beam-hardening reconciliation (β-debias).** Recon soft-tissue HU carries a small BH offset
  vs theory (this is why measured cond(G) ≈ 17–25 » theoretical ≈ 5). Fit a per-energy affine bias `β_E`
  from the **known-composition** calibration rods: each rod's theoretical mixture HU is `Σ_i f_i^true·p_i^theory`,
  so `measured_HU ≈ β_E(theory_HU)` is a 1-D fit per energy. Apply `β_E⁻¹` to test HU before decode. This
  uses only water + known-composition mixtures — **never a pure lipid/protein rod**. Report cond(G) from the
  theoretical endpoints and the residual after β-debias.

## Composition generation — explicit recipe (KDE from real adipose + seed)

Both calibration and validation `(f_w,f_l,f_p)` are drawn by **KDE sampling of the real 59-point Woodard 1986
Fig.1 adipose dataset** + a random seed (following `wlp_adipose_sampler.jl`):
```
draw_wlp(seed, n):
  rows = adipose CSV (mass%); keep state∈{healthy,obese,reduced,comparison,unspecified}, lipid_pct ≥ 50
  L  = lipid_pct of kept rows            (~56 values, the rich axis)
  (sl, sp) = (lipid_pct, protein_pct) of the 'protein' rows   (~19, the sparse split)
  rng = MersenneTwister(seed)
  bwL = std(L) · length(L)^(−1/5)                       # Scott 1-D KDE bandwidth
  z   = logit( sp ./ (100 .− sl) );  μz,σz = mean,std(z) # logit protein-share of the non-lipid remainder
  repeat n times:
    l  = clamp( L[rand(rng)] + bwL·randn(rng), 50, 99 )  # lipid mass% — smoothed-bootstrap KDE draw
    qq = logistic( μz + σz·randn(rng) )                  # protein share ∈(0,1) ⇒ f_p > 0
    p  = qq·(100 − l);  w = 100 − l − p                  # close in mass%
    (f_w,f_l,f_p) = mass_to_vol(w, l, p; ρ = 1.00/0.92/1.35)   # mass% → volume fraction
```
- **Calibration:** `seed_cal` (e.g. 20260708), `n = 21 × N_calib_sims` (Moderate → ~5–8 sims → ~105–168
  comps). Paint the 21 `wlplat` rods each sim; the f_w↔f_l (−0.96) and f_l↔f_p correlations emerge from
  closure (physically faithful).
- **Validation:** `seed_val` (disjoint, e.g. 91260708), `n ≥ 100` across the ~5–10 test sims (21-rod
  circular + 16-sector), **plus** the **cachectic** stress point (`verified_compositions.csv`: mass
  4.24/83.86/12.76 → volume) and a few water-rich / muscle-like comps to widen the `f_w` range for CCC.
- Seeds fixed and recorded → reproducible; calibration and validation composition sets are **disjoint by
  construction** (different seed streams). The pure vertices are used only for the *theoretical* endpoint
  computation, never as physical rods.

## Noise model (CORRECTED: nonlinear / convex quadratic)

`σ_E(HU)` is **not** linear — it is convex (roughly quadratic), with a minimum near water/soft-tissue HU and
rising toward both lipid (very negative) and dense HU. Model per energy:
```
σ_E(HU) = a_E·HU² + b_E·HU + c_E     (convex quadratic; require a_E ≥ 0)
```
Fit by WLS across the calibration rods' measured σ-ladder (many HU levels). Also keep the **measured
σ-ladder as SSoT** and evaluate Σ by clamped interpolation of it (`sigma_meas`-style), overlaying the
quadratic fit as the model curve. Correlation `ρ_{40,70}` from centered per-voxel residuals. Feed both into
`sigma_cov`'s `curve` slot so Σ tracks the true convex σ(HU). Plot the ladder + quadratic fit per energy,
and `σ_40` vs `σ_70` (correlation ρ̄).

---

## Solver structure (BOTH paths, per user)

1. **Per-region (cluster) pooled MAP — headline accuracy + detectability.** Each rod/sector = one cluster.
   Pool its ROI-core voxels → mean `(HU_40,HU_70)` with `Σ/N` → run the Step-5 MAP once per cluster → one
   `(f_w,f_l,f_p)` + tight SEM. √N pooling is why this is *more accurate* than per-voxel, and the SEM
   supplies the 5-HU detectability error bars. Two cluster sources:
   - **Known ROI** (rod/sector masks from the phantom labels) — for validation vs ground truth.
   - **Unsupervised** (k-means/superpixels on the `(HU_40,HU_70)` feature per voxel; `Clustering.jl`) — the
     general method with no known geometry; validation error bars come from cluster membership.
2. **Per-voxel MAP + coupled-TV — delivered fraction maps.** Step-5 MAP per voxel (independent), then
   Step-6 TV → boundary-preserving `(f_w,f_l,f_p)` maps + σ maps for the delivered-vs-true images.

Report both; compare per-voxel vs pooled RMSE/CCC to show the √N gain.

---

## Forward simulation (all inline; UPDATE BasisSimulator to latest first, re-verify API)

**Step 0:** `git fetch origin --tags && git pull --ff-only` in `/Users/shunie/Developer/BasisSimulator.jl`
(→ v0.8.0), then `Pkg.develop` it and ensure `XrayAttenuation` resolves to 0.3.x. API is signature-stable
v0.2.1→v0.8.0 (all functions below verified against v0.8.0; only `SimOptions` gained additive kwargs
`projector`/`use_pcct_scatter*` — our calls are unaffected). Note verified gotchas: `voxel_size_cm` is a
**3-tuple**; **collimation is a `CTProtocol` arg, not `Scanner`**; scanner is baked into the workspace (not a
`simulate!` arg).

**Materials.** Water `BS.XA.Materials.basis_water` (ρ1.00), lipid `BS.XA.Materials.basis_lipid` (ρ0.92),
protein inline from Woodard&White 1986 `Dict(1=>0.066,6=>0.534,7=>0.170,8=>0.220,16=>0.010)` ρ1.35.
`ZA_ratio`/`I` may be 0.0 (attenuation uses only density+composition — verified).

**Volume-fraction → effective Material** (exact): `ρ_eff=Σf_v ρ_v`; `m_v=f_v ρ_v/ρ_eff`;
`comp[Z]=Σ_v m_v comp_v[Z]`.

**Phantom (reuse the reproducible QRM-thorax masks).**
- **Circular/star + calibration:** copy `qrm_thorax_wlplat_1850x1350_uint8.raw`+`.toml` into `data/`; inline a
  ~10-line `load_phantom_mask` (reads shape/voxel from TOML). 21 rods Ø16 mm, labels 8..28 (centre + inner
  ring r=26 mm ×8 + outer ring r=44 mm ×12), water anchor label 6, lipid anchor label 7, on the stadium
  body (300×200 mm) + fat ring + spine + ribs + muscle fill. Re-paint rod labels → materials each sim.
- **Sector geometry:** on the same body, partition the heart disk (Ø110 mm at (185,115)) inline into **16
  solid sectors = 8 angular × 2 radial halves**, paint each with a test comp. (Simple inline geometry; the
  wlplat body is reused, not rebuilt.)

**Acquisition (reusable inline fn, per example 07 / latest API):** dual-kVp EICT (80/140 kVp) →
`simulate!` → Cong water/iodine basis (spectrum via `resolve_source_spectrum_full` with the *same*
sim_opts/geom) → FDK FBP at **512×512×8**, fov ~22 cm → z-median → VMI at 40 & 70 keV via
`synth_vmi_2basis(vol_water, c_iodine_mg_per_mL; energy_keV=E)`. GPU auto-detect (Metal). **Slice-completeness:**
no library helper exists — compute `z_usable = c·(1−R_body/STI)` in-notebook and set `ReconOptions.z_cm` so all
8 slices fall in the fully-sampled band, or reconstruct taller and drop cone-truncated edge slices (validate
by edge-vs-centre ROI σ/mean).

**Physics correctness (flag inline):** water+iodine basis faithfully reproduces lipid/protein HU (their
μ(E) is in the 2-D PE/Compton span; no K-edge in band) — their fitted iodine density is **negative; never
clamp `c_iodine ≥ 0`**. 40 keV amplifies signal+noise vs 70 keV — report separately. Use eroded ROI cores
for all HU/σ statistics.

**Scan budget.** Calibration (Moderate): ~5–8 wlplat sims, each 21 rods painted with `draw_wlp(seed_cal)`
adipose comps (→ ~105–168 distinct known comps) + the measured water anchor (label 6) for `p_w`; lipid/protein
endpoints are theoretical (no pure rods). ~40–60 noise-replicate scans (seed-varied, `z_median=0`) for the
convex σ(HU)+ρ fit. Test set: ~5–10 packed sims (21-rod circular + 16-sector, `draw_wlp(seed_val)` comps +
cachectic, unique seed) → 100+ ROIs. Minutes–tens-of-minutes on Metal.

---

## Notebook cell order (lean; minimal comments/markdown)

**Forward:** imports+GPU backend · protein material + `wlp_material` · calibration compositions (18 adipose
via inline Woodard sampler + 3 anchors) · `load_phantom_mask` + 16-sector builder + label→material repaint ·
scanner/protocol/options (512×512×8) · `run_acquisition` (sim→Cong→FBP→z-median→VMI + slice-completeness) ·
calibration acquisition + display · ROI-core helper + `theoretical_hu` ground truth.

**Calibration & priors:** theoretical `p_l,p_p` + measured `p_w` + `cond(G)` · β-debias fit from
known-composition rods (recon→theory) · convex-quadratic σ_E(HU) + ρ fit (+ measured ladder) → `sigma_cov`
closure · adipose prior: inline KDE sampler (mass→vol ρ 1.0/0.92/1.35) + Normal(f_w)·Normal(f_l)·Gamma(f_p)
fits (skew check → 2–3-Gaussian mixture if |skew|≳0.5).

**Inverse:** inline solver — `sigma_cov`, GLS/`map_flp`, MAP Newton (`L,∇L,H`), `fwl` closed form,
`tv_denoise_coupled`, k-means cluster path, `PRIOR_MODE` switch · apply to test set → pooled per-cluster
`(f,SEM)` (known-ROI + unsupervised) + per-voxel MAP+TV maps.

**Plots (each relationship named):** calibration `f_w` vs HU_40/HU_70 (colored by f_p); barycentric triangle
HU_40 vs HU_70 with W/L/P vertices (40/70 vs 70/150); fraction pairs `f_w–f_l`, `f_w–f_p`, `f_l–f_p` +
simplex/ternary; σ vs HU per energy (convex fit + ladder); `σ_40` vs `σ_70` (ρ̄); prior marginals
Normal/Normal/Gamma; joint `(f_l,f_p)` covariance ellipse; delivered map vs true map (f_w/f_l/f_p rows:
true/recovered/signed-error, `:jet` (0,1) fractions, `:balance` error), both geometries.

**Validation:** recovered-vs-true scatter per fraction, 1:1 line, **CCC/slope/RMSE/R²**, ROI SEM error bars;
detectability panel — (1) |recovered−true|→HU ≤ 5 HU with bars, (2) ≥5-HU ROI separability; accuracy table
across `PRIOR_MODE` and pooled-vs-voxel.

---

## Reuse map (copy into cells — do not `include`)

| Notebook cell | Copy from |
|---|---|
| `sigma_cov` (+ `curve`/`sigma_meas`), GLS `gls_fw` | `wl-noise-aware-mmd/src/wl_decompose.jl` |
| `map_flp`, simplex projection, `bary_decompose` | `wl-noise-aware-mmd/src/wlp_decompose.jl` |
| Bayesian MAP (`L/∇L/H`, Gamma-Newton, `fwl`) | `wl-noise-aware-mmd/src/wlp_bayes.jl` |
| `tv_denoise_coupled` | `wl-noise-aware-mmd/src/wl_denoise.jl` |
| adipose sampler + Normal/Normal/Gamma prior | `wl-noise-aware-mmd/src/wlp_adipose_sampler.jl`, `wlp_prior_cov.jl` |
| CCC/slope/RMSE/SEM, `safe_save`, `:jet` | `wl-noise-aware-mmd/notebooks/qrm_wlp_noise_mmd.jl:1044-1057, 74-77` |
| forward pipeline (scanner/Cong/FBP/VMI, GPU helper) | `BasisSimulator.jl/docs/notebooks/07_qrm_thorax_pure_material_vmi.jl` (latest) |
| σ-ladder measurement + β-debias pattern (`fit_noise_wlp`) | `wl-noise-aware-mmd/src/wlp_qrm.jl` |

Copy masks: `data/qrm_thorax/qrm_thorax_wlplat_1850x1350_uint8.{raw,toml}`. Copy adipose CSVs:
`~/Developer/adipose_composition/composition_distribution.csv` + `sources/verified_compositions.csv`.

---

## Verification (end-to-end, not "should work")

1. **Forward sanity:** 21 calibration rods' ROI-mean `hu[40]/[70]` match `theoretical_hu(wlp_material(...),E)`
   within the β-debias residual; measured water anchor `p_w ≈ 0`; after β-debias the known-composition rods
   sit on the theoretical mixture line; slice-completeness (`z_usable`) drops only truncated edge slices.
2. **Solver unit checks (inline `demo()` with asserts):** noiseless voxel → MAP recovers `(f_w,f_l,f_p)`
   exactly; Newton converges to the analytic quadratic optimum when priors off; strict convexity (`α>1`).
3. **Headline:** full test set → `f_water` CCC — **gate > 0.9**; report CCC/slope/RMSE/R² for all three
   (expect `f_p` weakest — document honestly); pooled-vs-voxel comparison shows the √N gain.
4. **Detectability:** ROI-mean |recovered−true|→HU ≤ 5 HU with SEM bars (sense 1) and ≥5-HU ROIs separable
   (sense 2).
5. **Run it:** execute end-to-end on Metal (`pluto-collab run wlp_decomposition.jl --stale` or headless
   Julia); confirm plots render, accuracy table populates. Report actual numbers.

**Open risks:** (a) `f_p` CCC is physics-limited at 40/70 keV — gate is on `f_water`, `f_p` reported
honestly; (b) runtime hardware-bound (GPU assumed); (c) 512×512×8 cone-beam z-coverage may leave <8 usable
slices — keep only complete ones; (d) latest-BasisSimulator API deltas patched into forward cells before coding.
