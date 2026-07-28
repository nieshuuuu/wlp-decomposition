# The math of `pcat_deconv_design.jl`

How the Oxford clinical FAI HU gradient becomes the phantom's tissue-domain water/lipid/protein
composition. Single source of truth for the algorithm; the script is the implementation.

---

## §0. What is, and is not, an image operation

There is **no image-domain processing in this algorithm.** It operates on a table of 20 numbers.

The only step that ever touches a reconstructed image is §2, where the blur width `σ` is measured
off the simulation's own radial HU profile. Everything after that is a 20-point curve fit and an
algebraic HU-to-fractions inversion. In particular there is no deconvolution filter, no Wiener
filter, no Richardson–Lucy, no unsharp mask, and nothing is applied to a voxel grid.

This matters because "deconvolution" invites the wrong mental model. The output is not a corrected
image. It is a **pre-distorted design specification**: the composition you put *into* the phantom so
that simulating it — which blurs once — returns the clinical curve.

---

## Known

| symbol | meaning | value |
|---|---|---|
| `d_i` | distance from the vessel wall, layer `i` | `1, 2, …, 20` mm (`N = 20`) |
| `y_i` | clinical HU at layer `i` | `-75.37 … -88.08` (healthy) |
| `s_i` | standard error of the **cohort mean** HU at layer `i` | `1.00 … 1.75` HU |
| `ρ_i` | protein-to-water ratio `f_p/f_w` from the clinical composition | `0.088 … 0.104` |
| `HU_w` | water endpoint | `0` by definition |
| `HU_l` | lipid endpoint at `E_eff = 70` keV | `-111.69` |
| `HU_p` | protein endpoint at `E_eff = 70` keV | `+270.58` |
| `σ` | in-plane radial blur width | `1.05` mm (fitted, §2) |

## Want to know

The tissue-domain volume fractions `(f_w, f_l, f_p)_i` for each of the 20 layers, in each of the
two groups (healthy, diseased) — the numbers written into
`pcat_ct/oxford_deconvolved_composition.csv` and used as the phantom's ground truth.

## Assumptions

- **A1.** The clinical radial profile is the tissue profile convolved once with a 1-D Gaussian in
  the radial coordinate. *(This is the load-bearing modelling assumption, and §11 shows it does
  almost no work at the fitted parameters.)*
- **A2.** The tissue profile is smooth and monotone — a diffusion-like lipid gradient away from the
  vessel — so three parameters suffice.
- **A3.** No vessel-wall term. The Oxford measurement is gated to `[-190, -30]` HU, so voxels
  contaminated by the `+43` HU wall were already excluded upstream; including a wall in the blur
  overshoots the clinical curve by about `18` HU at `r = 1–2` mm.
- **A4.** `μ` is volume-additive and water is the HU zero, so HU is **exactly** barycentric in the
  volume fractions. This is not an approximation.
- **A5.** The protein-to-water ratio `ρ_i` is the same in the tissue domain as in the clinical
  domain. This is the closure that replaces re-running the Woodard-1986 adipose prior; it moves
  along the prior's own isoline rather than inventing a new one.
- **A6.** `s_i` is the standard error of the **mean over a cohort of >100 patients**, not a
  single-patient standard deviation. Weighting by `1/s_i²` therefore fits the *cohort mean curve*,
  which is the right target for a phantom of "the average patient" — but the per-patient standard
  deviation is `s_i · sqrt(N_patients)`, of order `10–18` HU. Every correction computed below is an
  order of magnitude smaller than the population it represents.

---

## §1. The blur operator

Discretise the radial axis on `r_1 … r_n` (the script uses `-6.0 : 0.1 : 30.0`, `n = 361`, extending
inside the wall so the blur sees the real edge). The blur is a row-normalised Gaussian:

$$W_{ij} \;=\; \frac{\exp\!\big[-(r_i - r_j)^2 / 2\sigma^2\big]}{\sum_{k} \exp\!\big[-(r_i - r_k)^2 / 2\sigma^2\big]}$$

**Why row-normalised.** The grid is finite. Without dividing by the row sum, points near either end
lose kernel mass and a constant profile would not map to itself. Normalising makes each output a
weighted *mean* of the input, so

$$W\mathbf{1} = \mathbf{1}$$

which is the identity the whole of §5 rests on.

## §2. Measuring `σ` — the only step that reads an image

Take the simulation's own radial HU profile: `m^(v)` measured, `t^(v)` the phantom's per-label
theoretical HU binned identically, for vessel `v`. Fit one scalar:

$$\hat\sigma \;=\; \arg\min_{\sigma}\; \frac{1}{|K|}\sum_{v \in \{\mathrm{rca1},\,\mathrm{lcx}\}}\; \sum_{i \in K} \Big[ \big(W_\sigma\, t^{(v)}\big)_i - m^{(v)}_i \Big]^2$$

with the window `K` = the bins satisfying `-1 mm <= r_i <= 6 mm`, and `σ` scanned over
`0.05 : 0.025 : 2.5`.

**Why a window.** Beyond `6` mm the profile is flat and carries no information about blur; inside
`-1` mm you are in the lumen. **Why those two vessels.** Thick fat cuffs, full radial span, no lung
on the outside — they are the cleanest step edges available.

**Caveat on record.** `σ̂ = 1.05` mm (FWHM `2.473` mm) is an *effective radial* width: it absorbs the
vessel's curvature, the chamfer distance transform, the `0.352` mm binning and the `0.5` mm phantom
raster on top of the true point spread. Direct edge-spread measurement across straight
high-contrast interfaces in the same reconstruction gives FWHM `1.00–1.12` mm, i.e. `σ ≈ 0.45` mm.
§11 shows the difference does not propagate.

## §3. The tissue model — three knobs

$$T(r;\,A,B,\tau) \;=\; A + B\,e^{-\max(r,0)/\tau}$$

`A` is the far-field asymptote, `A + B` the value at the wall surface, `τ` the decay length.
`max(r,0)` continues the fat profile inward as a constant rather than adding a wall (A3).

## §4. Forward prediction at the measurement points

Let `j(i)` be the grid index nearest `d_i`. The model's prediction of the *clinical* measurement is

$$\hat y_i \;=\; \big(W\,T\big)_{j(i)}$$

**Why forward.** Direct inversion of `W` amplifies the measurement noise in exactly the band the
blur removed: Van Cittert at 60 iterations produced `+7.1 / -10.0 / +7.6 / -6.3 / +8.5 / -8.7` HU
alternating every millimetre. Blurring forward and fitting three parameters has no such mode, and
the residual (§9) reports honestly whether three parameters were enough.

## §5. The structural fact — linear in `(A, B)`

Write `e_τ` for the vector `e_τ(r) = exp(-max(r,0)/τ)`. Then `T = A\mathbf{1} + B e_τ`, and because
`W` is linear and `W\mathbf{1} = \mathbf{1}` (§1),

$$W T \;=\; A\,W\mathbf{1} + B\,W e_\tau \;=\; A\,\mathbf{1} + B\,W e_\tau$$

Define the **only quantity that needs the expensive blur**:

$$u_i(\tau) \;\equiv\; \big(W e_\tau\big)_{j(i)}, \qquad i = 1 \dots 20$$

so that

$$\hat y_i \;=\; A + B\,u_i(\tau)$$

**Why this matters.** For fixed `τ` the model is a straight line in `(A, B)`. The original code
searched all three on a grid — `51 × 71 × 80 = 289{,}680` blurs — and the diseased fit came back
pinned at `A = -110.0`, the *first* value of `A ∈ -110 : 1 : -60`. That is a boundary of the search
box, not a minimum. Solving `(A, B)` in closed form deletes both grids and the bug with them.

## §6. The objective

$$\chi^2(A,B,\tau) \;=\; \sum_{i=1}^{20} w_i \big(A + B\,u_i(\tau) - y_i\big)^2, \qquad w_i = \frac{1}{s_i^{\,2}}$$

**Why `1/s²`.** Inverse-variance weighting on the standard error of the cohort mean — the layers the
published figure pins down tightest pull hardest. See A6 for what this does *not* mean.

## §7. Normal equations

Set `∂χ²/∂A = 0` and `∂χ²/∂B = 0`. With the five weighted sums

$$S_0 = \sum_i w_i, \quad S_u = \sum_i w_i u_i, \quad S_{uu} = \sum_i w_i u_i^2, \quad S_y = \sum_i w_i y_i, \quad S_{uy} = \sum_i w_i u_i y_i$$

the stationarity conditions are a `2 × 2` linear system:

$$\begin{pmatrix} S_0 & S_u \\ S_u & S_{uu} \end{pmatrix} \begin{pmatrix} A \\ B \end{pmatrix} \;=\; \begin{pmatrix} S_y \\ S_{uy} \end{pmatrix}$$

## §8. Closed form

$$\Delta \;=\; S_0 S_{uu} - S_u^{\,2}$$

$$A(\tau) \;=\; \frac{S_{uu}\,S_y - S_u\,S_{uy}}{\Delta}, \qquad B(\tau) \;=\; \frac{S_0\,S_{uy} - S_u\,S_y}{\Delta}$$

`Δ` is a weighted variance of `u` and vanishes only if `u` is constant across the 20 layers, i.e.
if that `τ` carries no shape at all. The script skips such a `τ` explicitly rather than dividing by
zero and letting a `NaN` fail the `<` comparison silently.

## §9. `τ` by one-dimensional scan, with a boundary guard

$$\hat\tau \;=\; \arg\min_{\tau \,\in\, \{0.5,\,1.0,\,\dots,\,120.0\}} \; \chi^2\big(A(\tau),\,B(\tau),\,\tau\big)$$

then **assert** `0.5 < τ̂ < 120.0`. A fit sitting on either end of the grid is a boundary solution
and must not reach the phantom silently — that is precisely how the old `A = -110.0` hid.

The reported goodness of fit is the weighted root-mean-square residual in units of the standard
error of the mean:

$$\bar\chi \;=\; \sqrt{\chi^2_{\min} / N}, \qquad N = 20$$

Fitted values: healthy `(A, B, τ) = (-98.638, 24.989, 22.50)` with `χ̄ = 0.467`;
diseased `(-127.639, 60.555, 47.00)` with `χ̄ = 0.764`. **`χ̄ < 1` means the three-parameter model
fits the twenty points to inside their error bars** — the model is smoothing, not straining.

The script also asserts the defining property of the solution — that the weighted residual is
orthogonal to both columns of the design matrix:

$$\sum_i w_i\big(A + Bu_i - y_i\big) = 0, \qquad \sum_i w_i u_i \big(A + Bu_i - y_i\big) = 0$$

which is what §7 says and is the check that the algebra in §8 was transcribed correctly.

## §10. Tissue HU per layer

$$H_i \;=\; T(d_i) \;=\; A + B\,e^{-d_i/\tau}$$

## §11. How much did the blur actually do?

For a smooth `f`, a normalised Gaussian of width `σ` acts to second order as

$$ (Wf)(r) \;\approx\; f(r) + \tfrac{1}{2}\sigma^2 f''(r) $$

For the fitted model, `f'' = (B/\tau^2)\,e^{-r/\tau}`, so the blur's effect on the prediction is

$$\big|\hat y - T\big| \;\approx\; \frac{\sigma^2}{2\tau^2}\,B\,e^{-r/\tau}$$

The scale-free factor is `σ²/(2τ²)`. At `σ = 1.05` mm and `τ = 22.5` mm that is
`1.1025 / 1012.5 = 0.00109`, i.e. **0.11 %**. With `B = 25.0` HU the correction at `r = 1` mm is

$$\frac{1.05^2}{2 \times 22.5^2} \times 25.0 \times e^{-1/22.5} \;=\; 0.5513 \times 0.049383 \times 0.95653 \;=\; 0.026 \;\text{HU}$$

**Consequences, both verified numerically** by re-running the whole fit at both blur widths:

1. Re-running the whole fit at the directly measured `σ = 0.45` mm shifts the recovered tissue HU by
   at most **`0.034` HU** — `-74.736 → -74.765` at `d = 1` mm, `-88.365 → -88.363` at `d = 20` mm
   (healthy; diseased is `0.034` and `0.0001`). The measured maximum lands right on the `0.026` HU
   predicted above. `σ` is effectively inert here, so the wrong `σ` is harmless for this file.
2. Since the forward blur changes the model by `≲ 0.03` HU while the clinical-to-tissue difference
   reaches `2.1` HU, **the deliverable of this step is the three-parameter smoothing, not the
   deconvolution.** What it removes is the clinical curve's non-monotone wiggle at `d = 2–5` mm —
   which sits inside the standard error of the mean, and far inside the per-patient spread (A6).

The `≲ 0.03` HU bound fails only at `r = 0`, where `max(r,0)` puts a kink in `T` and the Taylor
expansion above does not apply. That is why the largest measured discrepancy, `0.08` HU, is at
`d = 1` mm — the layer adjacent to the kink.

## §12. HU is exactly barycentric

By A4, `μ_mix = f_w μ_w + f_l μ_l + f_p μ_p`, and `HU = 1000(μ - μ_w)/μ_w` with `HU_w = 0`:

$$H \;=\; f_w\,\mathrm{HU}_w + f_l\,\mathrm{HU}_l + f_p\,\mathrm{HU}_p \;=\; f_l\,\mathrm{HU}_l + f_p\,\mathrm{HU}_p$$

$$f_w + f_l + f_p = 1$$

**One equation, two free fractions.** A single energy fixes a *line segment* in the composition
triangle, not a point. This is the whole reason a closure is needed.

The prior-free bound on protein follows by setting `f_w = 0`:

$$f_p^{\max} \;=\; \frac{H - \mathrm{HU}_l}{\mathrm{HU}_p - \mathrm{HU}_l}$$

## §13. Closure by the frozen protein-to-water ratio

Impose A5: `f_p = ρ f_w`, hence `f_l = 1 - f_w - f_p = 1 - f_w(1+ρ)`. Substituting into §12:

$$H \;=\; \big[1 - f_w(1+\rho)\big]\mathrm{HU}_l + \rho f_w \mathrm{HU}_p$$

$$H - \mathrm{HU}_l \;=\; f_w\big[\rho\,\mathrm{HU}_p - (1+\rho)\,\mathrm{HU}_l\big]$$

$$\boxed{\;f_w \;=\; \frac{H - \mathrm{HU}_l}{\rho\,\mathrm{HU}_p - (1+\rho)\,\mathrm{HU}_l}, \qquad f_p = \rho f_w, \qquad f_l = 1 - f_w - f_p \;}$$

With `HU_l = -111.69` and `HU_p = +270.58` the denominator is `270.58 ρ + 111.69 (1+ρ)`.

**Self-consistency check built into the script:** feeding the *clinical* `H` back through this
formula must return the clinical CSV's own fractions. It does, to `Δ ≤ 0.0011`.

## §14. Worked example — healthy, `d = 1` mm, by hand

Clinical composition at this layer is `(f_w, f_l, f_p) = (0.2394, 0.7357, 0.0249)`, so

$$\rho = \frac{0.0249}{0.2394} = 0.10401$$

The fitted tissue HU from §10, with `A = -98.638`, `B = 24.989`, `τ = 22.5` mm:

$$H \;=\; -98.638 + 24.989 \times e^{-1/22.5} \;=\; -98.638 + 24.989 \times 0.956529 \;=\; -98.638 + 23.902 \;=\; -74.736$$

Denominator:

$$0.10401 \times 270.58 \;+\; 1.10401 \times 111.69 \;=\; 28.145 + 123.307 \;=\; 151.452$$

Numerator:

$$-74.736 - (-111.69) \;=\; 36.954$$

Therefore

$$f_w = \frac{36.954}{151.452} = 0.24399, \qquad f_p = 0.10401 \times 0.24399 = 0.02538, \qquad f_l = 1 - 0.24399 - 0.02538 = 0.73063$$

The CSV row reads `healthy,1,0.2440,0.7306,0.0254`. Agreement to the printed digits.

---

## §15. Provenance

This procedure is **not** taken from a published method. It is a design step written for this
phantom: the parametric-forward-model-instead-of-inverse-filter idea is standard practice in
ill-posed inversion, but the specific three-parameter profile, the frozen-ratio closure (A5) and
the self-consistency criterion in §0 are local choices made in `pcat_deconv_design.jl`.

The *inputs* have sources — the clinical FAI gradient is Oxford-style perivascular HU versus
distance, the endpoints come from NIST triglyceride and Woodard & White 1986 protein, and the
adipose prior that produced the upstream `ρ` is the digitised Woodard 1986 Fig. 1 distribution.
The *inversion procedure in this file* has no citation, and should not be given one.
