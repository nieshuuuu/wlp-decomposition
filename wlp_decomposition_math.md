# Water–lipid–protein decomposition on a QRM-thorax — the method, step by step

This is the math behind `wlp_decomposition.jl`: recover the **volumetric fractions**
$(f_w, f_l, f_p)$ of water/lipid/protein mixtures, per voxel and per region, from a dual-energy
CT scan synthesized to two virtual monoenergetic images (VMI) at **70 and 150 keV**. Everything
is simulated — a stadium QRM-thorax phantom, `BasisSimulator.jl` v0.8.0 forward physics, and the
inverse — so ground truth is exact and every number below is reproducible from the cached sims
(§9). The one external data file is a Woodard-1986 adipose table used only to draw physiological
compositions.

The notebook delivers **three** estimators, not one, because a fat measurement has three
different failure modes and each estimator answers one of them:

1. a **per-voxel calibration decode** $f=\mathrm{surf}(\mathrm{HU}_{70},\mathrm{HU}_{150})$ for
   point composition accuracy (§5.1);
2. a **boundary-agnostic map** — the per-voxel decode cleaned by a $\sigma_f$-weighted
   edge-preserving Huber-TV, using no ground-truth boundary because real fat gives you none
   (§5.4);
3. an **integrated-HU** (mass-conservation) estimator for the *total* lipid of a small object,
   which the object-extent measure under-reports when partial volume smears fat past its visible
   edge (§5.5).

A voxel whose noisy decode lands *outside* the composition triangle is handled by a noise-ellipse
maximum-likelihood projection (§5.3), applied at the pooled-ROI level.

Same forward model feeds all three; §6 validates them against truth; §7 explains the 70/150 keV
choice; §8 states plainly what the code does **not** do and where the numbers stop transferring.

> **Reproducibility gate.** The recon must be quantitative before any decode is trustworthy: on
> the 52 calibration ROIs the reconstructed HU matches the theoretical linear-mixture HU to a mean
> of **−2.3 HU at 70 keV** and **+0.5 HU at 150 keV** (|max| < 5 HU; §2.5). This requires
> BasisSimulator **v0.8.0 with the `:dd_fast` projector**, a **wide bowtie-free geometry** (1300
> detector columns ≈ 415 mm scan FOV, `bowtie=:none`) so the 350 mm fat ring is not truncated,
> and **984 views** to suppress aliasing. (Scanner: source-to-isocenter 625.6 mm, source-to-detector
> 1100 mm, 1300 × 256 detector at 0.6 × 0.625 mm.) A quantitative recon is a prerequisite, not a
> given — an under-resolved or bowtie-vignetted projector compresses the non-water HU range and
> silently caps the recoverable fractions.

---

## 1. What you calibrate, once

Before the inverse runs, five things are fixed from the calibration sims. Each is a property of
*this* acquisition (spectrum, projector, recon), not of the method, so none transfer to a real
scanner unchanged — §8 is explicit about that. The five:

1. **endpoints** $p_w, p_l, p_p$ — where pure water, lipid, and protein sit in the
   $(\mathrm{HU}_{70},\mathrm{HU}_{150})$ plane. Water is $ (0,0) $ by the HU definition; lipid and
   protein are **theoretical** (NIST cross-sections, §2.2), because no real scan supplies a pure
   lipid or pure collagen rod.
2. **calibration surfaces** $c_w, c_l, c_p$ — the coefficients of three quadratics in
   $(\mathrm{HU}_{70},\mathrm{HU}_{150})$, fit by least squares on known mixtures (§5.1).
3. **noise model** $\sigma_E(\mathrm{HU})$ per energy — a convex quadratic, degenerating to
   near-constant over the soft-tissue range actually sampled (§5.2).
4. **inter-energy correlation** $\rho$ — measured from the paired within-ROI residuals (§5.2).
5. **decode-to-map regulariser** — the $\sigma_f$ weight and the Huber-TV $(\lambda,\varepsilon)$
   that turn the speckled per-voxel decode into a map (§5.4).

---

## 2. The forward model

### 2.1 A mixture is one effective material — exactly

A voxel that is volume fractions $(f_w, f_l, f_p)$ of water/lipid/protein, with pure-material
densities $\rho_w, \rho_l, \rho_p$, is built into a single `XrayAttenuation.Material` so the
simulator sees one homogeneous medium. The construction (the notebook's `wlp_material`):

$$
\rho_{\text{eff}} = \sum_v f_v\,\rho_v ,
\qquad
m_v = \frac{f_v\,\rho_v}{\rho_{\text{eff}}}\ \ (\text{mass fraction}),
\qquad
w_Z = \sum_v m_v\,w_{Z}^{(v)} ,
$$

where $w_Z^{(v)}$ is element $Z$'s mass fraction in pure material $v$. Densities used:
$\rho_w = 1.00$, $\rho_l = 0.92$, $\rho_p = 1.35\ \mathrm{g/cm^3}$ (lipid from the Paternò-2020
basis set; protein is Woodard & White 1986 soft-tissue collagen, composition
$w = \{\mathrm H\,0.066,\ \mathrm C\,0.534,\ \mathrm N\,0.170,\ \mathrm O\,0.220,\ \mathrm S\,0.010\}$).

This is not an approximation. The linear attenuation of the effective material telescopes back to
the volume-weighted sum of the pure linear attenuations, at **every** photon energy:

$$
\mu_{\text{eff}}(E)
= \rho_{\text{eff}} \sum_Z w_Z \Big(\tfrac{\mu}{\rho}\Big)_Z(E)
= \rho_{\text{eff}} \sum_v m_v \sum_Z w_Z^{(v)} \Big(\tfrac{\mu}{\rho}\Big)_Z(E)
= \sum_v f_v\,\rho_v \Big(\tfrac{\mu}{\rho}\Big)_v(E)
= \sum_v f_v\,\mu_v(E) .
$$

The middle step uses $\rho_{\text{eff}} m_v = f_v \rho_v$; the last uses
$\mu_v = \rho_v (\mu/\rho)_v$. So **volume fractions add linearly in $\mu$** — the fact the whole
inverse leans on.

### 2.2 The mixture HU is a barycentric point — the decomposition triangle

A VMI at energy $E$ is monoenergetic by construction, so there is no spectrum to average over:
its HU is the single-energy definition applied to $\mu_{\text{eff}}(E)$,

$$
\mathrm{HU}_{\text{mix}}(E) = 1000\,\frac{\mu_{\text{eff}}(E) - \mu_w(E)}{\mu_w(E)} .
$$

Substitute §2.1's sum and use closure $f_w + f_l + f_p = 1$ to eliminate $f_w$:

$$
\mu_{\text{eff}} - \mu_w
= f_l(\mu_l - \mu_w) + f_p(\mu_p - \mu_w)
\quad\Longrightarrow\quad
\boxed{\ \mathrm{HU}_{\text{mix}}(E) = f_l\,p_l(E) + f_p\,p_p(E)\ }
$$

with the **theoretical endpoints**

$$
p_i(E) = 1000\,\frac{\mu_i(E) - \mu_w(E)}{\mu_w(E)} ,
\qquad p_w(E) \equiv 0 .
$$

So a mixture lands at the barycentric combination $f_w\,p_w + f_l\,p_l + f_p\,p_p$ of the three
vertices — the noiseless locus is the **triangle** with corners $p_w, p_l, p_p$, and every
calibration core must fall inside it (it does; fig 1). Evaluated from NIST XCOM cross-sections at
the canonical pair:

| endpoint | $\mathrm{HU}_{70}$ | $\mathrm{HU}_{150}$ |
|---|---|---|
| water $p_w$ | 0.0 | 0.0 |
| lipid $p_l$ | **−111.7** | **−81.2** |
| protein $p_p$ | **+270.6** | **+290.1** |

(Legacy 40 keV values, kept only for the `WLP_PAIR` knob: lipid −212.7, protein +204.2 HU.)

### 2.3 The phantom

A stadium QRM-thorax built in code (PCATSim geometry), not a mask file: a
$925 \times 675$ grid at 0.4 mm/px carrying a fat-ring / muscle / lung stadium body (half-length
50 mm; nested radii lung 80, muscle 100, fat ring 125 mm), two lungs split by a mediastinal muscle
column, ribs, a spine with vertebral arch, and a **heart cavity** (radius 55 mm at image point
(185, 115) mm) that holds the material inserts. Three insert geometries share this body:

- **circular** — 13 hex-packed inserts of ø **19.9 mm**, the diameter *derived* from the recon
  pixel (0.7422 mm) so that a 2-pixel-eroded interior core is ≈ 209 mm² (radius 11 recon-px). Used
  for calibration and for the held-out circular test.
- **sector** — the heart disk (radius 50 mm) partitioned into **8 angular × 2 radial = 16 solid
  sectors**, a deliberately different (held-out) shape.
- **size series** — a single centred fat insert at radii 4, 6, 9, 12 mm, one per sim, for the
  integrated-HU partial-volume study (§5.5).

The inserts are extruded along $z$ (the phantom is $z$-uniform), so after recon the usable slices
are near-identical denoised planes.

The stadium body itself is anatomical **scaffolding**, not scored material: the fat ring is
`BS.XA.Materials.adipose` (ICRU-44 Adipose Tissue, ρ 0.92), not the W/L/P lipid basis. Adipose is
not pure triglyceride — decomposed in the W/L/P basis it is $f_l \approx 0.81$, $f_w \approx 0.24$,
$f_p \approx -0.05$ (just outside the triangle, below the water–lipid edge). A correct map therefore
renders the ring as ~0.8 lipid, bright but not saturated; only the *scored* inserts (calibration,
circular, sector) carry known $(f_w, f_l, f_p)$ and enter any accuracy number.

### 2.4 Simulate → basis → VMI

Each acquisition (the notebook's `run_acq`) is one dual-kVp EICT simulation reconstructed to two
VMIs:

1. **dual-kVp scan** — 80 kVp and 140 kVp tubes (mA split 0.65 / 0.35 of a 407/405 reference,
   0.5 s rotation, 2.5 mm collimation, 4.5 mm added Al), `SimOptions(fidelity=:eict, projector=:dd_fast, use_noise=true)`, on the wide bowtie-free scanner of the gate note.
2. **material basis** — a Cong water/iodine two-basis decomposition of the 80/140 sinogram pair
   (spectra resolved with `resolve_source_spectrum_full` on the *same* geometry). Lipid and protein
   have no K-edge in band, so the water/iodine basis represents them faithfully; their fitted
   iodine density is legitimately **negative** and is never clamped.
3. **FBP** — FDK reconstruction (soft filter) to 512×512×3 at 380 mm FOV, then a 3-slice
   cross-$z$ median (`zmed=1`).
4. **VMI synthesis** — `synth_vmi_2basis` evaluates the basis maps at 70 and 150 keV, giving the
   two channels $(\mathrm{HU}_{70},\mathrm{HU}_{150})$ the inverse consumes.

All statistics use **eroded ROI cores** (2-px erosion) to keep partial-volume edge voxels out of
the HU/σ measurements.

### 2.5 Forward fidelity — recon vs theory

Because §2.1–2.2 predict the mixture HU exactly, the simulator is falsifiable: the reconstructed
ROI-mean HU of each known-composition insert must equal $f_l p_l(E) + f_p p_p(E)$. Across all 52
calibration cores:

$$
\text{recon} - \text{theory} :\quad
\begin{cases}
70\ \text{keV}: & -2.26 \pm 1.16\ \mathrm{HU},\ |\text{max}| = 4.8 \\
150\ \text{keV}: & +0.51 \pm 1.76\ \mathrm{HU},\ |\text{max}| = 3.7
\end{cases}
$$

A residual bias of a couple of HU at 70 keV (mild beam-hardening / basis-synthesis offset) and
essentially zero at 150 keV, everywhere under 5 HU. The mixture model is linear-additive as
derived, and the recon is quantitative — so a decode built on it is measuring physics, not
artifact. (The notebook's inline warning still cites a pure-lipid "−204 vs −213 HU" figure; that
is a stale **40 keV** number from an earlier run and does not describe the 70/150 pair.)

---

## 3. Why three materials from two energies is hard

The measurement is a point $m = (\mathrm{HU}_{70},\mathrm{HU}_{150})$ in a plane; the unknown is a
point $(f_w, f_l, f_p)$ on the 2-simplex $f_w+f_l+f_p=1$ — also two free coordinates. Writing the
water-referenced problem with $\theta = (f_l, f_p)$,

$$
m - p_w = G\,\theta + \varepsilon ,
\qquad
G = \big[\,p_l - p_w \ \big|\ p_p - p_w\,\big]
= \begin{pmatrix} p_l(70) & p_p(70) \\ p_l(150) & p_p(150) \end{pmatrix} ,
$$

is a **square** $2\times2$ system: exactly determined, no spare equation. There is no residual to
test, no way to detect a voxel that is not a WLP mixture, and — critically — noise passes straight
through the inverse $G^{-1}$ with no averaging. Adding protein turned a projection (2-material
water/lipid, overdetermined, self-checking) into an inversion (3-material, exactly determined,
credulous). That is the structural cost of the third material.

How bad the inversion is depends on the shape of the triangle, measured by the conditioning of
$G$. At the canonical pair,

$$
G = \begin{pmatrix} -111.7 & 270.6 \\ -81.2 & 290.1 \end{pmatrix} ,
\qquad
\operatorname{cond}(G) = 16.9 .
$$

The three tissues are all low-$Z$ and near-water, so their vertices are nearly collinear — the
triangle is a **sliver**, and water and protein in particular sit close together in the plane. A
sliver means $G$ maps a round noise ball in HU-space into a long thin ellipse in
$(f_l, f_p)$-space: the fraction most aligned with the sliver's short axis, **$f_p$**, is the
high-variance one. This is the Alvarez–Macovski ceiling — CT attenuation is intrinsically 2-D
(photoelectric + Compton), so three low-$Z$ materials cannot be cleanly separated by two energies
no matter how clean the data. The information that stabilises $f_p$ has to come from **spatial
pooling** ($\sqrt N$ over an ROI core) or from a **prior** — not from more energies at this scale.

§7 returns to $\operatorname{cond}(G)$: the 70/150 pair is 3.5× worse-conditioned than 40/70
(cond 4.8), yet the pooled ROI accuracy is unaffected, which is the counterintuitive finding the
`WLP_PAIR` experiment was built to show.

---

## 4. Drawing the compositions

Calibration and test compositions come from **disjoint random streams** (different seeds), so the
surface is never fit and tested on the same mixture. Two generators:

**Physiological (`draw_wlp`)** — a smoothed-bootstrap KDE over the 59-point Woodard-1986 adipose
table (`data/adipose_composition_distribution.csv`), keeping rows with lipid ≥ 50 % in states
{healthy, obese, reduced, comparison, unspecified}. Let $L$ be the kept lipid mass-percents and
$(s_l, s_p)$ the lipid/protein split of the "protein" rows. For each draw:

$$
l = \operatorname{clamp}\!\big(L[\text{rand}] + b_L\,\mathcal N(0,1),\ 50,\ 99\big),
\quad b_L = \operatorname{std}(L)\,|L|^{-1/5}\ (\text{Scott});
$$
$$
q = \operatorname{logistic}\!\big(\mu_z + \sigma_z\,\mathcal N(0,1)\big),
\quad z = \operatorname{logit}\!\frac{s_p}{100 - s_l};
\qquad p = q\,(100 - l),\quad w = 100 - l - p ,
$$

then mass→volume by $f_v \propto (\text{mass}_v/\rho_v)$, normalised. The logit keeps $f_p > 0$;
the KDE reproduces the physiological $f_w\!\leftrightarrow\!f_l$ anticorrelation from closure. This
generator supplies the **adipose prior** object the notebook builds (2000 draws) — but see §8:
that prior is *not* consumed by the decode that runs.

**Spanning (`diverse_comps`)** — to stress the surface beyond adipose, the *test* inserts are drawn
uniformly: $f_l \in [0.05, 0.95]$, $f_p \in [0, 0.30]$, $f_w = 1 - f_l - f_p$ floored at 0.02. This
widens the true-fraction ranges the CCC is computed over ($f_w \in [0.02, 0.83]$,
$f_l \in [0.05, 0.91]$, $f_p \in [0, 0.30]$), so the correlation coefficients are not inflated by a
narrow spread.

| set | seeds | sims × inserts | ROIs |
|---|---|---|---|
| calibration | 11–14 (comps 1001–1004) | 4 × 13 circular | **52** |
| circular test | 201–205 (comps 91260709…) | 5 × 13 | 65 |
| sector test | 401,402,405,406 (comps 70260709…) | 4 × 16 | 64 |
| **test total** | | | **129** |

---

## 5. The inverse

### 5.1 The calibration surface — a direct decode

The exact inverse would be $\theta = G^{-1}(m - p_w)$ using the theoretical endpoints. The notebook
does **not** do that. Instead it fits an empirical map from measured HU straight to each fraction,
which absorbs the residual beam-hardening / basis offset of §2.5 without a separate debiasing step.
Define the quadratic feature vector

$$
\phi(\mathrm{HU}_{70},\mathrm{HU}_{150}) =
\big[\,1,\ \mathrm{HU}_{70},\ \mathrm{HU}_{150},\ \mathrm{HU}_{70}^2,\ \mathrm{HU}_{150}^2,\ \mathrm{HU}_{70}\mathrm{HU}_{150}\,\big] ,
$$

and fit three coefficient vectors $c_w, c_l, c_p$ by ordinary least squares against the known
fractions of the 52 calibration cores:

$$
c_i = \arg\min_c \sum_{k=1}^{52}\big(c^{\mathsf T}\phi(m_k) - f_{i,k}\big)^2 ,
\qquad i \in \{w, l, p\} .
$$

The three surfaces are fit **independently**, so a raw evaluation need not sum to one; the decode
closes it by normalising:

$$
(\tilde f_w, \tilde f_l, \tilde f_p) = \big(c_w^{\mathsf T}\phi,\ c_l^{\mathsf T}\phi,\ c_p^{\mathsf T}\phi\big),
\qquad
\hat f_i = \frac{\tilde f_i}{\tilde f_w + \tilde f_l + \tilde f_p} .
$$

Fitted coefficients (canonical pair; $\mathrm{HU}$ in units of HU):

$$
c_w = [\,0.9817,\ 0.03056,\ -0.03209,\ 1.0\!\times\!10^{-5},\ 3.0\!\times\!10^{-5},\ -4.0\!\times\!10^{-5}\,]
$$
$$
c_l = [\,0.00791,\ -0.02424,\ 0.02265,\ -1.0\!\times\!10^{-5},\ -3.0\!\times\!10^{-5},\ 3.0\!\times\!10^{-5}\,]
$$
$$
c_p = [\,0.01040,\ -0.00632,\ 0.00944,\ -0.0,\ -1.0\!\times\!10^{-5},\ 1.0\!\times\!10^{-5}\,]
$$

The quadratic terms are $O(10^{-5})$ — the surface is nearly the linear inverse $G^{-1}$, with the
curvature mopping up second-order recon nonlinearity. Fit quality on the calibration set:
$R^2(f_w)=0.996$, $R^2(f_l)=0.997$, $R^2(f_p)=0.998$.

**Worked decode** (a circular test insert, held out from the fit). Measured ROI-mean HU
$(\mathrm{HU}_{70},\mathrm{HU}_{150}) = (-21.29,\ 0.47)$, true composition
$(f_w, f_l, f_p) = (0.301,\ 0.547,\ 0.152)$. Evaluate the three surfaces:

$$
\tilde f_w = c_w^{\mathsf T}\phi = 0.3196,\qquad
\tilde f_l = c_l^{\mathsf T}\phi = 0.5320,\qquad
\tilde f_p = c_p^{\mathsf T}\phi = 0.1484 ,
$$
$$
\tilde f_w + \tilde f_l + \tilde f_p = 1.00000
\quad\Longrightarrow\quad
(\hat f_w, \hat f_l, \hat f_p) = (0.320,\ 0.532,\ 0.148) .
$$

The raw surfaces already sum to 1.00000 here, so normalisation is a null correction — the three
independent fits are jointly consistent on real test data. The decode lands within 0.019 of truth
on every fraction. (Cross-check: this true composition sits at theoretical HU
$f_l p_l + f_p p_p = (-20.05,\ -0.41)$, within the §2.5 recon residual of the measured
$(-21.29, 0.47)$ — the loop closes.)

### 5.2 The noise model

Each energy carries a heteroscedastic noise level. The notebook fits a convex quadratic in HU by
least squares over the calibration cores' within-ROI σ, dropping to affine if the curvature comes
out negative:

$$
\sigma_E(\mathrm{HU}) = a_E\,\mathrm{HU}^2 + b_E\,\mathrm{HU} + c_E ,\qquad a_E \ge 0 .
$$

Over the soft-tissue HU range actually sampled (roughly −85 to +60 HU) the fit is essentially flat:

$$
\sigma_{70}(\mathrm{HU}) \approx 9.7\ \mathrm{HU}\ \ (a=0,\ b=0.0022,\ c=9.73),
\qquad
\sigma_{150}(\mathrm{HU}) \approx 2.2\ \mathrm{HU}\ \ (a=3.5\!\times\!10^{-5},\ b\approx0,\ c=2.23) .
$$

These are the quadratic-fit intercepts used for the per-voxel σ_f weighting in the map step (§5.4).
The out-of-triangle **noise ellipse** in §5.3 instead uses the pooled within-ROI residual covariance
directly — its standard deviations are $\sigma_{70}=9.7$ and $\sigma_{150}=2.3$ HU, matching the
intercepts above to within rounding (pooled $\sigma_{150}=2.27$ vs fitted $2.23$).

The 70 keV channel is ~4× noisier than 150 keV — the photoelectric-rich low-energy VMI amplifies
both contrast and noise. The two channels are synthesized from the *same* basis maps, so their
noise co-fluctuates; the inter-energy correlation from the pooled centred within-ROI residuals
(19 872 voxels) is

$$
\rho = \frac{\sum_i \delta_{70,i}\,\delta_{150,i}}
            {\sqrt{\sum_i \delta_{70,i}^2}\,\sqrt{\sum_i \delta_{150,i}^2}} = 0.787 .
$$

This defines the per-voxel noise covariance

$$
\Sigma = \begin{pmatrix} \sigma_{70}^2 & \rho\,\sigma_{70}\sigma_{150} \\
\rho\,\sigma_{70}\sigma_{150} & \sigma_{150}^2 \end{pmatrix} ,
$$

which the map step (§5.4) propagates into a per-voxel fraction uncertainty, and which also defines
the metric for the out-of-triangle handling next.

### 5.3 Points outside the triangle — the noise-ellipse MLE

§2.2 says a noiseless mixture lands *inside* the triangle $\{f_w, f_l, f_p \ge 0\}$. A noisy
measurement need not: the decode of §5.1 divides three independently-fit surfaces by their sum, so
a voxel scattered off the manifold by $\varepsilon$ can produce a **negative** fraction — a
composition that does not physically exist. This is not rare. On the 129 test ROIs, **27.8 % of
individual voxels decode to an infeasible composition** ($f_w < 0$ in 16 %, $f_p < 0$ in 7 %,
$f_l < 0$ in 5 %; the most negative reaches $f_w = -0.94$). The decode as written simply passes
these through — the reported fraction is then a linear extrapolation past a vertex, not a
composition.

The principled fix is the same maximum-likelihood step the 2-material water/lipid decode uses, now
in two dimensions. Under Gaussian noise with covariance $\Sigma$ (§5.2), the log-likelihood of a
composition $\theta = (f_l, f_p)$ given a measurement $m = (\mathrm{HU}_{70}, \mathrm{HU}_{150})$ is

$$
\log \mathcal L(\theta \mid m) = -\tfrac12\,(m - b - G\theta)^{\mathsf T}\,\Sigma^{-1}\,(m - b - G\theta) + \text{const},
$$

where $G = [\,p_l - p_w \mid p_p - p_w\,]$ is the endpoint matrix of §3 and $b$ is the calibration
bias of §2.5 (the mean recon−theory offset, $b = (-2.26,\ +0.51)$ HU). **The maximum-likelihood
feasible composition is the point on the triangle that minimises the $\Sigma^{-1}$-weighted
(Mahalanobis) distance to the measurement** — not the Euclidean-closest point. This distinction is
the whole content of "noise ellipse": because $\sigma_{70} \approx 4\,\sigma_{150}$ and
$\rho = 0.79$, the equal-likelihood contours are **tilted ellipses**, not circles, so the closest
feasible composition in likelihood is generally *not* the one you would get by clamping the raw
decode. Concretely, the projection maps the measurement to

$$
\hat\theta = \arg\min_{\theta \in \triangle}\ (\theta - \theta^\star)^{\mathsf T}\,A\,(\theta - \theta^\star),
\qquad A = G^{\mathsf T}\Sigma^{-1}G,
\qquad \theta^\star = G^{-1}(m - b),
$$

evaluated in closed form by projecting $\theta^\star$ onto each of the three triangle edges in the
metric $A$ and keeping the nearest — an $O(1)$ operation per point, no iteration
(`proj_simplex` / `decode_feas` in the notebook). The metric $A$ is exactly the Fisher information
of $\theta$, so this is the constrained MLE, and it reduces to the ordinary decode whenever the raw
result is already interior.

**Where to apply it — and where not.** The natural temptation is to project every voxel. That is
*wrong*, and quantifiably so: projecting each noisy voxel before averaging a ROI rectifies the
noise (a one-sided clamp is a nonlinear operation, so $\mathbb E[\text{clamp}(x)] \ne
\text{clamp}(\mathbb E[x])$ — Jensen), which drags the pooled fraction toward the interior and
*inflates* error. On these data per-voxel projection drops the pooled $f_w$ concordance from
$0.996$ to $0.980$. The infeasible voxels are not a modelling failure to be corrected point by
point — they are the expected noise excursions of genuine near-edge compositions: **96 % of them
sit within $2\sigma$ of the triangle** (median Mahalanobis distance $0.70\sigma$). Averaging the
ROI in the *measurement* domain lets those excursions cancel, which is what the $\sqrt N$ pooling of
§6 already does.

So the estimator keeps the two domains separate:

- **per-voxel point accuracy and the map** use the raw `decode` — noise is meant to average out
  spatially (§5.4), and projecting first would bias it;
- **the pooled ROI composition** is read through `decode_feas`, which is feasible by construction:
  it applies the noise-ellipse projection only to the pooled mean, and only when that mean still
  lands outside. Only **7 of the 129 ROI means** are outside the triangle at all, and projecting
  them leaves the concordance unchanged (pooled-feasible CCC $f_w/f_l/f_p = 0.996/0.997/0.998$)
  while guaranteeing every delivered composition is physical.

Figure `fig_decode_triangle_noise.png` shows both halves: the full sliver triangle with the measured
voxel cloud coloured by feasibility (left), and a zoom on the water–lipid corner where the noise
ellipse is legible and the 7 out-of-triangle ROI means are projected back onto the edge along the
Mahalanobis metric (right).

![The decomposition triangle and the noise-ellipse MLE. Left: the water/lipid/protein endpoints form
a near-collinear sliver (cond G = 16.9); 27.8 % of per-voxel decodes (purple) fall outside it under
noise. Right: zoom on the water–lipid corner — the noise covariance Σ is a tilted ellipse
(σ₇₀ = 9.7, σ₁₅₀ = 2.3 HU pooled residual std, ρ = 0.79), and the 7 of 129 ROI means whose pooled decode still lands
outside are projected back onto the triangle along the Σ⁻¹ (Mahalanobis) metric.](assets/fig_decode_triangle_noise.png)

### 5.4 From decode to map — the σ_f-weighted Huber-TV

The per-voxel decode applied straight to an image is unbiased but speckled: the sliver of §3 makes
each voxel's $(f_l, f_p)$ noisy. There are two honest ways to spend that noise down — average an
ROI and let $\sqrt N$ work (that is §6's pooled accuracy), or, when the deliverable is a *map*, let
each voxel borrow strength from its neighbours only where the map is locally flat. The second is
the delivered map.

**Step 1 — propagate σ into a per-voxel fraction precision.** The decode is a smooth function
$f(\mathrm{HU}_{70},\mathrm{HU}_{150})$, so first-order error propagation through the surface
gradients $\partial f_i/\partial\mathrm{HU}_E = c_i^{\mathsf T}\partial_E\phi$ gives each fraction's
variance under the §5.2 covariance:

$$
\operatorname{Var}(f_i) =
\Big(\tfrac{\partial f_i}{\partial\mathrm{HU}_{70}}\Big)^2 \sigma_{70}^2
+ \Big(\tfrac{\partial f_i}{\partial\mathrm{HU}_{150}}\Big)^2 \sigma_{150}^2
+ 2\rho\,\tfrac{\partial f_i}{\partial\mathrm{HU}_{70}}\tfrac{\partial f_i}{\partial\mathrm{HU}_{150}}\sigma_{70}\sigma_{150} ,
$$

and the data weight is the inverse total fraction variance,
$w = 1/\max(\operatorname{Var}(f_l) + \operatorname{Var}(f_p),\ 10^{-6})$.

*Worked* (at the §5.1 ROI, $\mathrm{HU}=(-21.29, 0.47)$): $\sigma_{70}=9.68$, $\sigma_{150}=2.23$,
$\rho=0.787$; the $f_l$ surface gradients are $\partial_{70}=-0.0240$, $\partial_{150}=+0.0221$,
giving a **per-voxel** $\sigma_{f_l} = 0.196$ and $\sigma_{f_p} = 0.046$, hence $w = 24.7$. That
per-voxel $\sigma_{f_l}\approx0.20$ is exactly the speckle a single voxel carries; the observed
within-core scatter of the decode at this ROI is $0.211$ — the noise model predicts the realized
per-voxel spread to ~7 %.

**Step 2 — MAP map = data pull + edge-preserving prior.** Read the clean map as the minimiser of a
negative log-posterior with a Gaussian data term (weight $w$) and a coupled Huber-TV prior on
$(f_l, f_p)$:

$$
\hat f = \arg\min_{f}\ \sum_i w_i\,\lVert f_i - \hat f_i^{\text{decode}}\rVert^2
\ +\ \lambda \sum_{\langle i,n\rangle} \varphi_\varepsilon\!\big(\lVert f_i - f_n\rVert\big) .
$$

The Huber coupling between neighbours is $c_{in} = \lambda/\max(\lVert f_i - f_n\rVert,\ \varepsilon)$
— strong where the map is flat (noise averages out), weak across a genuine material edge (kept).
Lagging the diffusivity turns the minimisation into a weighted-Jacobi sweep, applied jointly to
$(f_l, f_p)$ and simplex-projected each pass:

$$
f_i \leftarrow \frac{w_i\,\hat f_i^{\text{decode}} + \sum_n c_{in} f_n}{w_i + \sum_n c_{in}} ,
\qquad
f_w = 1 - f_l - f_p .
$$

Parameters $\lambda = 0.05$, $\varepsilon = 0.04$, 25 sweeps. The map is **boundary-agnostic** by
construction — it uses no ground-truth insert boundary, only HU-gated soft tissue
($-300 < \mathrm{HU}_{150} < 250$, cutting lung/gas below and bone/mineral above). This is the
honest map: real pericoronary fat gives you no ground-truth boundary, so a decomposition that
secretly used one would not transfer. Figures 3–4 contrast it against the ground-truth-pooled map,
which looks cleaner precisely because it cheats with a boundary you do not have.

**Why the per-pass projection is *Euclidean*, not the §5.3 Mahalanobis metric.** The map's
simplex projection is a plain non-negativity clamp, and it is deliberately kept separate from the
ROI-level MLE of §5.3. Two measured reasons. (i) On this phantom's fat ring the Mahalanobis metric
is the *wrong* projection: the ring is ICRU-44 adipose, whose true composition
($f_l = 0.81,\ f_w = 0.24,\ f_p = -0.05$) sits just below the water–lipid edge, i.e. barely outside
the triangle. The metric $A = G^{\mathsf T}\Sigma^{-1}G$ is $\sim\!1700\times$ anisotropic with its
cheap direction nearly along $f_l$, so projecting adipose onto the $f_p=0$ edge slides it *up the
lipid axis* to $f_l \approx 0.99$ (looks pure lipid), whereas the Euclidean clamp leaves it at the
correct $0.81$. (ii) On the scored heart rods the two clamps are indistinguishable (median
$|\Delta f_l| = 0.0006$; ROI $f_l$ CCC $0.975$ Euclidean vs $0.977$ Mahalanobis). So the
noise-ellipse metric earns its place only at the pooled ROI decode (§5.3), where the quantity is a
single mean; on a *per-voxel* map any one-sided clamp already trades ROI accuracy for a feasible
picture (unclamped ROI $f_l$ CCC $0.992$ → clamped $0.975$, the Jensen effect of §5.3 acting
pixelwise), and the gentler Euclidean clamp is the safer default. `fig_map_clamp.png` shows the
three side by side.

![Delivered f_l map over the heart under three per-voxel clamp strategies. Left: true rod
fractions. The unclamped raw decode has the best ROI f_l CCC (0.992) but 83 % of its voxels are
individually infeasible; the TV + Euclidean clamp (current delivered map) and the TV + Mahalanobis
clamp are visually indistinguishable (CCC 0.975 vs 0.977), and both trade ROI accuracy for a
feasible picture. The fat ring stays bright in all three; under a per-voxel Mahalanobis clamp it
would saturate to pure lipid, which is why the map keeps the Euclidean
projection.](assets/fig_map_clamp.png)

### 5.5 Integrated-HU — total lipid without a boundary

Point accuracy and the map both estimate a *fraction field*. A different quantity matters for small
fat objects: the **total lipid** they contain. Partial volume smears a small fat insert's signal
past its visible edge, so measuring lipid only inside the object's apparent extent under-reports the
total. A normalised FBP point-spread function conserves the integral, so the lost lipid is recovered
by integrating over a **generous region** (object + skirt) rather than the object alone — provided
you subtract a background, because the surrounding muscle is not lipid-free in the affine decode.

The estimator uses the **affine** lipid decode $\hat f_l^{\text{aff}} = c_{l,\text{aff}}^{\mathsf T}[1,\mathrm{HU}_{70},\mathrm{HU}_{150}]$
(linear, so it commutes with the PSF; $c_{l,\text{aff}} = [0.00962,\ -0.02392,\ 0.02194]$). Fit a
per-scan quadratic muscle background $b(i,j)$ over a 60 mm muscle annulus (captures cupping), then

$$
\text{lipid area} = \int_{\text{region}} \big(\hat f_l^{\text{aff}}(i,j) - b(i,j)\big)\, dA ,
$$

evaluated at integration radius = object radius + margin. True excess lipid for a disk of radius $r$
is $\pi r^2 (f_l^{\text{aff,core}} - b_0)$. Across the size series (true fat fraction $f_l = 0.85$,
integrated at a fixed +8 px skirt):

| object radius | true lipid (mm²) | object-extent (naive) | integrated (+8 px) |
|---|---|---|---|
| 4 mm | 46 | 43 (**94 %**) | 54 (118 %) |
| 6 mm | 103 | 108 (105 %) | 121 (118 %) |
| 9 mm | 231 | 247 (107 %) | 272 (117 %) |
| 12 mm | 410 | 435 (106 %) | 455 (111 %) |

The naive object-extent measure loses **6 %** of a 4 mm fat object (and the loss grows as objects
shrink toward the PSF width); the fixed-skirt integrated measure over-counts by 11–18 % because the
skirt is a fixed pixel count, so it captures relatively more background at small radii. Both
estimators also carry a **−10.5 % composition bias** at $f_l = 0.85$ (the affine decode's
linearisation error: $\hat f_l^{\text{aff}} = 0.761$ vs the true 0.85), common to both and separable
from the geometric partial-volume effect. The message of fig 6 is the *shape* — the naive curve
falls away for small objects while the integrated curve stays roughly flat — not the exact
percentage, which depends on the skirt rule.

---

## 6. Validation against truth

All metrics use the **129 held-out test ROIs** (§4), disjoint from the 52 calibration cores. Point
accuracy is the per-voxel decode averaged over each eroded core; agreement is Lin's concordance
correlation coefficient (CCC), which penalises both scatter and any departure from the identity
line (unlike Pearson $r$):

$$
\text{CCC} = \frac{2\,s_{tp}}{s_t^2 + s_p^2 + (\bar t - \bar p)^2} .
$$

| fraction | CCC | slope | RMSE | $R^2$ |
|---|---|---|---|---|
| $f_w$ | **0.996** | 1.009 | 0.023 | 0.991 |
| $f_l$ | **0.997** | 1.014 | 0.019 | 0.994 |
| $f_p$ | **0.998** | 0.995 | 0.005 | 0.997 |

All three exceed the pre-registered $\text{CCC} > 0.9$ target, with slopes within 1.5 % of unity
(no systematic gain error). $f_p$ has the smallest RMSE only because its true range is narrow
($[0, 0.30]$); relative to its range it is the hardest fraction, exactly as the sliver of §3
predicts.

**Detectability.** Can a real composition change be seen above noise? Map each ROI's decode back to
HU via the endpoints and compare against the truth-composition HU; the discrepancy is the noise
floor of the method in HU:

$$
\text{at 150 keV}:\ \text{mean } 0.68\ \mathrm{HU},\ p_{90}=1.3,\ 100\%\ \text{of ROIs} < 5\ \mathrm{HU};
$$
$$
\text{at 70 keV}:\ \text{mean } 0.98\ \mathrm{HU},\ p_{90}=2.0,\ 100\%\ \text{of ROIs} < 5\ \mathrm{HU}.
$$

Every ROI sits under the 5 HU detectability target on both channels — a 5 HU inflammation shift
(the pericoronary-fat use case) would clear this floor. Pooled-ROI $f_l$ CCC equals the per-voxel
$f_l$ CCC to three digits (0.997), confirming the pooling in §5.4 is not what carries the accuracy
here — the surface itself is faithful; pooling is insurance for the map, and essential only for
$f_p$.

---

## 7. Why 70/150 keV

The endpoints, and hence $\operatorname{cond}(G)$, are set by the VMI energies. The `WLP_PAIR` knob
was added to test the intuition that a *wider* energy split should decompose *better*:

| pair | $p_l$ (HU) | $p_p$ (HU) | $\operatorname{cond}(G)$ |
|---|---|---|---|
| 40 / 70 keV | (−212.7, −111.7) | (+204.2, +270.6) | 4.76 |
| **70 / 150 keV** | (−111.7, −81.2) | (+270.6, +290.1) | **16.86** |

The 70/150 pair is **3.5× worse-conditioned** — the wider split pushes both endpoints toward the
Compton plateau where the two energies see nearly the same contrast, flattening the triangle. Naively
that predicts worse fractions. Yet the pooled ROI CCCs are statistically tied between the two pairs
(both ≥ 0.99). The resolution: at the ROI level, $\sqrt N$ averaging over a ~380-voxel core beats the
per-voxel variance down below the accuracy that matters, so conditioning stops being the binding
constraint — the surface fit and the recon fidelity are. 70/150 keV is chosen as the canonical pair
because those VMI energies are what real dual-source and photon-counting systems deliver cleanly (40
keV VMIs are noisier and less standard on deployed scanners), and the conditioning penalty is paid in
per-voxel speckle — which the map's TV step is there to absorb — not in pooled accuracy. The 40 keV
constants are kept in the code for reference but are not the deployed pair.

---

## 8. What the code does *not* do — and where numbers stop transferring

Honesty for publication requires stating the gap between the design plan
(`create-a-new-project-immutable-brooks.md`) and the delivered notebook, because they diverged
during revision.

**The Bayesian MAP in the plan is not the estimator that runs.** The plan specifies a maximum a
posteriori solve — a $\text{Normal}(f_w)\cdot\text{Normal}(f_l)\cdot\text{Gamma}(f_p)$ adipose prior
against a Gaussian data term, minimised by Newton iteration, with a `PRIOR_MODE` switch over
$\{$free, broad, gamma, fwl-closed$\}$. **That solver is scaffolded but never executed.** The
notebook builds the prior object (`bayes_prior_broad`, from the §4 KDE draws) but nothing consumes
it: there is no Newton step, no posterior, no `PRIOR_MODE` branch on the delivered path. What
actually produces every number in this document is the **empirical calibration surface** of §5.1 —
a discriminative least-squares map from HU to fractions — cleaned by the §5.4 Huber-TV. The prior's
only surviving role is *generative*: it draws the physiological calibration compositions (§4). A
reader should treat the WLP estimator as **calibration-based, not Bayesian**; the prior scaffolding
is latent and would need to be wired in and re-validated before any MAP claim could be made.

**The map's regulariser is TV, not a learned prior.** The §5.4 coupling is a hand-set Huber-TV
($\lambda, \varepsilon$ fixed, not fit), so the "prior" on the map is a smoothness assumption, not
the adipose distribution. It is deliberately boundary-agnostic; the ground-truth-pooled comparison
map (figs 3–4) is shown only to quantify what a boundary would buy, not as a deliverable.

**The README is stale.** The committed `README.md` describes an earlier run — 40 keV, ø28 mm rods,
97 ROIs, and an old CCC table. The current pipeline is 70/150 keV, ø19.9 mm inserts, 52 calibration
+ 129 test ROIs, with the metrics of §6. Trust this document and the cached sims over the README.

**Every scanner-specific number is a property of this simulation.** The endpoints are spectrum
averages that this bowtie-free `:dd_fast` EICT geometry produces; the noise levels
($\sigma_{70}\approx9.7$, $\sigma_{150}\approx2.2$) and $\rho = 0.787$ are this acquisition's; the
surface coefficients are fit to them. On a real Canon/Siemens/photon-counting system the *recipe*
transfers — endpoints from co-located rods, three surfaces from known mixtures, σ and ρ from repeat
scans, then decode + TV — but **none of the numbers do**. They must be re-measured per system, as in
the sibling `calibration_comparison.md`.

**Scope of ground truth.** All accuracy claims are *simulation-vs-simulation*: the truth is the
composition painted into the phantom, recovered through a forward+inverse both written here. This
validates the estimator's math and the recon's fidelity (§2.5); it does **not** validate against
physical phantoms or patients, which is future work.

---

## 9. Reproducing every number

The cached simulation outputs are the source of truth for this document:

```
data/wlp_sim_cache_70_150.jls    # 4 calibration sims  -> .calrois (52 cores)
data/wlp_test_cache_70_150.jls   # 5 circular test sims -> .testrois (65)
data/wlp_sect_cache_70_150.jls   # 4 sector test sims  -> .sectrois (64)
data/wlp_int_cache_70_150.jls    # size series          -> .isims (r = 4,6,9,12 mm)
```

Each ROI record carries the eroded-core voxel HU (`v_lo`, `v_hi` — the 70 and 150 keV channels),
the core-mean HU (`m_lo`, `m_hi`), within-core σ (`s_lo`, `s_hi`), the true
fractions (`fw`, `fl`, `fp`), and the label. The `_70_150` caches were written by **Julia 1.12**
(serialization data-version 30); a Julia ≥ 1.12 is required to `deserialize` them (1.11 reads only
up to version 26).

The endpoints and $\operatorname{cond}(G)$ come from NIST XCOM cross-sections via
`XrayAttenuation.jl` **v0.3.2** — `linear_attenuation_coeff(material, E·keV)` is a log-log
interpolation of the per-element total-with-coherent mass-attenuation table, combined by mass
fraction and scaled by density, exactly as §2.1–2.2 use it. Endpoints, all fit coefficients, the
worked decode, the σ_f propagation, the detectability floor, and the integrated-HU series in this
document were regenerated directly from these four caches; the diagnostic line the notebook prints
on load,

```
PAIR 70/150 keV | cal n=52 R²(f_w)=0.996 ρ=0.787 | TEST n=129:
  f_w CCC=0.996 f_l CCC=0.997 f_p CCC=0.998 | cond(G)=16.9
  | integrated 111–118% vs naive 94–107%
```

matches this document field for field.



