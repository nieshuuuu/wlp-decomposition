# Design the phantom in the MEASUREMENT domain, not the tissue domain.
#
# The error being fixed: oxford_wlp_composition.csv is derived from CLINICAL HU, which is already
# blurred by the clinical scanner's point spread. Assigning those fractions as the phantom's
# TRUTH and then simulating a CT blurs a second time, so the simulated measurement necessarily
# sits below the clinical curve. Blurred twice.
#
# The fix: estimate the point spread from this very simulation, deconvolve the Oxford radial
# profile to recover a tissue-domain profile, and assign THAT to the phantom. Re-simulating then
# blurs exactly once and should land back on the clinical curve — which is the testable claim.
#
# ORDER OF OPERATIONS. HU is EXACTLY barycentric in the volume fractions — mu is volume-additive
# and water is the HU zero — so
#     HU = f_l*HU_l + f_p*HU_p,     f_w + f_l + f_p = 1
# and one 120 kVp measurement fixes a LINE SEGMENT in the composition triangle, not a point. Single
# energy under-determines three materials by exactly one degree of freedom, and the Woodard-1986
# adipose prior is what supplies it. The pipeline therefore has two operations, in this order:
#
#     (a) clinical HU -> tissue HU     a fit, entirely in HU space; composition never appears
#     (b) tissue  HU -> composition    the prior, applied ONCE, in the domain we actually believe
#
# (a) runs FIRST precisely because it needs no composition. Nothing is ever computed in the clinical
# composition domain and nothing is transported between domains, so the only approximations left are
# the prior itself and the smooth-profile assumption of section 3 — both physical statements about
# adipose tissue. Derivation: docs/pcat_deconv_design_math.md
using Statistics: mean, median
using Printf: @printf, @sprintf
using DelimitedFiles: readdlm
using Unitful: @u_str
import Unitful
import BasisSimulator as BS
import CairoMakie as CM

const OUT = joinpath(@__DIR__, "pcat_ct")

# ── 0. the adipose prior, included from its canonical home ───────────────────────────
# SSoT: the sampler lives in wl-noise-aware-mmd and is INCLUDED here, never copied. The seed matches
# that repo's examples/oxford_wlp_composition.jl so both draw the identical cloud.
const WLNAM = normpath(joinpath(@__DIR__, "..", "wl-noise-aware-mmd"))
isdir(WLNAM) ||
    error("adipose prior unavailable: expected wl-noise-aware-mmd beside this repo, at $WLNAM")
for f in ("wl_mixture.jl", "wlp_mixture.jl", "wlp_priors.jl", "wlp_adipose_sampler.jl")
    include(joinpath(WLNAM, "src", f))
end

# Endpoints are COMPUTED, not pasted. E_eff = 70.0 keV is the effective energy of the 120 kVp
# closure this HU curve was measured through (wl-noise-aware-mmd data/analysis/oxford_closure.toml).
const E_EFF = 70.0
mu(m, E) = Float64(Unitful.ustrip(u"cm^-1", BS.XA.linear_attenuation_coeff(m, E * u"keV")))
hu_of(m, E) = 1000.0 * (mu(m, E) - mu(_WL_WATER, E)) / mu(_WL_WATER, E)
const HU_L, HU_P = hu_of(_WL_LIPID, E_EFF), hu_of(_WLP_PROT, E_EFF)

const PRIOR = adipose_sample_comps(200_000; seed = 20260727)
const P_FW = Float64[c.f_w for c in PRIOR]
const P_FL = Float64[c.f_l for c in PRIOR]
const P_FP = Float64[c.f_p for c in PRIOR]
const P_HU = P_FL .* HU_L .+ P_FP .* HU_P

"""
    adipose_posterior(hu, σ) -> (f = (f_w,f_l,f_p), sd, ess, hu_back)

Importance-weight the adipose prior by a Gaussian HU likelihood centred on `hu` with width `σ`, and
return the posterior mean composition. `hu_back` is that mean's own barycentric HU: a weighted mean
of compositions need NOT be exactly HU-consistent, and the gap is reported rather than assumed away.
"""
function adipose_posterior(hu, σ)
    lw = -0.5 .* ((P_HU .- hu) ./ σ) .^ 2
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    m(x) = sum(w .* x)
    s(x) = sqrt(max(0.0, sum(w .* (x .- m(x)) .^ 2)))     # PRIOR-conditional spread, not measurement error
    f = (m(P_FW), m(P_FL), m(P_FP))
    sd = (s(P_FW), s(P_FL), s(P_FP))
    # This becomes the phantom's GROUND TRUTH (simulated materials + every accuracy score's x-axis),
    # so an unphysical triple must fail here rather than propagate silently.
    (all(0.0 .<= f .<= 1.0) && sum(f) ≈ 1.0) ||
        error("adipose_posterior: hu=$hu sigma=$σ produced a non-composition $f")
    (f = f, sd = sd, ess = 1 / sum(w .^ 2), hu_back = f[2] * HU_L + f[3] * HU_P)
end

@printf("adipose prior: %d draws; endpoints at %.1f keV are HU_l = %.3f, HU_p = %.3f\n",
        length(P_FW), E_EFF, HU_L, HU_P)

# ── 1. the clinical HU gradient ───────────────────────────────────────────────────────
# SSoT is the gradient file itself, not a composition derived from it. Note this also means the
# script no longer needs the SMB share mounted.
const GRAD = joinpath(WLNAM, "data", "oxford_fai_gradient.csv")
oxford = Dict{String, Vector{NTuple{3,Float64}}}()   # group -> [(d, hu, standard error of the mean)]
let hdr = nothing
    for ln in eachline(GRAD)
        (startswith(strip(ln), "#") || isempty(strip(ln))) && continue
        f = strip.(split(strip(ln), ','))
        if hdr === nothing; hdr = f; continue; end
        ix(n) = (k = findfirst(==(n), hdr); k === nothing && error("$GRAD: no column '$n'"); k)
        d = parse(Float64, f[ix("distance_mm")])
        for (g, hc, sc) in (("healthy",  "healthy_hu",  "healthy_hu_standard_error_of_the_mean"),
                            ("diseased", "diseased_hu", "diseased_hu_standard_error_of_the_mean"))
            push!(get!(oxford, g, NTuple{3,Float64}[]),
                  (d, parse(Float64, f[ix(hc)]), parse(Float64, f[ix(sc)])))
        end
    end
end
for g in keys(oxford); sort!(oxford[g], by = x -> x[1]); end

# ── 2. point spread, estimated from this simulation's own radial profile ──────────────
# measured(r) = (truth ⊛ G_σ)(r). Fit σ on the vessels with clean, well-populated profiles.
prof = readdlm(joinpath(OUT, "pcat_radial_profile.csv"), ','; header = true)
rows, hdr = prof[1], vec(prof[2])
ci(n) = findfirst(==(n), hdr)
function vessel_profile(v)
    sel = [i for i in axes(rows,1) if rows[i, ci("vessel")] == v]
    (r = Float64[rows[i, ci("r_mm")] for i in sel],
     m = Float64[rows[i, ci("measured_HU70")] for i in sel],
     t = Float64[rows[i, ci("truth_HU70")] for i in sel])
end
function blur1d(y, r, σ)
    n = length(y); out = similar(y)
    for i in 1:n
        num = 0.0; den = 0.0
        for j in 1:n
            w = exp(-(r[i]-r[j])^2 / (2σ^2)); num += w*y[j]; den += w
        end
        out[i] = num/den
    end
    out
end
FITV = ["rca1", "lcx"]                      # thick cuffs, full radial span, no lung on the outside
function psf_cost(σ)
    c = 0.0; n = 0
    for v in FITV
        p = vessel_profile(v); isempty(p.r) && continue
        b = blur1d(p.t, p.r, σ)
        keep = [i for i in eachindex(p.r) if -1.0 <= p.r[i] <= 6.0]
        c += sum((b[keep] .- p.m[keep]).^2); n += length(keep)
    end
    c / max(n,1)
end
σs = 0.05:0.025:2.5
costs = [psf_cost(σ) for σ in σs]
σ̂ = σs[argmin(costs)]
fwhm = 2.3548 * σ̂
@printf("\npoint spread fitted on %s: sigma = %.3f mm, FWHM = %.3f mm (residual %.1f HU rms)\n",
        join(FITV, "+"), σ̂, fwhm, sqrt(minimum(costs)))

# ── 3. recover the tissue profile by FORWARD MODELLING, not deconvolution ────────────
# Direct deconvolution rings: the Oxford profile is sampled at 1 mm but the point spread is
# 2.47 mm FWHM, so recovering that band means amplifying the CSV's own measurement noise (its
# stated standard error of the mean is 1.0-1.75 HU). Van Cittert at 60 iterations produced
# +7.1/-10.0/+7.6/-6.3/+8.5/-8.7 HU alternating every millimetre — an artefact, not physiology.
#
# Instead: assume the tissue-domain gradient is smooth and monotone (it is a diffusion-like
# lipid gradient away from the vessel), parametrise it with three numbers, blur it, and fit the
# BLURRED model to the clinical curve. No inverse filter, no ringing, and the residual reports
# honestly whether three parameters were enough.
#     tissue(r) = A + B*exp(-r/tau)      r > 0
#     tissue(r) = HU_wall                r <= 0   (the wall the blur mixes in at small r)
# NO WALL TERM in the forward model. The Oxford profile is gated to [-190,-30] HU, so voxels
# contaminated by the +43 HU vessel wall read above -30 and were already excluded from the
# clinical measurement being fitted. Including the wall in the blur made the re-blurred check
# overshoot the clinical curve by ~18 HU at r = 1-2 mm. The fat profile is simply continued
# inward instead, which is what the gated measurement effectively averages.
function tissue_model(r, p)
    A, B, τ = p
    [A + B*exp(-max(ri, 0.0)/τ) for ri in r]
end
const TAU_GRID = 0.5:0.5:120.0
function fit_tissue(d, hu, sem, σ)
    # A and B enter LINEARLY and blur1d is row-normalised, so blur(A + B*e) = A + B*blur(e): only τ
    # needs the O(n^2) blur, and (A,B) fall out of a 2x2 weighted least squares in closed form.
    # That deletes their grids, and with them the boundary bug that pinned the diseased fit at
    # A = -110.0 — the first value of the old `A in -110.0:1.0:-60.0`, i.e. an edge, not a minimum.
    rf = collect(-6.0:0.1:30.0)      # extends inside the wall, so the blur sees the real edge
    j = [argmin(abs.(rf .- di)) for di in d]
    w = 1 ./ sem .^ 2                # the CSV's own standard error of the mean, per layer
    best = nothing; bestc = Inf
    for τ in TAU_GRID
        u = blur1d([exp(-max(ri, 0.0)/τ) for ri in rf], rf, σ)[j]
        s0, su, suu = sum(w), sum(w .* u), sum(w .* u .^ 2)
        sy, suy = sum(w .* hu), sum(w .* u .* hu)
        det = s0 * suu - su^2
        abs(det) < 1e-12 && continue                      # u degenerate: this τ carries no shape
        A = (suu * sy - su * suy) / det; B = (s0 * suy - su * sy) / det
        c = sum(w .* (A .+ B .* u .- hu) .^ 2)
        c < bestc && (bestc = c; best = (A, B, τ))
    end
    # τ is the only gridded parameter left. A fit sitting on either end is a boundary solution, not
    # a minimum, and it must not reach the phantom silently — that is exactly how the old bug hid.
    best === nothing && error("fit_tissue: no usable τ in TAU_GRID")
    first(TAU_GRID) < best[3] < last(TAU_GRID) ||
        error("fit_tissue: τ pinned at the grid edge ($(best[3]) mm) — widen TAU_GRID")
    # the defining property of the least-squares solution: the weighted residual is orthogonal to
    # both columns of the design matrix [1, u]. If the closed form above is wrong, this is what says so.
    let u = blur1d([exp(-max(ri, 0.0)/best[3]) for ri in rf], rf, σ)[j],
        res = w .* (best[1] .+ best[2] .* u .- hu)
        scale = sum(w) * maximum(abs, hu)
        (abs(sum(res)) < 1e-8 * scale && abs(sum(res .* u)) < 1e-8 * scale) ||
            error("fit_tissue: normal equations not satisfied — closed form is wrong")
    end
    # return the RAW χ², not a display statistic — reduced χ² needs the caller's ν and deriving it
    # here would fix a convention the caller cannot see.
    (best, bestc)
end
const NPAR = 3                                   # A, B, τ — the degrees of freedom the fit spends

println("\ntissue-domain profile recovered by forward-model fit (what the phantom should contain):")
@printf("%-9s %5s %10s %12s %8s %10s %10s %10s %9s\n",
        "group","d_mm","HU_clin","HU_tissue","Δ_HU","f_w","f_l","f_p","prior ess")
newcomp = Dict{Tuple{String,Int}, NTuple{3,Float64}}()
newsd   = Dict{Tuple{String,Int}, NTuple{3,Float64}}()   # prior-conditional spread, for the bands
fitpars = Dict{String, NTuple{3,Float64}}()   # the figure replots THESE; never refit it separately
fitrms = Dict{String, Float64}()              # residual standard deviation, annotated on the figure
hu_gaps = Float64[]                           # |posterior mean's own HU - the HU it was given|
for g in ("healthy","diseased")
    p = oxford[g]
    d = Float64[x[1] for x in p]; hu = Float64[x[2] for x in p]; sem = Float64[x[3] for x in p]
    (pars, chi2) = fit_tissue(d, hu, sem, σ̂)
    tis = tissue_model(d, pars)
    fitpars[g] = pars
    # Goodness of fit is reported as the residual standard deviation in HU, NOT as reduced
    # chi-square. All 20 layers come from the SAME cohort and share the scanner's blur, so their
    # errors are correlated — and the model's free offset A absorbs the common mode outright. A
    # DIAGONAL chi²/ν therefore reads far below 1 (0.26 healthy, 0.69 diseased) for that reason
    # alone and is not a goodness-of-fit number here. The rms residual assumes no independence.
    # Full argument, with the three hypotheses that were tested: docs/pcat_deconv_design_math.md §9.
    fitrms[g] = sqrt(sum((hu .- tis) .^ 2) / (length(d) - NPAR))
    @printf("  %-9s fit A=%.1f B=%.1f tau=%.2f mm  (chi-square %.2f, residual %.2f HU rms on %d dof)\n",
            g, pars..., chi2, fitrms[g], length(d) - NPAR)
    for i in eachindex(d)
        # The prior runs HERE and only here — on the tissue HU, the value we actually believe.
        po = adipose_posterior(tis[i], sem[i])
        newcomp[(g, Int(d[i]))] = po.f
        newsd[(g, Int(d[i]))] = po.sd
        push!(hu_gaps, abs(po.hu_back - tis[i]))
        Int(d[i]) <= 6 && @printf("%-9s %5d %10.2f %12.2f %+8.2f %10.4f %10.4f %10.4f %9.0f\n",
                                  g, Int(d[i]), hu[i], tis[i], tis[i]-hu[i], po.f..., po.ess)
    end
end
@printf("\nposterior means are HU-consistent to %.3f HU (a weighted mean of compositions need not be)\n",
        maximum(hu_gaps))

open(joinpath(OUT, "oxford_deconvolved_composition.csv"), "w") do io
    println(io, "# Oxford radial profile inverted to the TISSUE domain by forward-model fit")
    println(io, "# (sigma = $(round(σ̂,digits=3)) mm, FWHM = $(round(fwhm,digits=3)) mm), so that")
    println(io, "# re-simulating blurs ONCE and the measurement returns to the clinical curve.")
    println(io, "# Composition = Woodard-1986 adipose prior applied ONCE, to the TISSUE HU (200k draws,")
    println(io, "# importance-weighted by a Gaussian likelihood at that layer's standard error of the mean).")
    println(io, "# No composition is computed in the clinical domain. See pcat_deconv_design.jl")
    println(io, "# The *_posterior_standard_deviation columns are the PRIOR-conditional spread at that")
    println(io, "# layer's HU, not a measurement error. Nothing downstream reads them; they are the bands.")
    println(io, join(("group", "distance_mm",
                      "water_volume_fraction", "lipid_volume_fraction", "protein_volume_fraction",
                      "water_volume_fraction_posterior_standard_deviation",
                      "lipid_volume_fraction_posterior_standard_deviation",
                      "protein_volume_fraction_posterior_standard_deviation"), ","))
    for g in ("healthy","diseased"), d in 1:20
        println(io, "$g,$d," * join(vcat(collect(newcomp[(g,d)]), collect(newsd[(g,d)])), ","))
    end
end
println("\nwrote ", joinpath(OUT, "oxford_deconvolved_composition.csv"))

# ── figure ────────────────────────────────────────────────────────────────────────────
fig = CM.Figure(size = (1420, 560))
ax1 = CM.Axis(fig[1,1]; title = "point spread fit", titlesize = 16,
    xlabel = "Gaussian sigma (mm)", ylabel = "mean squared residual (HU²)")
CM.lines!(ax1, collect(σs), costs; linewidth = 2.5, color = :black)
CM.vlines!(ax1, σ̂; color = :red, linestyle = :dash)
CM.text!(ax1, σ̂ + 0.05, maximum(costs)*0.85;
    text = "sigma = $(round(σ̂,digits=3)) mm\nFWHM = $(round(fwhm,digits=2)) mm", fontsize = 13)
ax2 = CM.Axis(fig[1,2]; title = "Oxford profile: clinical vs exponential decay fitting",
    titlesize = 16, xlabel = "radial distance from the wall (mm)", ylabel = "HU")
const RF = collect(-6.0:0.1:30.0)
for (g, col) in (("healthy", CM.RGBf(.20,.45,.80)), ("diseased", CM.RGBf(.85,.20,.18)))
    p = oxford[g]; d = Float64[x[1] for x in p]; hu = Float64[x[2] for x in p]
    pars = fitpars[g]
    CM.lines!(ax2, d, hu; color = col, linewidth = 2.5, label = "$g — clinical")
    CM.lines!(ax2, d, tissue_model(d, pars); color = col, linewidth = 2,
              linestyle = :dash, label = "$g — exponential decay fitting")
    # The re-blurred curve is no longer drawn: it lands on the dashed line to within 0.03 HU, so it
    # was ink, not information. The closure claim it carried is kept as a printed number instead.
    bf = blur1d(tissue_model(RF, pars), RF, σ̂)
    rb = [bf[argmin(abs.(RF .- di))] for di in d]
    @printf("re-blur closure %-9s max|re-blurred - clinical| = %.2f HU, max|re-blurred - fit| = %.3f HU\n",
            g, maximum(abs.(rb .- hu)), maximum(abs.(rb .- tissue_model(d, pars))))
end
# legend goes top-right: past d = 13 mm every curve is below -81 HU, so that corner is empty.
CM.axislegend(ax2; position = :rt, labelsize = 11, framevisible = true,
              backgroundcolor = (:white, 0.85), padding = (8, 8, 6, 6))
# bottom-left is the other empty corner — the method and its goodness of fit go there
CM.text!(ax2, 1.3, -84.0;
    text = "chi-square fit of an exponential decay\n" *
           "T(r) = A + B exp(-r/tau)\n" *
           @sprintf("healthy    tau = %.1f mm,  residual %.2f HU rms",
                    fitpars["healthy"][3], fitrms["healthy"]) * "\n" *
           @sprintf("diseased   tau = %.1f mm,  residual %.2f HU rms",
                    fitpars["diseased"][3], fitrms["diseased"]),
    fontsize = 11, align = (:left, :top))
CM.save(joinpath(OUT, "pcat_deconv_design.png"), fig;
        px_per_unit = min(2.0, 2000 / maximum(fig.scene.viewport[].widths)))
println("figure -> ", joinpath(OUT, "pcat_deconv_design.png"))

# ── figure 2: the composition the phantom actually gets ───────────────────────────────
# The clinical-domain twin of this figure lives upstream (wl-noise-aware-mmd assets/
# oxford_wlp_profiles.png) and describes the Oxford cohort. THIS one describes the phantom.
let cH = CM.RGBf(.20,.45,.75), cD = CM.RGBf(.78,.22,.20), D = collect(1.0:20.0)
    frac(g, k) = [newcomp[(g, Int(d))][k] for d in D]
    spread(g, k) = [newsd[(g, Int(d))][k] for d in D]
    f2 = CM.Figure(size = (1320, 900))
    for (pos, k, ylab, ttl) in (((1,1), 1, "water volume fraction  f_w",   "(a) water — diseased is wetter at the wall"),
                                ((1,2), 2, "lipid volume fraction  f_l",   "(b) lipid — the mirror of (a)"),
                                ((2,1), 3, "protein volume fraction  f_p", "(c) protein — PRIOR-DRIVEN, band is prior spread"))
        ax = CM.Axis(f2[pos...]; title = ttl, xlabel = "distance from the vessel wall  d (mm)",
                     ylabel = ylab, titlesize = 13, xlabelsize = 11, ylabelsize = 11)
        for (g, c) in (("healthy", cH), ("diseased", cD))
            y = frac(g, k); e = spread(g, k)
            CM.band!(ax, D, y .- e, y .+ e; color = (c, 0.18))
            CM.scatterlines!(ax, D, y; color = c, markersize = 6, linewidth = 2, label = g)
        end
        CM.axislegend(ax; position = k == 2 ? :rb : :rt, labelsize = 10, framevisible = false)
    end
    ax4 = CM.Axis(f2[2,2]; title = "(d) Δ = diseased − healthy — the FAI signal, converging with distance",
                  xlabel = "distance from the vessel wall  d (mm)", ylabel = "Δ volume fraction",
                  titlesize = 13, xlabelsize = 11, ylabelsize = 11)
    CM.hlines!(ax4, [0.0]; color = (:black, 0.4), linestyle = :dash)
    for (k, c, lab) in ((1, CM.RGBf(.15,.45,.70), "Δf_w"), (2, CM.RGBf(.85,.45,.10), "Δf_l"),
                        (3, CM.RGBf(.30,.60,.30), "Δf_p"))
        CM.scatterlines!(ax4, D, frac("diseased", k) .- frac("healthy", k);
                         color = c, markersize = 6, linewidth = 2, label = lab)
    end
    CM.axislegend(ax4; position = :rb, labelsize = 10, framevisible = false)
    CM.Label(f2[0, :], "PCAT phantom composition — TISSUE domain, adipose prior applied once at the " *
             @sprintf("fitted HU (E_eff = %.1f keV endpoints)", E_EFF); fontsize = 14, font = :bold)
    CM.save(joinpath(OUT, "pcat_composition_profiles.png"), f2;
            px_per_unit = min(2.0, 2000 / maximum(f2.scene.viewport[].widths)))
    println("figure -> ", joinpath(OUT, "pcat_composition_profiles.png"))
end
