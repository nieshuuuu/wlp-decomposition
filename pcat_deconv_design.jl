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
# Closure for HU -> fractions. The Oxford model is one equation in two free fractions:
#     HU = f_l*(-111.69) + f_p*(270.58),   f_w + f_l + f_p = 1
# closed in the CSV by a Woodard&White adipose prior. That sampler is not reproduced here.
# Instead the protein-to-water RATIO from each CSV row is held fixed and HU is re-solved:
#     f_w = (HU + 111.69) / (111.69*(1+r) + 270.58*r),  f_p = r*f_w,  f_l = 1 - f_w - f_p
# This reproduces the CSV exactly when HU is unchanged (verified below), so it perturbs along
# the prior's own locus rather than inventing a new one.
using Statistics: mean, median
using Printf: @printf
using DelimitedFiles: readdlm
import CairoMakie as CM

const OUT = joinpath(@__DIR__, "pcat_ct")
const CSV = "/Volumes/Molloilab/Shu Nie/water-lipid-protein/oxford_wlp_composition.csv"
const HU_L, HU_P = -111.69, 270.58        # lipid / protein endpoints of the Oxford single-energy model

# ── 1. Oxford profile ─────────────────────────────────────────────────────────────────
oxford = Dict{String, Vector{NTuple{6,Float64}}}()   # group -> [(d, hu, fw, fl, fp, sem)]
let hdr = nothing
    for ln in eachline(CSV)
        startswith(ln, "#") && continue
        f = split(strip(ln), ',')
        if hdr === nothing; hdr = f; continue; end
        ix(n) = findfirst(==(n), hdr)
        g = f[ix("group")]
        push!(get!(oxford, g, NTuple{6,Float64}[]),
              (parse(Float64,f[ix("distance_mm")]), parse(Float64,f[ix("hu")]),
               parse(Float64,f[ix("water_volume_fraction")]),
               parse(Float64,f[ix("lipid_volume_fraction")]),
               parse(Float64,f[ix("protein_volume_fraction")]),
               parse(Float64,f[ix("hu_standard_error_of_the_mean")])))
    end
end
for g in keys(oxford); sort!(oxford[g], by = x -> x[1]); end

hu_to_frac(hu, ratio) = begin
    fw = (hu - HU_L) / (-HU_L * (1 + ratio) + HU_P * ratio)
    fp = ratio * fw
    f = (fw, 1 - fw - fp, fp)
    # These become the phantom's GROUND TRUTH (simulated materials + every accuracy score's
    # x-axis), so an out-of-range fraction must fail here, not propagate silently — today the
    # fit-grid bounds keep hu in range, but that is an accident of the grid, not an invariant.
    all(0.0 .<= f .<= 1.0) || error("hu_to_frac out of [0,1]: hu=$hu ratio=$ratio -> $f")
    f
end

println("closure check — re-solving the CSV's own HU must return the CSV's own fractions:")
for g in ("healthy","diseased"), d in (1, 10, 20)
    row = oxford[g][d]
    r = row[5] / row[3]
    f = hu_to_frac(row[2], r)
    @printf("  %-9s %2dmm  CSV (%.4f, %.4f, %.4f)  resolved (%.4f, %.4f, %.4f)  Δ=%.5f\n",
            g, d, row[3], row[4], row[5], f..., maximum(abs.(f .- (row[3],row[4],row[5]))))
end

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
    (best, sqrt(bestc/length(d)))
end

println("\ntissue-domain profile recovered by forward-model fit (what the phantom should contain):")
@printf("%-9s %5s %10s %12s %8s %10s %10s %10s\n",
        "group","d_mm","HU_clin","HU_tissue","Δ_HU","f_w","f_l","f_p")
newcomp = Dict{Tuple{String,Int}, NTuple{3,Float64}}()
fitpars = Dict{String, NTuple{3,Float64}}()   # the figure replots THESE; never refit it separately
for g in ("healthy","diseased")
    p = oxford[g]
    d = Float64[x[1] for x in p]; hu = Float64[x[2] for x in p]
    ratio = Float64[x[5]/x[3] for x in p]
    sem = Float64[x[6] for x in p]
    (pars, resid) = fit_tissue(d, hu, sem, σ̂)
    fitpars[g] = pars
    @printf("  %-9s fit A=%.1f B=%.1f tau=%.2f mm  (weighted residual %.2f sigma)\n",
            g, pars..., resid)
    tis = tissue_model(d, pars)
    for i in eachindex(d)
        f = hu_to_frac(tis[i], ratio[i])
        newcomp[(g, Int(d[i]))] = f
        Int(d[i]) <= 6 && @printf("%-9s %5d %10.2f %12.2f %+8.2f %10.4f %10.4f %10.4f\n",
                                  g, Int(d[i]), hu[i], tis[i], tis[i]-hu[i], f...)
    end
end

open(joinpath(OUT, "oxford_deconvolved_composition.csv"), "w") do io
    println(io, "# Oxford radial profile inverted to the TISSUE domain by forward-model fit")
    println(io, "# (sigma = $(round(σ̂,digits=3)) mm, FWHM = $(round(fwhm,digits=3)) mm), so that")
    println(io, "# re-simulating blurs ONCE and the measurement returns to the clinical curve.")
    println(io, "# Closure: protein/water ratio held at the CSV row's own value; see pcat_deconv_design.jl")
    println(io, "group,distance_mm,water_volume_fraction,lipid_volume_fraction,protein_volume_fraction")
    for g in ("healthy","diseased"), d in 1:20
        f = newcomp[(g,d)]
        println(io, "$g,$d,$(f[1]),$(f[2]),$(f[3])")
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
CM.Legend(fig[2, 1:2], ax2; orientation = :horizontal, framevisible = false, labelsize = 12)
CM.save(joinpath(OUT, "pcat_deconv_design.png"), fig;
        px_per_unit = min(2.0, 2000 / maximum(fig.scene.viewport[].widths)))
println("figure -> ", joinpath(OUT, "pcat_deconv_design.png"))
