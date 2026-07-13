#!/usr/bin/env julia
# Consumer for wlp_model_<pair>.toml (produced ONLY by wlp_decomposition.jl — do not edit
# the constants here; re-export from the notebook). stdlib-only: TOML, LinearAlgebra,
# Statistics, Serialization. Decode math is copied 1:1 from the notebook (this repo's
# convention is self-contained artifacts, not includes).
#
# Apply to new data (co-registered VMI pair from the SAME acquisition chain):
#     M = load_wlp_model("wlp_model_70_150.toml")
#     fw, fl, fp = decode_maps_tv(M, vmi_low, vmi_high)      # 2D slice
#     fw, fl, fp = decode_maps(M, vol_low, vol_high)          # any-dim, per-voxel, no TV
# For a NEW chain/scanner: measure known-fraction ROIs in the new images and
#     cw, cl, cp, cl_aff = fit_surface(hu_low_means, hu_high_means, fw, fl, fp)
# (the calibration_table section of the TOML documents exactly what to measure).
#
# Self-check: `julia apply_wlp_model.jl` reproduces the notebook's held-out CCC from the
# cached test ROIs, standalone — proof the extraction is lossless.

import TOML
using LinearAlgebra, Statistics, Serialization, Printf

# ── decode math, 1:1 with the notebook ──
poly2(h4, h7) = [1.0, h4, h7, h4^2, h7^2, h4 * h7]
surf(c, h4, h7) = dot(c, poly2(h4, h7))
dpoly4(h4, h7) = [0.0, 1.0, 0.0, 2h4, 0.0, h7]
dpoly7(h4, h7) = [0.0, 0.0, 1.0, 0.0, 2h7, h4]
quad_sigma(c, H) = c[1] * H^2 + c[2] * H + c[3]

struct WLPModel
    Elo::Float64; Ehi::Float64
    cw::Vector{Float64}; cl::Vector{Float64}; cp::Vector{Float64}; cl_aff::Vector{Float64}
    sclo::Vector{Float64}; schi::Vector{Float64}; rho::Float64
    Σhu::Matrix{Float64}; bias_hu::Vector{Float64}
    G::Matrix{Float64}; Amet::Matrix{Float64}          # derived from endpoints + Σhu
    hu_lo::Float64; hu_hi::Float64                     # soft-tissue gate on the HIGH channel
end

function load_wlp_model(path::AbstractString)
    t = TOML.parsefile(path)
    ep = t["endpoints_hu"]; PW, PL, PP = ep["water"], ep["lipid"], ep["protein"]
    G = [PL[1]-PW[1] PP[1]-PW[1]; PL[2]-PW[2] PP[2]-PW[2]]
    S = t["noise"]["Sigma_hu"]; Σ = [S[1][1] S[1][2]; S[2][1] S[2][2]]
    p2 = t["poly2"]; nz = t["noise"]; g = t["gate"]
    WLPModel(t["pair"]["E_low_keV"], t["pair"]["E_high_keV"],
        p2["cw"], p2["cl"], p2["cp"], p2["cl_affine"],
        nz["sigma_quad_low"], nz["sigma_quad_high"], nz["rho"],
        Σ, nz["bias_hu"], G, G' * inv(Σ) * G, g["soft_hu_lo"], g["soft_hu_hi"])
end

decode(M::WLPModel, a, b) = (x = surf(M.cw, a, b); y = surf(M.cl, a, b); z = surf(M.cp, a, b);
    s = x + y + z; (x / s, y / s, z / s))
aff_l(M::WLPModel, a, b) = M.cl_aff[1] + M.cl_aff[2] * a + M.cl_aff[3] * b

insimplex(fl, fp) = fl ≥ -1e-9 && fp ≥ -1e-9 && (fl + fp) ≤ 1 + 1e-9
function proj_simplex(M::WLPModel, θ)                  # Mahalanobis-closest simplex point
    insimplex(θ...) && return θ
    V = ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]); best = V[1]; bd = Inf
    for (p, q) in ((V[1], V[2]), (V[1], V[3]), (V[2], V[3]))
        u = q .- p; t = clamp(dot(u, M.Amet * (collect(θ) .- p)) / dot(u, M.Amet * u), 0.0, 1.0)
        c = p .+ t .* u; dv = c .- collect(θ); d = dot(dv, M.Amet * dv); d < bd && (bd = d; best = c)
    end
    (best[1], best[2])
end
function decode_feas(M::WLPModel, a, b)                # ROI means: surface, else MLE projection
    r = (surf(M.cw, a, b), surf(M.cl, a, b), surf(M.cp, a, b))
    all(r .≥ -1e-9) && (s = sum(r); return (r[1] / s, r[2] / s, r[3] / s))
    θ = M.G \ ([a, b] .- M.bias_hu); (fl, fp) = proj_simplex(M, (θ[1], θ[2])); (1 - fl - fp, fl, fp)
end

function decode_maps(M::WLPModel, vlo, vhi)            # per-voxel, any-dim, NaN outside gate
    @assert size(vlo) == size(vhi)
    fw = fill(NaN, size(vhi)); fl = copy(fw); fp = copy(fw)
    for I in CartesianIndices(vhi)
        (M.hu_lo < vhi[I] < M.hu_hi) || continue
        d = decode(M, Float64(vlo[I]), Float64(vhi[I])); fw[I] = d[1]; fl[I] = d[2]; fp[I] = d[3]
    end
    (fw, fl, fp)
end

function sigma_f_weight(M::WLPModel, hlo, hhi)
    w = fill(NaN, size(hhi))
    for I in CartesianIndices(hhi)
        (M.hu_lo < hhi[I] < M.hu_hi) || continue
        h4 = Float64(hlo[I]); h7 = Float64(hhi[I])
        s4 = quad_sigma(M.sclo, h4); s7 = quad_sigma(M.schi, h7)
        g4l = dot(M.cl, dpoly4(h4, h7)); g7l = dot(M.cl, dpoly7(h4, h7))
        g4p = dot(M.cp, dpoly4(h4, h7)); g7p = dot(M.cp, dpoly7(h4, h7))
        vl = g4l^2 * s4^2 + g7l^2 * s7^2 + 2M.rho * g4l * g7l * s4 * s7
        vp = g4p^2 * s4^2 + g7p^2 * s7^2 + 2M.rho * g4p * g7p * s4 * s7
        w[I] = 1.0 / max(vl + vp, 1e-6)
    end
    w
end

# coupled edge-preserving Huber-TV on (f_l,f_p); w = σ_f data weight. Never Gaussian.
function tv_coupled(yl, yp, mask; lambda=0.05, iters=25, eps=0.04, w=nothing)
    nx, ny = size(yl)
    fl = [mask[i, j] ? Float64(yl[i, j]) : 0.0 for i in 1:nx, j in 1:ny]
    fp = [mask[i, j] ? Float64(yp[i, j]) : 0.0 for i in 1:nx, j in 1:ny]
    fl2 = copy(fl); fp2 = copy(fp); inb(i, j) = 1 ≤ i ≤ nx && 1 ≤ j ≤ ny && mask[i, j]
    smp(a, b) = (a = max(a, 0.0); b = max(b, 0.0); s = a + b; s > 1 ? (a / s, b / s) : (a, b))
    for _ in 1:iters
        @inbounds for j in 1:ny, i in 1:nx
            mask[i, j] || (fl2[i, j] = fl[i, j]; fp2[i, j] = fp[i, j]; continue)
            wij = (w === nothing || !isfinite(w[i, j])) ? 1.0 : w[i, j]
            rl = wij * Float64(yl[i, j]); rp = wij * Float64(yp[i, j]); den = wij
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                inb(i + di, j + dj) || continue
                dl = fl[i+di, j+dj] - fl[i, j]; dp = fp[i+di, j+dj] - fp[i, j]
                c = lambda / max(sqrt(dl^2 + dp^2), eps)
                rl += c * fl[i+di, j+dj]; rp += c * fp[i+di, j+dj]; den += c
            end
            fl2[i, j], fp2[i, j] = smp(rl / den, rp / den)
        end
        fl, fl2 = fl2, fl; fp, fp2 = fp2, fp
    end
    ([mask[i, j] ? fl[i, j] : NaN for i in 1:nx, j in 1:ny],
     [mask[i, j] ? fp[i, j] : NaN for i in 1:nx, j in 1:ny])
end

# the notebook's delivered-map pipeline: per-voxel decode + σ_f-weighted Huber-TV (2D)
function decode_maps_tv(M::WLPModel, mlo, mhi; lambda=0.05, iters=25, eps=0.04)
    fw, fl, fp = decode_maps(M, mlo, mhi)
    gate = .!isnan.(fl); w = sigma_f_weight(M, mlo, mhi)
    fl_tv, fp_tv = tv_coupled(fl, fp, gate; lambda, iters, eps, w)
    fw_tv = map((a, b) -> isnan(a) ? NaN : 1 - a - b, fl_tv, fp_tv)
    (fw_tv, fl_tv, fp_tv)
end

# ── refit for a NEW chain: same recipe as the notebook's calibration cell ──
function fit_surface(m_lo, m_hi, fw, fl, fp)
    X = reduce(vcat, [poly2(m_lo[i], m_hi[i])' for i in eachindex(m_lo)])
    Xa = hcat(ones(length(m_lo)), m_lo, m_hi)
    (cw = X \ fw, cl = X \ fl, cp = X \ fp, cl_aff = Xa \ fl)
end
fit_sigma_quad(hu, sig) = (X = hcat(hu .^ 2, hu, ones(length(hu))); c = X \ sig;
    c[1] < 0 && (Xa = hcat(hu, ones(length(hu))); ca = Xa \ sig; c = [0.0, ca[1], ca[2]]); c)

# ── self-check: reproduce the notebook's held-out accuracy from the caches ──
function metrics(t, r)
    mt, mr = mean(t), mean(r); st2 = mean((t .- mt) .^ 2); str = mean((t .- mt) .* (r .- mr))
    sr2 = mean((r .- mr) .^ 2)
    (ccc = 2str / (st2 + sr2 + (mt - mr)^2), rmse = sqrt(mean((r .- t) .^ 2)))
end

function selfcheck(dir=@__DIR__; tag="70_150")
    path = joinpath(dir, "wlp_model_$(tag).toml")
    M = load_wlp_model(path); prov = TOML.parsefile(path)["provenance"]
    rois = vcat(deserialize(joinpath(dir, "wlp_test_cache_$(tag).jls")).testrois,
                deserialize(joinpath(dir, "wlp_sect_cache_$(tag).jls")).sectrois)
    rec = map(rois) do r                                # the notebook's headline arithmetic:
        ps = [decode(M, r.v40[j], r.v70[j]) for j in eachindex(r.v40)]  # per-voxel decode,
        (mean(getindex.(ps, 1)), mean(getindex.(ps, 2)), mean(getindex.(ps, 3)))  # core mean
    end
    names = ("f_w", "f_l", "f_p")
    cccs = ntuple(k -> metrics([getfield(r, (:fw, :fl, :fp)[k]) for r in rois], getindex.(rec, k)).ccc, 3)
    for k in 1:3
        @printf("%s  CCC %.6f  (notebook: %.6f)\n", names[k], cccs[k], prov["test_ccc"][k])
    end
    Δ = maximum(abs.(collect(cccs) .- prov["test_ccc"]))
    @assert Δ < 1e-9 "extracted model does not reproduce the notebook (max ΔCCC = $Δ)"
    # smoke the array path on the cached delivered-map slice
    sim = joinpath(dir, "wlp_sim_cache_$(tag).jls")
    if isfile(sim)
        D = deserialize(sim); mid = size(D.map70v, 3) ÷ 2 + 1
        fw, fl, fp = decode_maps_tv(M, D.map40v[:, :, mid], D.map70v[:, :, mid])
        @printf("map smoke: %d gated voxels, f_l ∈ [%.2f, %.2f]\n",
            count(!isnan, fl), minimum(filter(!isnan, fl)), maximum(filter(!isnan, fl)))
    end
    println("SELF-CHECK PASS — extraction is lossless.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    selfcheck()
end
