# Apply THIS repo's 3-material (water/lipid/protein) model to REAL 70/150 keV VMI CT and compare,
# at identical anchors and identical gate, against the 2-material (water/lipid) GLS baseline.
#   (1) human CCTA 57955439  — pericardium/fibrous, inflamed fat, lipid gradient
#   (2) Hamid QRM/Gammex phantom — iodine/bone/lung rejection + fat-body image quality (σ)
#
# 3-material : wlp-decomposition/wlp_model_70_150.toml via apply_wlp_model.jl (THIS repo).
# 2-material : wl-noise-aware-mmd/src/wl_decompose.jl decompose_volume (frozen GLS). No wlp method
#              from wl-noise-aware-mmd is used.
#
# ESTIMATOR (skill-mandated tissue prior): decode → σ_f-weighted coupled Huber-TV → L1 protein
# sparsity (fp admitted only where the data demands it) → HU-consistent lipid refit. Where fp→0 the
# estimator reduces EXACTLY to the 2-material GLS, so clean water/lipid keeps 2-material image
# quality; protein appears only in genuinely off-line (fibrous/pericardial) tissue.
#
#   julia --project=. apply_to_real_ct.jl

using LinearAlgebra, Statistics, Printf
import CairoMakie as CM

const WLP = @__DIR__
include(joinpath(WLP, "apply_wlp_model.jl"))                 # 3-material consumer (decode, tv_coupled, σ_f)
const MMD = "/Users/shunie/Developer/wl-noise-aware-mmd"
include(joinpath(MMD, "src", "wl_decompose.jl"))             # 2-material GLS (stdlib)
include(joinpath(MMD, "src", "wl_denoise.jl"))               # tv_denoise_weighted (2-material delivered)

const RAW = joinpath(MMD, "data", "naeotom", "raw")
const HAMRAW = joinpath(MMD, "data", "naeotom", "hamid", "raw")
const THEO = joinpath(MMD, "data", "naeotom", "analysis_57955439")
const OUT = joinpath(WLP, "assets"); mkpath(OUT)

const HU_W = (0.0, 0.0)
const HU_L = (-111.695, -81.213)                             # NIST triglyceride, shared by both methods
const M = load_wlp_model(joinpath(WLP, "wlp_model_70_150.toml"))
const PL = M.G[:, 1]; const PP = M.G[:, 2]                   # lipid / protein endpoint vectors (water=0)
const BETA = 0.12                                            # L1 protein-admission threshold (tuned below)

# local noise covariance measured from a scan's own uniform-fat ROI (endpoints stay theoretical NIST)
function measure_Σ(a, b, mask)
    r70 = Float64.(a[mask]); r150 = Float64.(b[mask])
    r70 .-= mean(r70); r150 .-= mean(r150)
    σ7 = std(r70); σ1 = std(r150); ρ = cor(r70, r150)
    ([σ7^2 ρ*σ7*σ1; ρ*σ7*σ1 σ1^2], σ7, σ1, ρ)
end

# ── helpers ──────────────────────────────────────────────────────────────────────────────────
function load_raw(path)
    isfile(path) || error("missing raw: $path")
    m = match(r"_(\d+)x(\d+)x(\d+)_float32", basename(path))
    nx, ny, nz = parse.(Int, m.captures)
    reshape(collect(reinterpret(Float32, read(path))), nx, ny, nz)
end
disp(x) = reverse(x; dims = 2)
function erode1(mask)
    nx, ny = size(mask); out = falses(nx, ny)
    @inbounds for j in 2:ny-1, i in 2:nx-1
        out[i, j] = mask[i, j] && mask[i-1, j] && mask[i+1, j] && mask[i, j-1] && mask[i, j+1]
    end
    out
end
# per-voxel SD of a 2-D map inside a mask, after removing the mask mean (local-noise proxy)
function map_sd(fmap, mask)
    v = filter(isfinite, fmap[mask]); length(v) < 2 && return NaN
    std(v .- mean(v))
end
shared_gate(v70, v150) = (v70 .≥ HU_L[1] - 40) .& (v70 .≤ 150) .&
                         (v150 .≥ M.hu_lo) .& (v150 .≤ M.hu_hi)

# ── 3-material estimator with tissue prior (per 2-D slice) ─────────────────────────────────────
# Water/lipid FIRST (clean 1-D GLS on the raw HU = 2-material image quality), then admit protein
# only where a STRONGLY-smoothed, L1-thresholded off-line signal survives. Where f_p→0 the lipid
# fraction is exactly the 2-material projection, so clean fat keeps 2-material noise; f_p appears
# only in spatially-coherent fibrous/pericardial tissue.
function decompose3(a, b, gate, Σloc; beta = BETA, sm_lambda = 0.12, sm_iters = 50)
    Σil = inv(Σloc); den = (PL' * Σil * PL)                  # 1-D GLS weighting on the SCAN's own noise
    fw_u, fl_u, fp_u = decode_maps(M, a, b)                  # raw per-voxel decode (wlp-gated)
    w = sigma_f_weight(M, a, b)
    _, fp_s = tv_coupled(fl_u, fp_u, gate; lambda = sm_lambda, iters = sm_iters, eps = 0.04, w = w)  # smooth protein
    nx, ny = size(a)
    FW = fill(NaN, nx, ny); FL = similar(FW); FP = similar(FW)
    @inbounds for j in 1:ny, i in 1:nx
        gate[i, j] || continue
        fps = fp_s[i, j]; isfinite(fps) || continue
        fp = max(fps - beta, 0.0)                            # L1 protein admission (else 0)
        d1 = a[i, j] - fp * PP[1]; d2 = b[i, j] - fp * PP[2]  # lipid from RAW HU, protein removed
        fl = (PL[1] * (Σil[1, 1]*d1 + Σil[1, 2]*d2) + PL[2] * (Σil[2, 1]*d1 + Σil[2, 2]*d2)) / den
        fl = clamp(fl, 0.0, 1.0)
        if fl + fp > 1; s = fl + fp; fl /= s; fp /= s; end
        FL[i, j] = fl; FP[i, j] = fp; FW[i, j] = 1 - fl - fp
    end
    (FW, FL, FP)
end
tv2(y, gate) = tv_denoise_weighted(y, Float64.(gate); lambda = 0.05, iters = 25, huber_eps = 0.04, mask = gate)

# ════════════════════════════════════════════════════════════════════════════════════════════
# TARGET 1 — human CCTA 57955439 (process the 3 display slices only; all the figure + σ needs)
# ════════════════════════════════════════════════════════════════════════════════════════════
println("\n══ TARGET 1: human CCTA 57955439 ══")
const Z0 = 110                                              # theolipid slab starts at z=110
const SHOWZ = [149, 179, 209]                              # absolute z (== reference figure)
v70f = load_raw(joinpath(RAW, "naeotom_57955439_mono70keV_513x512x339_float32.raw"))
v150f = load_raw(joinpath(RAW, "naeotom_57955439_mono150keV_513x512x339_float32.raw"))
fl2v = load_raw(joinpath(THEO, "fl_theolipid_57955439_513x512x141_float32.raw"))   # 2-mat, raw, slab

# measure the human scan's own noise covariance from uniform subcut fat (pooled over slices)
let r70 = Float64[], r150 = Float64[]
    global Σh, σ7h, σ1h, ρh
    for z in SHOWZ
        a = v70f[:, :, z]; b = v150f[:, :, z]
        fm = erode1((a .≥ -130) .& (a .≤ -70) .& (b .≥ -110) .& (b .≤ -50))
        va = Float64.(a[fm]); vb = Float64.(b[fm])
        append!(r70, va .- mean(va)); append!(r150, vb .- mean(vb))
    end
    σ7h = std(r70); σ1h = std(r150); ρh = cor(r70, r150)
    Σh = [σ7h^2 ρh*σ7h*σ1h; ρh*σ7h*σ1h σ1h^2]
end
@printf("human fat noise σ70/σ150 = %.1f/%.1f  ρ=%.2f\n", σ7h, σ1h, ρh)

hum = Dict{Int,NamedTuple}()
fat2 = Float64[]; fat3 = Float64[]                          # fat-ROI f_l pooled over slices
for z in SHOWZ
    a = v70f[:, :, z]; b = v150f[:, :, z]
    fl2 = fl2v[:, :, z-Z0+1]
    gate = shared_gate(a, b) .& .!isnan.(fl2)              # identical voxel set for both methods
    fw3, fl3, fp3 = decompose3(a, b, gate, Σh)
    fl2d = tv2(fl2, gate)                                   # 2-material delivered (matched TV)
    fatm = erode1((a .≥ -130) .& (a .≤ -70) .& (b .≥ -110) .& (b .≤ -50) .& gate)  # uniform fat ROI
    append!(fat2, filter(isfinite, fl2d[fatm])); append!(fat3, filter(isfinite, fl3[fatm]))
    hum[z] = (a = a, fw3 = fw3, fl3 = fl3, fp3 = fp3, fl2 = fl2d, gate = gate, fatm = fatm)
end
sd2 = std(fat2 .- mean(fat2)); sd3 = std(fat3 .- mean(fat3))
@printf("fat ROI %d vox · mean f_l 2-mat %.3f  3-mat %.3f\n", length(fat2), mean(fat2), mean(fat3))
@printf("delivered per-voxel SD f_l : 2-mat %.4f  3-mat %.4f   (ratio %.2f×)\n", sd2, sd3, sd3 / sd2)

# β sweep (image-quality check): fat-ROI σ(f_l) as protein admission tightens
print("β sweep fat σ(f_l):")
for β in (0.0, 0.06, 0.12, 0.18, 0.24)
    acc = Float64[]
    for z in SHOWZ
        h = hum[z]; _, fl, _ = decompose3(h.a, v150f[:, :, z], h.gate, Σh; beta = β)
        append!(acc, filter(isfinite, fl[h.fatm]))
    end
    @printf("  β=%.2f→%.3f", β, std(acc .- mean(acc)))
end
println()

# ── FIGURE H1: 3-material delivered (mirrors the reference 4-panel layout) ────────────────────
figH1 = CM.Figure(size = (1500, 320 * length(SHOWZ)))
for (r, z) in enumerate(SHOWZ)
    h = hum[z]; ct = disp(h.a)
    panels = [("70 keV CT", ct, :grays, (-160, 240), nothing),
              ("f_w  (jet)", disp(h.fw3), :jet, (0, 1), ct),
              ("f_l  lipid  (jet)", disp(h.fl3), :jet, (0, 1), ct),
              ("f_p  protein/fibrous  (jet, 0–0.4)", disp(h.fp3), :jet, (0, 0.4), ct)]
    for (c, (ttl, img, cmap, cr, under)) in enumerate(panels)
        ax = CM.Axis(figH1[r, c]; title = r == 1 ? ttl : "", titlesize = 13)
        CM.hidedecorations!(ax); ax.aspect = CM.DataAspect()
        under !== nothing && CM.heatmap!(ax, under; colormap = :grays, colorrange = (-160, 240))
        hm = CM.heatmap!(ax, img; colormap = cmap, colorrange = cr, nan_color = (:black, 0.0))
        c == 1 && CM.text!(ax, 8, 14; text = "z=$z", color = :yellow, fontsize = 12)
        (r == 1 && c ≥ 2) && CM.Colorbar(figH1[r, c, CM.Right()], hm; width = 10)
    end
end
CM.Label(figH1[0, :], "57955439 — 3-material water/lipid/protein (wlp-decomposition, 70/150 keV; tissue-prior MAP, β=$BETA)";
         fontsize = 14, font = :bold)
CM.save(joinpath(OUT, "fwlp_maps_57955439.png"), figH1; px_per_unit = 1.2)

# ── FIGURE H2: 2-material vs 3-material ────────────────────────────────────────────────────────
figH2 = CM.Figure(size = (1850, 320 * length(SHOWZ)))
for (r, z) in enumerate(SHOWZ)
    h = hum[z]; ct = disp(h.a)
    exc = (1 .- h.fl2) .- h.fw3                             # excess water assigned by 2-mat (=f_w²−f_w³)
    panels = [("70 keV CT", ct, :grays, (-160, 240), nothing),
              ("f_l  2-material (baseline)", disp(h.fl2), :jet, (0, 1), ct),
              ("f_l  3-material", disp(h.fl3), :jet, (0, 1), ct),
              ("f_p  3-material (0–0.4)", disp(h.fp3), :jet, (0, 0.4), ct),
              ("excess water in 2-mat  (f_w²−f_w³)", disp(exc), :balance, (-0.5, 0.5), ct)]
    for (c, (ttl, img, cmap, cr, under)) in enumerate(panels)
        ax = CM.Axis(figH2[r, c]; title = r == 1 ? ttl : "", titlesize = 12)
        CM.hidedecorations!(ax); ax.aspect = CM.DataAspect()
        under !== nothing && CM.heatmap!(ax, under; colormap = :grays, colorrange = (-160, 240))
        hm = CM.heatmap!(ax, img; colormap = cmap, colorrange = cr, nan_color = (:black, 0.0))
        c == 1 && CM.text!(ax, 8, 14; text = "z=$z", color = :yellow, fontsize = 12)
        (r == 1 && c ≥ 2) && CM.Colorbar(figH2[r, c, CM.Right()], hm; width = 10)
    end
end
CM.Label(figH2[0, :], "57955439 — 2-material vs 3-material (identical anchors + gate). Pericardium/fibrous appears in f_p; 2-material misassigns it as water (right).";
         fontsize = 13, font = :bold)
CM.save(joinpath(OUT, "compare_2mat_vs_3mat_57955439.png"), figH2; px_per_unit = 1.2)

# ── FIGURE H3: pericardial-fat zoom — inflammation surrogate (f_w in fat) + pericardium (f_p) ──
function heart_box(a; pad = 140)
    idx = findall(a .> 220)                                 # iodine blood pool ⇒ heart centre
    isempty(idx) && return (axes(a, 1), axes(a, 2))
    ci = clamp(round(Int, median(getindex.(idx, 1))), 1, size(a, 1))
    cj = clamp(round(Int, median(getindex.(idx, 2))), 1, size(a, 2))
    (max(1, ci - pad):min(size(a, 1), ci + pad), max(1, cj - pad):min(size(a, 2), cj + pad))
end
figH3 = CM.Figure(size = (1400, 350 * length(SHOWZ)))
for (r, z) in enumerate(SHOWZ)
    h = hum[z]; ri, rj = heart_box(h.a)
    ct = disp(h.a[ri, rj])
    fwfat = copy(h.fw3); fwfat[h.fl3 .< 0.4] .= NaN         # water fraction WITHIN lipid-dominant fat
    panels = [("70 keV CT", ct, :grays, (-160, 240), nothing),
              ("f_l  lipid", disp(h.fl3[ri, rj]), :jet, (0, 1), ct),
              ("f_w in fat  (inflammation surrogate, 0–0.35)", disp(fwfat[ri, rj]), :jet, (0, 0.35), ct),
              ("f_p  protein/fibrous  (pericardium)", disp(h.fp3[ri, rj]), :jet, (0, 0.4), ct)]
    for (c, (ttl, img, cmap, cr, under)) in enumerate(panels)
        ax = CM.Axis(figH3[r, c]; title = r == 1 ? ttl : "", titlesize = 12)
        CM.hidedecorations!(ax); ax.aspect = CM.DataAspect()
        under !== nothing && CM.heatmap!(ax, under; colormap = :grays, colorrange = (-160, 240))
        hm = CM.heatmap!(ax, img; colormap = cmap, colorrange = cr, nan_color = (:black, 0.0))
        c == 1 && CM.text!(ax, 8, 14; text = "z=$z", color = :yellow, fontsize = 12)
        (r == 1 && c ≥ 2) && CM.Colorbar(figH3[r, c, CM.Right()], hm; width = 10)
    end
end
CM.Label(figH3[0, :], "57955439 — pericardial-fat zoom. Inflamed fat = higher f_w within lipid; pericardium/fibrous = f_p (invisible to the 2-material model).";
         fontsize = 13, font = :bold)
CM.save(joinpath(OUT, "fwlp_pericardium_zoom_57955439.png"), figH3; px_per_unit = 1.3)

v70f = nothing; v150f = nothing; fl2v = nothing; GC.gc()

# ════════════════════════════════════════════════════════════════════════════════════════════
# TARGET 2 — Hamid QRM/Gammex phantom
# ════════════════════════════════════════════════════════════════════════════════════════════
println("\n══ TARGET 2: Hamid QRM/Gammex phantom ══")
h70f = load_raw(joinpath(HAMRAW, "hamid_study3_large_mono70keV_512x512x45_float32.raw"))
h150f = load_raw(joinpath(HAMRAW, "hamid_study3_large_mono150keV_512x512x45_float32.raw"))
const HZM = 23                                              # z=22 (0-idx) canonical rod slice
a = h70f[:, :, HZM]; b = h150f[:, :, HZM]
gate = shared_gate(a, b)

# 2-material baseline: frozen GLS, theoretical anchors, flat σ line (uniform phantom), ρ from fat body
fatm = erode1((a .≥ -120) .& (a .≤ -45) .& (a .> -500))
σ70 = map_sd(a, fatm); σ150 = map_sd(b, fatm)
fr70 = Float64.(a[fatm]) .- mean(Float64.(a[fatm])); fr150 = Float64.(b[fatm]) .- mean(Float64.(b[fatm]))
ρ = cor(fr70, fr150)
out2 = decompose_volume(a, b; HU_w = HU_W, HU_l = HU_L, ab_low = (0.0, σ70), ab_high = (0.0, σ150), rho = ρ)
hfw2 = out2.fhat; hfl2 = 1.0 .- hfw2; hfl2[.!gate] .= NaN
hfl2d = tv2(hfl2, gate)
Σp = [σ70^2 ρ*σ70*σ150; ρ*σ70*σ150 σ150^2]                 # phantom's own measured noise cov
hfw3, hfl3, hfp3 = decompose3(a, b, gate, Σp)
@printf("phantom fat ROI %d vox · σ_HU 70/150 = %.1f/%.1f · ρ=%.2f\n", count(fatm), σ70, σ150, ρ)
@printf("phantom fat body  mean f_l: 2-mat %.3f  3-mat %.3f  (f_p %.3f)\n",
        mean(filter(isfinite, hfl2d[fatm])), mean(filter(isfinite, hfl3[fatm])), mean(filter(isfinite, hfp3[fatm])))
@printf("phantom fat body  σ(f_l): 2-mat %.4f  3-mat %.4f\n", map_sd(hfl2d, fatm), map_sd(hfl3, fatm))

rods = [("iodine 10mg", 228, 240), ("bone/Ca200", 290, 232), ("iodine 5mg", 257, 269), ("iodine 7.5mg", 262, 205)]
println("rod rejection (both methods share the gate):")
for (lab, x, y) in rods
    @printf("  %-13s v70=%.0f v150=%.0f → %s\n", lab, a[x+1, y+1], b[x+1, y+1], gate[x+1, y+1] ? "KEPT (leak)" : "rejected")
end

figP = CM.Figure(size = (1850, 360))
pan = [("70 keV CT", disp(a), :grays, (-160, 400), nothing),
       ("150 keV CT", disp(b), :grays, (-160, 400), nothing),
       ("soft-tissue gate", disp(Float64.(gate)), :grays, (0, 1), nothing),
       ("f_l  2-material", disp(hfl2d), :jet, (0, 1), disp(a)),
       ("f_l  3-material", disp(hfl3), :jet, (0, 1), disp(a)),
       ("f_p  3-material (0–0.4)", disp(hfp3), :jet, (0, 0.4), disp(a))]
for (c, (ttl, img, cmap, cr, under)) in enumerate(pan)
    ax = CM.Axis(figP[1, c]; title = ttl, titlesize = 12)
    CM.hidedecorations!(ax); ax.aspect = CM.DataAspect()
    under !== nothing && CM.heatmap!(ax, under; colormap = :grays, colorrange = (-160, 400))
    hm = CM.heatmap!(ax, img; colormap = cmap, colorrange = cr, nan_color = (:black, 0.0))
    c ≥ 4 && CM.Colorbar(figP[1, c, CM.Right()], hm; width = 8)
end
CM.Label(figP[0, :], "Hamid QRM/Gammex phantom (z=22) — iodine×3 + bone rejected by both; fat body decodes. No protein rod (ground truth).";
         fontsize = 13, font = :bold)
CM.save(joinpath(OUT, "phantom_hamid_2mat_vs_3mat.png"), figP; px_per_unit = 1.3)

# ── numeric summary ──────────────────────────────────────────────────────────────────────────
open(joinpath(OUT, "real_ct_summary.csv"), "w") do io
    println(io, "target,roi,metric,2material,3material,n_vox")
    @printf(io, "human_57955439,subcut_fat,delivered_SD_fl,%.4f,%.4f,%d\n", sd2, sd3, length(fat2))
    @printf(io, "human_57955439,subcut_fat,mean_fl,%.3f,%.3f,%d\n", mean(fat2), mean(fat3), length(fat2))
    @printf(io, "hamid_phantom,fat_body,SD_fl,%.4f,%.4f,%d\n", map_sd(hfl2d, fatm), map_sd(hfl3, fatm), count(fatm))
end
println("\nwrote → $OUT :  fwlp_maps_57955439.png  compare_2mat_vs_3mat_57955439.png  phantom_hamid_2mat_vs_3mat.png  real_ct_summary.csv")
