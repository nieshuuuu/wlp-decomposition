# Apply THIS repo's 3-material (water/lipid/protein) model to REAL 70/150 keV VMI CT and
# compare it, at identical anchors and identical gate, against the 2-material (water/lipid)
# GLS baseline. Two targets:
#   (1) human CCTA 57955439  — the anatomical win (pericardium/fibrous, inflamed fat, lipid gradient)
#   (2) Hamid QRM/Gammex phantom — image quality (σ), iodine/bone/lung rejection, fat-anchor accuracy
#
# 3-material algorithm  : wlp-decomposition/wlp_model_70_150.toml via apply_wlp_model.jl (THIS repo).
# 2-material baseline    : wl-noise-aware-mmd/src/wl_decompose.jl decompose_volume (frozen GLS core).
# NO wlp method from wl-noise-aware-mmd is used. Both methods share the SAME NIST endpoints and the
# SAME lung/bone/iodine gate, so the only variable is the third (protein) material.
#
#   julia --project=. apply_to_real_ct.jl
#
# Run under --project=. (CairoMakie for figures). Data lives in the wl-noise-aware-mmd repo.

using LinearAlgebra, Statistics, Printf, DelimitedFiles
import CairoMakie as CM

const WLP = @__DIR__
include(joinpath(WLP, "apply_wlp_model.jl"))                 # 3-material consumer (stdlib)
const MMD = "/Users/shunie/Developer/wl-noise-aware-mmd"
include(joinpath(MMD, "src", "wl_decompose.jl"))             # 2-material GLS (stdlib, no BasisSimulator)
include(joinpath(MMD, "src", "wl_denoise.jl"))               # tv_denoise_weighted (2-material delivered)

const RAW = joinpath(MMD, "data", "naeotom", "raw")
const HAMRAW = joinpath(MMD, "data", "naeotom", "hamid", "raw")
const THEO = joinpath(MMD, "data", "naeotom", "analysis_57955439")
const OUT = joinpath(WLP, "assets"); mkpath(OUT)

# theoretical NIST endpoints at 70/150 keV — SAME anchors for both methods (SSoT: match the
# wlp model's endpoints_hu + the 2-material theolipid baseline). protein endpoint (270.6,290.1)
# lives inside the wlp model only.
const HU_W = (0.0, 0.0)
const HU_L = (-111.695, -81.213)                             # NIST triglyceride, f_l=1 => pure lipid

const M = load_wlp_model(joinpath(WLP, "wlp_model_70_150.toml"))

# ── raw I/O (dims parsed from the lab-convention filename) ───────────────────────────────────
function load_raw(path)
    isfile(path) || error("missing raw: $path")
    m = match(r"_(\d+)x(\d+)x(\d+)_float32", basename(path))
    nx, ny, nz = parse.(Int, m.captures)
    reshape(collect(reinterpret(Float32, read(path))), nx, ny, nz)
end
write_raw(name, A) = write(joinpath(OUT, name), Array{Float32}(A))
disp(x) = reverse(x; dims = 2)                               # spine-down radiological display

# 1-voxel 4-neighbour erosion (strip partial-volume rim voxels before ROI stats)
function erode1(mask)
    nx, ny = size(mask); out = falses(nx, ny)
    @inbounds for j in 2:ny-1, i in 2:nx-1
        out[i, j] = mask[i, j] && mask[i-1, j] && mask[i+1, j] && mask[i, j-1] && mask[i, j+1]
    end
    out
end

# per-voxel SD of a quantity inside a 3-D ROI, slice-detrended (removes each slice's ROI mean)
function roi_sd(vol, mask3)
    res = Float64[]
    for k in axes(vol, 3)
        m = @view mask3[:, :, k]; any(m) || continue
        v = Float64.(vol[:, :, k][m]); v = filter(isfinite, v .- mean(filter(isfinite, v)))
        append!(res, v)
    end
    isempty(res) ? NaN : std(res)
end

# shared soft-tissue gate: like the 2-material baseline (v70 in [HU_l-40, 150] removes gas/lung
# below and iodine/calcium/bone above) AND the wlp high-channel gate (v150 in [-300, 250]).
shared_gate(v70, v150) = (v70 .≥ HU_L[1] - 40) .& (v70 .≤ 150) .&
                         (v150 .≥ M.hu_lo) .& (v150 .≤ M.hu_hi)

# 3-material per-voxel decode over a slab (already NaN outside the wlp gate); returns raw maps
function wlp_raw(v70, v150)
    fw, fl, fp = decode_maps(M, v70, v150)                   # per-voxel, wlp-gated, no TV
    (fw, fl, fp)
end

# ════════════════════════════════════════════════════════════════════════════════════════════
# TARGET 1 — human CCTA 57955439
# ════════════════════════════════════════════════════════════════════════════════════════════
println("\n══ TARGET 1: human CCTA 57955439 ══")
v70f = load_raw(joinpath(RAW, "naeotom_57955439_mono70keV_513x512x339_float32.raw"))
v150f = load_raw(joinpath(RAW, "naeotom_57955439_mono150keV_513x512x339_float32.raw"))
const Z0, Z1 = 110, 250                                       # same slab as the theolipid baseline
v70 = v70f[:, :, Z0:Z1]; v150 = v150f[:, :, Z0:Z1]
v70f = nothing; v150f = nothing; GC.gc()                      # free the full volumes
nx, ny, nz = size(v70)
@printf("slab %d:%d  (%d slices)  dims %d×%d\n", Z0, Z1, nz, nx, ny)

# 2-material baseline = the PUBLISHED theolipid maps (raw GLS, theoretical anchor) — load as-is
fl2 = load_raw(joinpath(THEO, "fl_theolipid_57955439_513x512x141_float32.raw"))
fw2 = load_raw(joinpath(THEO, "fw_theolipid_57955439_513x512x141_float32.raw"))
@assert size(fl2) == size(v70) "theolipid slab dims mismatch"

# 3-material (this repo) — raw per-voxel decode
fw3, fl3, fp3 = wlp_raw(v70, v150)

# identical voxel set for both: 2-material gate (theolipid NaNs) ∩ wlp shared gate
gate = .!isnan.(fl2) .& shared_gate(v70, v150)
for A in (fw2, fl2); A[.!gate] .= NaN; end
for A in (fw3, fl3, fp3); A[.!gate] .= NaN; end
@printf("shared gate keeps %.1f%% of slab voxels\n", 100 * count(gate) / length(gate))
@assert count(gate) > 100_000 "gate too aggressive"

# σ (image quality): per-voxel SD of f_l in a uniform subcutaneous-fat ROI, raw, same ROI both
fatm = (v70 .≥ -130) .& (v70 .≤ -70) .& (v150 .≥ -110) .& (v150 .≤ -50) .& gate
fate = similar(fatm); for k in axes(fatm, 3); fate[:, :, k] = erode1(fatm[:, :, k]); end
nfat = count(fate)
# RAW per-voxel σ: 3-material carries more variance by construction (extra DOF on the
# ill-conditioned W/L/P sliver) — this is exactly what the method's σ_f-weighted TV controls, so
# the fair image-quality comparison is at the DELIVERED stage below.
sd2r = roi_sd(fl2, fate); sd3r = roi_sd(fl3, fate)
fl2m = mean(filter(isfinite, fl2[fate])); fl3m = mean(filter(isfinite, fl3[fate]))
@printf("fat ROI %d vox · mean f_l: 2-mat %.3f  3-mat %.3f\n", nfat, fl2m, fl3m)
@printf("RAW per-voxel SD f_l : 2-mat %.4f  3-mat %.4f\n", sd2r, sd3r)

# delivered (TV) maps over the whole slab — both methods, edge-preserving Huber-TV (λ=0.05, 25 it):
# 3-mat = σ_f-weighted coupled (f_l,f_p) [decode_maps_tv]; 2-mat = Huber-TV on f_l [tv_denoise_weighted].
fw3d = fill(NaN, size(v70)); fl3d = similar(fw3d); fp3d = similar(fw3d); fl2d = similar(fw3d)
for k in axes(v70, 3)
    a = v70[:, :, k]; b = v150[:, :, k]; gk = collect(@view gate[:, :, k])
    fw_t, fl_t, fp_t = decode_maps_tv(M, a, b)
    fl2k = tv_denoise_weighted(fl2[:, :, k], Float64.(gk); lambda = 0.05, iters = 25, huber_eps = 0.04, mask = gk)
    @inbounds for j in axes(fw3d, 2), i in axes(fw3d, 1)
        keep = gk[i, j]
        fw3d[i, j, k] = keep ? fw_t[i, j] : NaN
        fl3d[i, j, k] = keep ? fl_t[i, j] : NaN
        fp3d[i, j, k] = keep ? fp_t[i, j] : NaN
        fl2d[i, j, k] = keep ? fl2k[i, j] : NaN
    end
end
sd2d = roi_sd(fl2d, fate); sd3d = roi_sd(fl3d, fate)
@printf("DELIVERED per-voxel SD f_l : 2-mat %.4f  3-mat %.4f  (SE of ROI mean ÷√N: %.5f / %.5f)\n",
        sd2d, sd3d, sd2d / sqrt(nfat), sd3d / sqrt(nfat))

const SHOW = [40, 70, 100]                                    # slab-relative → abs z = 149,179,209
dimstr = "$(nx)x$(ny)x$(nz)"
write_raw("fw3_57955439_$(dimstr)_float32.raw", fw3d)
write_raw("fl3_57955439_$(dimstr)_float32.raw", fl3d)
write_raw("fp3_57955439_$(dimstr)_float32.raw", fp3d)

# ── FIGURE H1: 3-material delivered map (mirrors the reference 4-panel layout) ────────────────
figH1 = CM.Figure(size = (1500, 320 * length(SHOW)))
for (r, s) in enumerate(SHOW)
    ct = disp(v70[:, :, s])
    panels = [("70 keV CT", ct, :grays, (-160, 240), nothing),
              ("f_w  (jet)", disp(fw3d[:, :, s]), :jet, (0, 1), ct),
              ("f_l  lipid  (jet)", disp(fl3d[:, :, s]), :jet, (0, 1), ct),
              ("f_p  protein/fibrous  (jet)", disp(fp3d[:, :, s]), :jet, (0, 1), ct)]
    for (c, (ttl, img, cmap, crange, under)) in enumerate(panels)
        ax = CM.Axis(figH1[r, c]; title = r == 1 ? ttl : "", titlesize = 13)
        CM.hidedecorations!(ax); ax.aspect = CM.DataAspect()
        under !== nothing && CM.heatmap!(ax, under; colormap = :grays, colorrange = (-160, 240))
        hm = CM.heatmap!(ax, img; colormap = cmap, colorrange = crange, nan_color = (:black, 0.0))
        c == 1 && CM.text!(ax, 8, 14; text = "z=$(Z0 + s - 1)", color = :yellow, fontsize = 12)
        (r == 1 && c ≥ 2) && CM.Colorbar(figH1[r, c, CM.Right()], hm; width = 10)
    end
end
CM.Label(figH1[0, :], "57955439 — 3-material water/lipid/protein (wlp-decomposition, 70/150 keV VMI; theoretical NIST anchors, σ_f-weighted Huber-TV)";
         fontsize = 14, font = :bold)
CM.save(joinpath(OUT, "fwlp_maps_57955439.png"), figH1; px_per_unit = 1.2)

# ── FIGURE H2: 2-material vs 3-material (the comparison) ──────────────────────────────────────
# col: CT | f_l 2-mat | f_l 3-mat | f_p 3-mat | excess water assigned by 2-mat (f_w2 - f_w3)
dq = 0.5
figH2 = CM.Figure(size = (1850, 320 * length(SHOW)))
for (r, s) in enumerate(SHOW)
    ct = disp(v70[:, :, s])
    # excess water = f_w(2-mat) − f_w(3-mat), both TV-delivered, on the shared gate
    exc = (1 .- fl2d[:, :, s]) .- fw3d[:, :, s]
    panels = [("70 keV CT", ct, :grays, (-160, 240), nothing),
              ("f_l  2-material (baseline)", disp(fl2d[:, :, s]), :jet, (0, 1), ct),
              ("f_l  3-material", disp(fl3d[:, :, s]), :jet, (0, 1), ct),
              ("f_p  3-material (protein/fibrous)", disp(fp3d[:, :, s]), :jet, (0, 1), ct),
              ("excess water in 2-mat  (f_w²−f_w³)", disp(exc), :balance, (-dq, dq), ct)]
    for (c, (ttl, img, cmap, crange, under)) in enumerate(panels)
        ax = CM.Axis(figH2[r, c]; title = r == 1 ? ttl : "", titlesize = 12)
        CM.hidedecorations!(ax); ax.aspect = CM.DataAspect()
        under !== nothing && CM.heatmap!(ax, under; colormap = :grays, colorrange = (-160, 240))
        hm = CM.heatmap!(ax, img; colormap = cmap, colorrange = crange, nan_color = (:black, 0.0))
        c == 1 && CM.text!(ax, 8, 14; text = "z=$(Z0 + s - 1)", color = :yellow, fontsize = 12)
        (r == 1 && c ≥ 2) && CM.Colorbar(figH2[r, c, CM.Right()], hm; width = 10)
    end
end
CM.Label(figH2[0, :], "57955439 — 2-material vs 3-material (identical anchors + gate). Pericardium/fibrous appears in f_p; 2-material misassigns it as water (right).";
         fontsize = 13, font = :bold)
CM.save(joinpath(OUT, "compare_2mat_vs_3mat_57955439.png"), figH2; px_per_unit = 1.2)

# ════════════════════════════════════════════════════════════════════════════════════════════
# TARGET 2 — Hamid QRM/Gammex phantom (ground truth: fat body + iodine×3 + bone; NO protein rod)
# ════════════════════════════════════════════════════════════════════════════════════════════
println("\n══ TARGET 2: Hamid QRM/Gammex phantom ══")
h70f = load_raw(joinpath(HAMRAW, "hamid_study3_large_mono70keV_512x512x45_float32.raw"))
h150f = load_raw(joinpath(HAMRAW, "hamid_study3_large_mono150keV_512x512x45_float32.raw"))
const HZ = 18:30                                              # clean rod band (z=22 canonical)
h70 = h70f[:, :, HZ]; h150 = h150f[:, :, HZ]
hnx, hny, hnz = size(h70)

# 2-material baseline on the phantom: frozen GLS at the SAME theoretical anchors.
# uniform phantom ⇒ flat σ(HU) line ab=(0,σ) per channel; ρ from fat-body residuals.
hbody = h70 .> -500
hfat = (h70 .≥ -120) .& (h70 .≤ -45) .& hbody                # QRM adipose body
hfate = similar(hfat); for k in axes(hfat, 3); hfate[:, :, k] = erode1(hfat[:, :, k]); end
σh70 = roi_sd(h70, hfate); σh150 = roi_sd(h150, hfate)
fr70 = Float64.(h70[hfate]) .- mean(Float64.(h70[hfate]))
fr150 = Float64.(h150[hfate]) .- mean(Float64.(h150[hfate]))
ρh = cor(fr70, fr150)
@printf("phantom fat ROI %d vox · σ_HU (70/150)= %.1f / %.1f · ρ=%.3f\n", count(hfate), σh70, σh150, ρh)

out2 = decompose_volume(h70, h150; HU_w = HU_W, HU_l = HU_L,
                        ab_low = (0.0, σh70), ab_high = (0.0, σh150), rho = ρh)
hfw2 = out2.fhat; hfl2 = 1.0 .- hfw2
hfw3, hfl3, hfp3 = wlp_raw(h70, h150)

hgate = shared_gate(h70, h150)
for A in (hfw2, hfl2); A[.!hgate] .= NaN; end
for A in (hfw3, hfl3, hfp3); A[.!hgate] .= NaN; end
@printf("phantom shared gate keeps %.1f%% of voxels\n", 100 * count(hgate) / length(hgate))

# image quality (σ) + fat-anchor accuracy in the fat body ROI
hsd2 = roi_sd(hfl2, hfate); hsd3 = roi_sd(hfl3, hfate)
hfl2m = mean(filter(isfinite, hfl2[hfate])); hfl3m = mean(filter(isfinite, hfl3[hfate]))
hfp3m = mean(filter(isfinite, hfp3[hfate]))
@printf("phantom fat body  f_l: 2-mat %.3f  3-mat %.3f  (f_p 3-mat %.3f)\n", hfl2m, hfl3m, hfp3m)
@printf("phantom fat body  per-voxel SD f_l: 2-mat %.4f  3-mat %.4f\n", hsd2, hsd3)

# rod rejection: both methods must NaN the iodine (×3) + bone rods. Report gate pass at rod centres.
rods = [("iodine 10mg", 228, 240), ("bone/Ca200", 290, 232),
        ("iodine 5mg", 257, 269), ("iodine 7.5mg", 262, 205)]  # (label, x, y) at z=22 (0-idx)
zc = 23 - (first(HZ) - 1)                                      # z=22(0-idx)=23(1-idx) → slab index
println("rod rejection (kept=leaks into soft-tissue decomposition):")
for (lab, x, y) in rods
    kept = hgate[x+1, y+1, zc]
    @printf("  %-14s v70=%.0f v150=%.0f  → %s\n", lab, h70[x+1, y+1, zc], h150[x+1, y+1, zc],
            kept ? "KEPT (leak)" : "rejected")
end

# ── FIGURE P1: phantom rejection + maps (mid slab slice) ─────────────────────────────────────
zm = zc
figP = CM.Figure(size = (1850, 360))
pan = [("70 keV CT", disp(h70[:, :, zm]), :grays, (-160, 400), nothing),
       ("150 keV CT", disp(h150[:, :, zm]), :grays, (-160, 400), nothing),
       ("soft-tissue gate", disp(Float64.(hgate[:, :, zm])), :grays, (0, 1), nothing),
       ("f_l  2-material", disp(hfl2[:, :, zm]), :jet, (0, 1), disp(h70[:, :, zm])),
       ("f_l  3-material", disp(hfl3[:, :, zm]), :jet, (0, 1), disp(h70[:, :, zm])),
       ("f_p  3-material", disp(hfp3[:, :, zm]), :jet, (0, 1), disp(h70[:, :, zm]))]
for (c, (ttl, img, cmap, cr, under)) in enumerate(pan)
    ax = CM.Axis(figP[1, c]; title = ttl, titlesize = 12)
    CM.hidedecorations!(ax); ax.aspect = CM.DataAspect()
    under !== nothing && CM.heatmap!(ax, under; colormap = :grays, colorrange = (-160, 400))
    hm = CM.heatmap!(ax, img; colormap = cmap, colorrange = cr, nan_color = (:black, 0.0))
    c ≥ 4 && CM.Colorbar(figP[1, c, CM.Right()], hm; width = 8)
end
CM.Label(figP[0, :], "Hamid QRM/Gammex phantom (z=22) — iodine×3 + bone rejected by both; fat body decodes. No protein rod (see console).";
         fontsize = 13, font = :bold)
CM.save(joinpath(OUT, "phantom_hamid_2mat_vs_3mat.png"), figP; px_per_unit = 1.3)

# ── numeric summary CSV ──────────────────────────────────────────────────────────────────────
open(joinpath(OUT, "real_ct_summary.csv"), "w") do io
    println(io, "target,roi,metric,2material,3material,n_vox")
    @printf(io, "human_57955439,subcut_fat,pervoxel_SD_fl_raw,%.4f,%.4f,%d\n", sd2r, sd3r, nfat)
    @printf(io, "human_57955439,subcut_fat,pervoxel_SD_fl_delivered,%.4f,%.4f,%d\n", sd2d, sd3d, nfat)
    @printf(io, "human_57955439,subcut_fat,mean_fl,%.3f,%.3f,%d\n", fl2m, fl3m, nfat)
    @printf(io, "hamid_phantom,fat_body,pervoxel_SD_fl,%.4f,%.4f,%d\n", hsd2, hsd3, count(hfate))
    @printf(io, "hamid_phantom,fat_body,mean_fl,%.3f,%.3f,%d\n", hfl2m, hfl3m, count(hfate))
end

println("\nwrote figures + maps to $OUT")
foreach(println, ["  fwlp_maps_57955439.png", "  compare_2mat_vs_3mat_57955439.png",
                  "  phantom_hamid_2mat_vs_3mat.png", "  real_ct_summary.csv"])
