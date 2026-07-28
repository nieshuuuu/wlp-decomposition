# Decomposed map vs ground-truth map, side by side on the same crop and scale.
#
# Ground truth is the phantom's own per-label (f_w, f_l, f_p) — i.e. what an infinite-resolution
# scanner with a perfect decoder would return. Decoded is the poly2 surface applied to the
# measured 70/150 keV VMI pair. Both are shown on the SAME crop so the spatial blur is visible
# rather than summarised into a number.
import BasisSimulator as BS
import CairoMakie as CM
import TOML
using Statistics: mean, median
using Serialization: deserialize
using Printf: @printf
include(joinpath(@__DIR__, "wlp_tv.jl"))

const OUT = joinpath(@__DIR__, "pcat_ct")
const ACQ = get(ENV, "PCAT_ACQ", "pcat_acq_shell.jls")
const D = deserialize(joinpath(OUT, ACQ))
const MODEL = TOML.parsefile(joinpath(@__DIR__, "wlp_model_70_150.toml"))
const K, VOXMM = 6, 0.5
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
fat_label(k, i) = 40 + 6 * (K - k) + i
const RECON_N, RECON_NZ, RECON_FOV_CM = 512, 40, 18.0
const PX_MM = RECON_FOV_CM * 10 / RECON_N
const WALL0, LUM0 = 76, 82
const CW = Float64.(MODEL["poly2"]["cw"]); const CL = Float64.(MODEL["poly2"]["cl"])
const CP = Float64.(MODEL["poly2"]["cp"])
const GATE_LO = Float64(MODEL["gate"]["soft_hu_lo"]); const GATE_HI = Float64(MODEL["gate"]["soft_hu_hi"])

# RAW poly2 — no simplex here. The simplex must run once at the very end, after TV.
function decode_raw(hlo, hhi)
    b = (1.0, hlo, hhi, hlo^2, hhi^2, hlo * hhi)
    f = (sum(CW .* b), sum(CL .* b), sum(CP .* b))
    s = sum(f)
    abs(s) < 1e-12 ? (NaN, NaN, NaN) : (f[1]/s, f[2]/s, f[3]/s)
end
# ARGS[1] forces a scalar λ (dev knob); default is the card's λ(f̂) model, per voxel
const LAMBDA = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : nothing
const LAMTAG = LAMBDA === nothing ? "model" : string(LAMBDA)
const TVITERS = Int(MODEL["tv"]["iters"]); const TVEPS = Float64(MODEL["tv"]["eps"])

# ── grids + valid z ───────────────────────────────────────────────────────────────────
stub = Dict{Int, Any}(Int(l) => BS.XA.Materials.water for l in unique(D.slab))
m3 = BS.resample_to_recon(BS.Phantom(D.slab, stub, (VOXMM/10, VOXMM/10, VOXMM/10)),
                          D.geom, (RECON_N, RECON_N, RECON_NZ); method = :nearest)
nz = size(m3, 3)
myo = [let i = findall(x -> 15 <= Int(x) <= 18, m3[:, :, z])
           isempty(i) ? -Inf : mean(Float64.(D.hu_lo[:, :, z])[i]) end for z in 1:nz]
plate = let v = sort(filter(isfinite, myo)); median(v[(length(v)÷2+1):end]) end
good = [z for z in 1:nz if isfinite(myo[z]) && abs(myo[z] - plate) <= 8.0]
ZR = minimum(good):maximum(good)

# pick the slice with the most PCAT inside the valid range
pcount = [count(x -> 40 <= Int(x) < LUM0, m3[:, :, z]) for z in ZR]
Z = collect(ZR)[argmax(pcount)]
@info "valid z = $ZR; showing slice $Z ($(pcount[argmax(pcount)]) PCAT pixels)"

lab = m3[:, :, Z]
hlo = Float64.(D.hu_lo[:, :, Z]); hhi = Float64.(D.hu_hi[:, :, Z])

# ── ground truth + decoded maps ───────────────────────────────────────────────────────
gtw = fill(NaN, size(lab)); gtl = fill(NaN, size(lab)); gtp = fill(NaN, size(lab))
dew = fill(NaN, size(lab)); del = fill(NaN, size(lab)); dep = fill(NaN, size(lab))
for i in eachindex(lab)
    l = Int(lab[i])
    haskey(D.gt, l) && ((gtw[i], gtl[i], gtp[i]) = D.gt[l])
end

# poly2 -> noise-weighted coupled Huber-TV -> simplex ONCE. TV is half the delivered reader;
# poly2 alone is a per-voxel map and passes CT noise straight through.
rawl = zeros(size(lab)); rawp = zeros(size(lab))
wl = zeros(size(lab)); wp = zeros(size(lab)); valid = falses(size(lab))
for i in eachindex(lab)
    (GATE_LO <= hlo[i] <= GATE_HI) || continue
    f0, σ = wlp_sigma_f(hlo[i], hhi[i], MODEL, decode_raw)
    any(isnan, f0) && continue
    rawl[i] = f0[2]; rawp[i] = f0[3]
    wl[i] = 1 / σ[2]^2; wp[i] = 1 / σ[3]^2
    valid[i] = true
end
lam = LAMBDA === nothing ? wlp_lambda_map(MODEL, rawl, rawp, valid) : LAMBDA
@info "TV: lambda=$LAMTAG iters=$TVITERS eps=$TVEPS on $(count(valid)) decodable pixels"
tl, tp = wlp_tv!(rawl, rawp, wl, wp, valid; λ = lam, iters = TVITERS, eps = TVEPS)
for i in eachindex(lab)
    valid[i] || continue
    dew[i], del[i], dep[i] = wlp_simplex(tl[i], tp[i])   # once, after TV: every f in [0,1]
end

pcat = [(40 <= Int(l) < LUM0) || (88 <= Int(l) <= 127) for l in lab]   # grown PCAT + distance shells
ys, xs = (getindex.(findall(pcat), 1), getindex.(findall(pcat), 2))
pad = 24
r1, r2 = max(minimum(ys) - pad, 1), min(maximum(ys) + pad, RECON_N)
c1, c2 = max(minimum(xs) - pad, 1), min(maximum(xs) + pad, RECON_N)
crop(A) = A[r1:r2, c1:c2]
# anterior-up: heatmap(M) draws M[x,y] with the second index vertical and low j is anterior
# (sternum j=29.5, vertebra j=454.2), so reverse the second index. permutedims would rotate 90 deg.
disp(A) = reverse(A; dims = 2)
@info "crop $(r2-r1+1) x $(c2-c1+1) px = $(round((r2-r1+1)*PX_MM, digits=1)) x $(round((c2-c1+1)*PX_MM, digits=1)) mm"

# ── figure ────────────────────────────────────────────────────────────────────────────
# Volume-fraction maps use jet on a FIXED (0,1) range so panels are directly comparable.
# Protein is the stated exception: its true value is ~0.02, which renders as uniform dark blue
# on (0,1), so it gets its own (0, 0.12) range and the range is written into the panel title.
RNG = Dict(:w => (0.0, 1.0), :l => (0.0, 1.0), :p => (0.0, 0.12))
MATS = ((:w, "water", gtw, dew), (:l, "lipid", gtl, del), (:p, "protein", gtp, dep))

fig = CM.Figure(size = (1560, 1420))
for (col, (s, name, GT, DE)) in enumerate(MATS)
    lo, hi = RNG[s]
    # row 1 — ground truth (defined only where the phantom has a W/L/P material, i.e. the PCAT)
    ax1 = CM.Axis(fig[1, col]; aspect = CM.DataAspect(),
        title = "ground truth $name\nrange ($lo, $hi)", titlesize = 15)
    CM.heatmap!(ax1, disp(crop(gtw) .* 0 .+ 0.0); colormap = [CM.RGBf(.15,.15,.15)], colorrange = (0,1))
    CM.heatmap!(ax1, disp(crop(GT)); colormap = :jet, colorrange = (lo, hi), nan_color = :transparent)
    CM.hidedecorations!(ax1)
    # row 2 — decoded, same crop and same scale
    ax2 = CM.Axis(fig[2, col]; aspect = CM.DataAspect(),
        title = "decoded $name (70/150 keV)", titlesize = 15)
    hm = CM.heatmap!(ax2, disp(crop(DE)); colormap = :jet, colorrange = (lo, hi), nan_color = :black)
    CM.hidedecorations!(ax2)
    CM.Colorbar(fig[4, col], hm; vertical = false, height = 11, label = "volume fraction")
    # row 3 — decoded minus truth, PCAT only
    dif = fill(NaN, size(lab))
    for i in eachindex(lab)
        (pcat[i] && !isnan(DE[i]) && !isnan(GT[i])) && (dif[i] = DE[i] - GT[i])
    end
    ax3 = CM.Axis(fig[3, col]; aspect = CM.DataAspect(),
        title = "decoded − truth, PCAT only\nmean $(round(100*mean(filter(!isnan, dif)), digits=1)) %",
        titlesize = 15)
    CM.heatmap!(ax3, disp(crop(gtw) .* 0 .+ 0.0); colormap = [CM.RGBf(.15,.15,.15)], colorrange = (0,1))
    hd = CM.heatmap!(ax3, disp(crop(dif)); colormap = :balance, colorrange = (-0.4, 0.4),
                     nan_color = :transparent)
    CM.hidedecorations!(ax3)
    col == 3 && CM.Colorbar(fig[3, 4], hd; label = "decoded − truth", width = 12)
    global LASTHD = hd
end
CM.Label(fig[0, :],
    "PCAT decomposition: ground truth vs decoded map — slice z=$Z, " *
    "$(round(PX_MM, digits=3)) mm/px, 1 mm slice, no iodine, poly2 + coupled Huber-TV (lambda=$LAMTAG)";
    fontsize = 18, font = :bold)
CM.save(joinpath(OUT, "pcat_maps_gt_vs_decoded_lam$(LAMTAG).png"), fig; px_per_unit = 2)
println("figure -> ", joinpath(OUT, "pcat_maps_gt_vs_decoded_lam$(LAMTAG).png"))

for (s, name, GT, DE) in MATS
    d = [DE[i] - GT[i] for i in eachindex(lab) if pcat[i] && !isnan(DE[i]) && !isnan(GT[i])]
    @printf("%-8s n=%6d  mean(decoded-truth) = %+6.3f   median = %+6.3f\n",
            name, length(d), mean(d), median(d))
end
