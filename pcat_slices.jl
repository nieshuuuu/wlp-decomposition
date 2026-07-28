# Multi-slice montage: CT, ground-truth lipid, decoded lipid, and the difference, across z.
#
# ORIENTATION. Measured on the recon grid, not assumed: sternum (label 3) sits at mean j = 29.5
# and the vertebra (label 9) at j = 454.2, so LOW j is anterior. CairoMakie's heatmap draws M[x,y]
# with the FIRST index horizontal and the SECOND vertical, origin bottom-left, so the raw array
# renders anterior-DOWN. ImageJ shows these raws anterior-UP, so the second index is reversed.
#
# Do NOT use permutedims for this: it swaps the axes and ROTATES the slice 90 degrees. Verified
# by tracking the landmarks through each candidate transform — permutedims put sternum/vertebra
# at horizontal 483.5/58.8 (i.e. left-right), while reverse(dims=2) puts them at vertical
# 483.5/58.8, which is the intended anterior-up.
import BasisSimulator as BS
import CairoMakie as CM
import TOML
using Statistics: mean, median
using Serialization: deserialize
using Printf: @printf
include(joinpath(@__DIR__, "wlp_tv.jl"))
include(joinpath(@__DIR__, "pcat_orient.jl"))

const OUT = joinpath(@__DIR__, "pcat_ct")
const D = deserialize(joinpath(OUT, get(ENV, "PCAT_ACQ", "pcat_acq_shell.jls")))
const MODEL = TOML.parsefile(joinpath(@__DIR__, "wlp_model_70_150.toml"))
const K, VOXMM = 6, 0.5
const RECON_N, RECON_NZ, RECON_FOV_CM = 512, 40, 18.0
const PX_MM = RECON_FOV_CM * 10 / RECON_N
const WALL0, LUM0 = 76, 82
const CW = Float64.(MODEL["poly2"]["cw"]); const CL = Float64.(MODEL["poly2"]["cl"])
const CP = Float64.(MODEL["poly2"]["cp"])
const GLO = Float64(MODEL["gate"]["soft_hu_lo"]); const GHI = Float64(MODEL["gate"]["soft_hu_hi"])
const TVIT = Int(MODEL["tv"]["iters"]); const TVEPS = Float64(MODEL["tv"]["eps"])
const NSLICE = 6

decode_raw(a, b) = begin
    bb = (1.0, a, b, a^2, b^2, a*b)
    f = (sum(CW .* bb), sum(CL .* bb), sum(CP .* bb)); s = sum(f)
    abs(s) < 1e-12 ? (NaN, NaN, NaN) : (f[1]/s, f[2]/s, f[3]/s)
end

stub = Dict{Int,Any}(Int(l)=>BS.XA.Materials.water for l in unique(D.slab))
m3 = to_clinical_z(BS.resample_to_recon(BS.Phantom(D.slab, stub, (VOXMM/10,VOXMM/10,VOXMM/10)),
                                        D.geom, (RECON_N,RECON_N,RECON_NZ); method=:nearest))
const HU70V = to_clinical_z(Float64.(D.hu_lo))     # flipped with the label map, in one place
const HU150V = to_clinical_z(Float64.(D.hu_hi))
nz = size(m3,3)
myo = [let i=findall(x->15<=Int(x)<=18, m3[:,:,z]); isempty(i) ? -Inf : mean(HU70V[:,:,z][i]) end for z in 1:nz]
plv = let v = sort(filter(isfinite, myo)); median(v[(length(v)÷2+1):end]) end
gd = [z for z in 1:nz if isfinite(myo[z]) && abs(myo[z]-plv)<=8.0]
ZR = (minimum(gd)+1):(maximum(gd)-1)
# The volume is already in clinical z (slice 1 = most cranial), so ascending z IS cranial->caudal.
ZS = round.(Int, range(first(ZR), last(ZR), NSLICE))
@info "valid z $ZR (clinical z: 1 = cranial); showing slices $ZS"

# crop once, on the union of adipose across the shown slices
isfat(l) = (40 <= l < LUM0) || (88 <= l <= 127)
roi = falses(RECON_N, RECON_N)
for z in ZS, c in CartesianIndices(view(m3,:,:,z)); isfat(Int(m3[c,z])) && (roi[c] = true); end
ys, xs = getindex.(findall(roi),1), getindex.(findall(roi),2)
pad = 18
r1,r2 = max(minimum(ys)-pad,1), min(maximum(ys)+pad,RECON_N)
c1,c2 = max(minimum(xs)-pad,1), min(maximum(xs)+pad,RECON_N)
crop(A) = A[r1:r2, c1:c2]

fig = CM.Figure(size = (1180, 210 * NSLICE + 120))
for (row, z) in enumerate(ZS)
    lab = m3[:,:,z]
    h70 = HU70V[:,:,z]; h150 = HU150V[:,:,z]
    gtl = fill(NaN, size(lab))
    for i in eachindex(lab)
        l = Int(lab[i]); haskey(D.gt, l) && (gtl[i] = D.gt[l][2])
    end
    rl = zeros(size(lab)); rp = zeros(size(lab))
    wl = zeros(size(lab)); wp = zeros(size(lab)); val = falses(size(lab))
    for i in eachindex(lab)
        (GLO <= h70[i] <= GHI) || continue
        f0, σ = wlp_sigma_f(h70[i], h150[i], MODEL, decode_raw)
        any(isnan, f0) && continue
        rl[i]=f0[2]; rp[i]=f0[3]; wl[i]=1/σ[2]^2; wp[i]=1/σ[3]^2; val[i]=true
    end
    tl, tp = wlp_tv!(rl, rp, wl, wp, val; λ = wlp_lambda_map(MODEL, rl, rp, val),
                     iters = TVIT, eps = TVEPS)
    del = fill(NaN, size(lab))
    for i in eachindex(lab); val[i] && (del[i] = wlp_simplex(tl[i], tp[i])[2]); end
    dif = fill(NaN, size(lab))
    for i in eachindex(lab)
        (!isnan(gtl[i]) && !isnan(del[i])) && (dif[i] = del[i] - gtl[i])
    end
    nvalid = count(!isnan, dif)
    mb = nvalid > 0 ? 100*mean(filter(!isnan, dif)) : NaN

    panels = ((crop(h70), :grays, (-200.0, 150.0), "CT 70 keV"),
              (crop(gtl), :jet, (0.0, 1.0), "GT lipid"),
              (crop(del), :jet, (0.0, 1.0), "decoded lipid"),
              (crop(dif), :balance, (-0.4, 0.4), "decoded − truth"))
    for (col, (A, cmap, rng, ttl)) in enumerate(panels)
        ax = CM.Axis(fig[row, col]; aspect = CM.DataAspect(),
            title = row == 1 ? ttl : "", titlesize = 15,
            ylabel = col == 1 ? (row == 1 ? "z = $z  (cranial)" :
                                 row == NSLICE ? "z = $z  (caudal)" : "z = $z") : "",
            ylabelsize = 13)
        col > 1 && CM.heatmap!(ax, disp(crop(h70)); colormap = :grays,
                               colorrange = (-200.0, 150.0))
        CM.heatmap!(ax, disp(A); colormap = cmap, colorrange = rng,
                    nan_color = :transparent)
        CM.hidedecorations!(ax; label = false)
        col == 4 && CM.text!(ax, 0.03, 0.03; space = :relative, align = (:left, :bottom),
            text = "mean $(round(mb, digits=1)) %  (n=$nvalid)", fontsize = 11, color = :black)
    end
end
CM.Label(fig[0, :],
    "PCAT lipid: ground truth vs decoded, $(NSLICE) slices through the heart — " *
    "anterior UP (sternum j≈30, vertebra j≈454), $(round(PX_MM,digits=3)) mm/px, 1 mm slices";
    fontsize = 16, font = :bold)
CM.save(joinpath(OUT, "pcat_slices_lipid.png"), fig; px_per_unit = 2)
println("figure -> ", joinpath(OUT, "pcat_slices_lipid.png"))
