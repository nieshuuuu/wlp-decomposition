# Score the phantom the way the Oxford method actually scores a patient.
#
# Two changes from the earlier analysis, both taken straight from Antonopoulos 2017 (Sci Transl
# Med) methods and the 2021 virtual guide:
#
#  1. ADIPOSE GATE. "Adipose tissue was defined as all voxels with attenuation between -190 and
#     -30 HU ... FAI was defined as the average attenuation of the adipose tissue volume of
#     interest (within the prespecified window)." The ROI is HU-gated, not purely geometric.
#     This is an implicit partial-volume rejection: wall-contaminated voxels read above -30 and
#     drop out.
#
#  2. TWO ROI DEFINITIONS, not one.
#       FAI_PVAT  — "a layer of tissue within a radial distance from the outer coronary artery
#                   wall equal to the average diameter of the tracked segment". ONE thick layer.
#                   This is the number used clinically.
#       gradient  — "20 concentric cylindrical 1-mm-thick layers". Only for the radial curve.
#
# Truth is reported two ways, because they answer different questions:
#   truth_gated — mean true HU over the voxels the gate actually kept  -> selection bias
#   truth_geom  — mean true HU over the whole geometric layer          -> total bias
import BasisSimulator as BS
import CairoMakie as CM
import TOML
using Statistics: mean, std, cor, median
using Printf: @printf
using Serialization: deserialize
using Unitful: @u_str
include(joinpath(@__DIR__, "wlp_tv.jl"))

const OUT = joinpath(@__DIR__, "pcat_ct")
const ACQ = get(ENV, "PCAT_ACQ", "pcat_acq_tissue.jls")
const D = deserialize(joinpath(OUT, ACQ))
const MODEL = TOML.parsefile(joinpath(@__DIR__, "..", "wlp_model_70_150.toml"))
const K, VOXMM = 6, 0.5
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
const VGROUP = Dict("rca1"=>"healthy","lcx"=>"healthy","lad1"=>"healthy",
                    "lad2"=>"diseased","rca2"=>"diseased","lad3"=>"diseased")
fat_label(k, i) = 40 + 6 * (K - k) + i
const RECON_N, RECON_NZ, RECON_FOV_CM = 512, 40, 18.0
const PX_MM = RECON_FOV_CM * 10 / RECON_N
const WALL0, LUM0 = 76, 82
const ADIPOSE_LO, ADIPOSE_HI = -190.0, -30.0      # Antonopoulos 2017
const CW = Float64.(MODEL["poly2"]["cw"]); const CL = Float64.(MODEL["poly2"]["cl"])
const CP = Float64.(MODEL["poly2"]["cp"])
const TVIT = Int(MODEL["tv"]["iters"]); const TVEPS = Float64(MODEL["tv"]["eps"])
const GATE_LO = Float64(MODEL["gate"]["soft_hu_lo"]); const GATE_HI = Float64(MODEL["gate"]["soft_hu_hi"])
# λ is not a fixed number: the model card's [tv.lambda_model] evaluates it per voxel at the
# raw decode (wlp_lambda_map, per slice below).

decode_raw(hlo, hhi) = begin
    b = (1.0, hlo, hhi, hlo^2, hhi^2, hlo * hhi)
    f = (sum(CW .* b), sum(CL .* b), sum(CP .* b)); s = sum(f)
    abs(s) < 1e-12 ? (NaN, NaN, NaN) : (f[1]/s, f[2]/s, f[3]/s)
end

# ── truth LUT ─────────────────────────────────────────────────────────────────────────
const WM = BS.XA.Materials.basis_water; const LM = BS.XA.Materials.basis_lipid
const PM = BS.XA.Material("p", 0.0, 0.0u"eV", 1.35u"g/cm^3",
    Dict{Int,Float64}(1=>0.066, 6=>0.534, 7=>0.170, 8=>0.220, 16=>0.010))
ρv(m) = BS.XA.val(m.density)
function wlpmat(fw, fl, fp)
    r = fw*ρv(WM) + fl*ρv(LM) + fp*ρv(PM)
    mf = (fw*ρv(WM)/r, fl*ρv(LM)/r, fp*ρv(PM)/r); c = Dict{Int,Float64}()
    for (m, x) in ((WM,mf[1]),(LM,mf[2]),(PM,mf[3])), (Z,f) in m.composition
        c[Z] = get(c,Z,0.0) + x*f
    end
    BS.XA.Material("x", 0.0, 0.0u"eV", r*u"g/cm^3", c)
end
huof(m, E) = (mw = BS.compute_μ_at_energy(WM, E); 1000*(BS.compute_μ_at_energy(m,E)-mw)/mw)
truthHU = Dict{Int,Float64}()
for (i,v) in enumerate(VESSELS), k in 1:K
    l = fat_label(k, i-1); truthHU[l] = huof(wlpmat(D.gt[l]...), 70.0)
end

# ── grids + valid z ───────────────────────────────────────────────────────────────────
stub = Dict{Int,Any}(Int(l)=>BS.XA.Materials.water for l in unique(D.slab))
m3 = BS.resample_to_recon(BS.Phantom(D.slab, stub, (VOXMM/10,VOXMM/10,VOXMM/10)),
                          D.geom, (RECON_N,RECON_N,RECON_NZ); method=:nearest)
nz = size(m3,3)
myo = [let i=findall(x->15<=Int(x)<=18, m3[:,:,z]); isempty(i) ? -Inf : mean(Float64.(D.hu_lo[:,:,z])[i]) end for z in 1:nz]
pl = let v = sort(filter(isfinite, myo)); median(v[(length(v)÷2+1):end]) end
ZR = let g=[z for z in 1:nz if isfinite(myo[z]) && abs(myo[z]-pl)<=8.0]; minimum(g):maximum(g) end
lab = m3[:,:,ZR]; H70 = Float64.(D.hu_lo[:,:,ZR]); H150 = Float64.(D.hu_hi[:,:,ZR])
@info "valid z $ZR, $(round(PX_MM,digits=4)) mm/px"

# ── decoded W/L/P maps: poly2 -> noise-weighted coupled Huber-TV -> simplex once ──────
FW = fill(NaN,size(lab)); FL = fill(NaN,size(lab)); FP = fill(NaN,size(lab))
for z in axes(lab,3)
    h70 = @view H70[:,:,z]; h150 = @view H150[:,:,z]
    rl = zeros(size(h70)); rp = zeros(size(h70))
    wl = zeros(size(h70)); wp = zeros(size(h70)); val = falses(size(h70))
    for i in eachindex(h70)
        (GATE_LO <= h70[i] <= GATE_HI) || continue
        f0, σ = wlp_sigma_f(h70[i], h150[i], MODEL, decode_raw)
        any(isnan, f0) && continue
        rl[i]=f0[2]; rp[i]=f0[3]; wl[i]=1/σ[2]^2; wp[i]=1/σ[3]^2; val[i]=true
    end
    tl, tp = wlp_tv!(rl, rp, wl, wp, val; λ=wlp_lambda_map(MODEL, rl, rp, val), iters=TVIT, eps=TVEPS)
    fwz = @view FW[:,:,z]; flz = @view FL[:,:,z]; fpz = @view FP[:,:,z]
    for i in eachindex(h70)
        val[i] || continue
        fwz[i], flz[i], fpz[i] = wlp_simplex(tl[i], tp[i])   # once, after TV: every f in [0,1]
    end
end

# ── per-vessel diameter (from lumen area) and the FAI_PVAT layer ──────────────────────
function edt2(mask::BitMatrix, px)
    n,m = size(mask); d = [mask[i,j] ? 0.0 : 1e12 for i in 1:n, j in 1:m]
    for j in 1:m, i in 2:n;      d[i,j]=min(d[i,j], d[i-1,j]+px); end
    for j in 1:m, i in n-1:-1:1; d[i,j]=min(d[i,j], d[i+1,j]+px); end
    for i in 1:n, j in 2:m;      d[i,j]=min(d[i,j], d[i,j-1]+px); end
    for i in 1:n, j in m-1:-1:1; d[i,j]=min(d[i,j], d[i,j+1]+px); end
    d
end
diam = Dict{String,Float64}()
for (vi,v) in enumerate(VESSELS)
    a = [count(==(UInt8(LUM0+vi-1)), lab[:,:,z]) for z in axes(lab,3)]
    a = filter(>(3), a); isempty(a) && continue
    diam[v] = 2*sqrt(median(a)*PX_MM^2/π)       # equivalent circular diameter, mm
end

println("\n", "="^100)
println("CLINICAL FAI_PVAT ROI — one layer, thickness = vessel diameter, adipose-gated [-190,-30] HU")
println("="^100)
@printf("%-6s %8s %8s %9s %8s %10s %10s %8s %8s\n",
        "vessel","diam_mm","n_geom","n_gated","kept_%","meas_FAI","truth_gate","truth_geo","bias")
fai = NamedTuple[]
for (vi,v) in enumerate(VESSELS)
    haskey(diam,v) || continue
    Dv = diam[v]
    gi = CartesianIndex{3}[]
    for z in axes(lab,3)
        L = @view lab[:,:,z]
        solid = BitMatrix((L .== UInt8(WALL0+vi-1)) .| (L .== UInt8(LUM0+vi-1)))
        count(solid) < 5 && continue
        dr = edt2(solid, PX_MM)
        for c in CartesianIndices(L)
            (0 < dr[c] <= Dv) || continue
            haskey(truthHU, Int(L[c])) || continue        # only this vessel's own PCAT
            push!(gi, CartesianIndex(c[1], c[2], z))
        end
    end
    isempty(gi) && continue
    gg = [i for i in gi if ADIPOSE_LO <= H70[i] <= ADIPOSE_HI]
    length(gg) < 20 && continue
    tg = mean(truthHU[Int(lab[i])] for i in gg)
    tG = mean(truthHU[Int(lab[i])] for i in gi)
    push!(fai, (vessel=v, group=VGROUP[v], diam=Dv, n=length(gi), ng=length(gg),
                meas=mean(H70[gg]), tgate=tg, tgeom=tG,
                fw=mean(filter(!isnan,[FW[i] for i in gg])),
                fl=mean(filter(!isnan,[FL[i] for i in gg])),
                fp=mean(filter(!isnan,[FP[i] for i in gg])),
                gw=mean(D.gt[Int(lab[i])][1] for i in gg),
                gl=mean(D.gt[Int(lab[i])][2] for i in gg),
                gp=mean(D.gt[Int(lab[i])][3] for i in gg)))
    f = fai[end]
    @printf("%-6s %8.2f %8d %9d %8.1f %10.1f %10.1f %10.1f %+8.1f\n",
            v, Dv, f.n, f.ng, 100f.ng/f.n, f.meas, f.tgate, f.tgeom, f.meas-f.tgate)
end
@printf("\nmean FAI bias vs gated truth: %+.2f HU   vs geometric truth: %+.2f HU\n",
        mean(f.meas-f.tgate for f in fai), mean(f.meas-f.tgeom for f in fai))

println("\nW/L/P over the SAME clinical FAI_PVAT ROI (volume fraction %):")
@printf("%-6s %10s %10s %10s %10s %10s %10s\n","vessel","GT_water","ms_water","GT_lipid","ms_lipid","GT_prot","ms_prot")
for f in fai
    @printf("%-6s %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n",
            f.vessel, 100f.gw, 100f.fw, 100f.gl, 100f.fl, 100f.gp, 100f.fp)
end
for (nm,g,m) in (("water",[f.gw for f in fai],[f.fw for f in fai]),
                 ("lipid",[f.gl for f in fai],[f.fl for f in fai]),
                 ("protein",[f.gp for f in fai],[f.fp for f in fai]))
    @printf("  %-8s bias %+6.2f %%   RMSE %5.2f %%\n", nm, 100mean(m.-g), 100sqrt(mean((m.-g).^2)))
end

# ── 1 mm gradient layers, adipose-gated ───────────────────────────────────────────────
println("\n", "="^100)
println("1 mm GRADIENT LAYERS — adipose-gated (Oxford uses these only for the radial curve)")
println("="^100)
@printf("%-6s %4s %8s %9s %8s %9s %9s %8s %8s %8s\n",
        "vessel","sub","n_geom","n_gated","kept_%","meas_HU","truth_HU","GTlip%","mslip%","d_lip%")
rows = NamedTuple[]
for (vi,v) in enumerate(VESSELS), k in 1:K
    l = fat_label(k, vi-1)
    idx = findall(==(UInt8(l)), lab); length(idx) < 20 && continue
    g = [i for i in idx if ADIPOSE_LO <= H70[i] <= ADIPOSE_HI]
    length(g) < 20 && continue
    t = truthHU[l]; gt = D.gt[l]
    mw = mean(filter(!isnan,[FW[i] for i in g])); ml = mean(filter(!isnan,[FL[i] for i in g]))
    mp = mean(filter(!isnan,[FP[i] for i in g]))
    push!(rows,(vessel=v,group=VGROUP[v],sub=k,n=length(idx),ng=length(g),
                meas=mean(H70[g]),truth=t,gw=gt[1],gl=gt[2],gp=gt[3],fw=mw,fl=ml,fp=mp))
    r = rows[end]
    @printf("%-6s %4d %8d %9d %8.1f %9.1f %9.1f %8.1f %8.1f %+8.1f\n",
            v,k,r.n,r.ng,100r.ng/r.n,r.meas,r.truth,100r.gl,100r.fl,100(r.fl-r.gl))
end
function ccc(x, y)
    mx, my = mean(x), mean(y)
    2mean((x .- mx) .* (y .- my)) / (mean((x .- mx).^2) + mean((y .- my).^2) + (mx - my)^2)
end
println("\ngradient-layer W/L/P accuracy (adipose-gated, n=$(length(rows)) layers):")
@printf("%-9s %8s %8s %8s %8s\n","material","CCC","RMSE_%","bias_%","GTrange_%")
for (nm,gk,mk) in (("Water",:gw,:fw),("Lipid",:gl,:fl),("Protein",:gp,:fp))
    g=[r[gk] for r in rows]; m=[r[mk] for r in rows]
    @printf("%-9s %8.4f %8.2f %+8.2f %8.2f\n", nm, ccc(g,m), 100sqrt(mean((m.-g).^2)),
            100mean(m.-g), 100(maximum(g)-minimum(g)))
end
