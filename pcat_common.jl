# Shared base for every PCAT analysis script: the recon grid, the valid-z window, the poly2 -> TV ->
# simplex decode, and the geometry helpers. Extracted because pcat_20layer.jl, pcat_maps.jl,
# pcat_slices.jl and pcat_radial_profile.jl had all grown their own copy of the same 60 lines — the
# fourth copy is where "three near-duplicates" stops being an argument for leaving it alone.
#
# Contract: this file defines CONSTANTS and FUNCTIONS only. It reads no ARGS and no ENV, and it
# computes nothing at include time, so a caller wires the pieces in whatever order it needs (the
# valid-z trim, for instance, differs between the radial-profile and 20-layer scripts and that
# difference is now visible at the call site instead of buried in a copy).
import BasisSimulator as BS
import TOML
using Statistics: mean, std, median
using Serialization: deserialize
using Unitful: @u_str

const OUT = joinpath(@__DIR__, "pcat_ct")

# ── phantom / recon geometry (SSoT: matches pcat_ct_sim.jl) ───────────────────────────
const K, VOXMM, NSHELL = 6, 0.5, 20
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
const VGROUP = Dict("rca1"=>"healthy", "lcx"=>"healthy", "lad1"=>"healthy",
                    "lad2"=>"diseased", "rca2"=>"diseased", "lad3"=>"diseased")
const RECON_N, RECON_NZ, RECON_FOV_CM, RECON_Z_CM = 512, 40, 18.0, 4.0
const PX_MM = RECON_FOV_CM * 10 / RECON_N          # 0.3516 mm in plane
const DZ_MM = RECON_Z_CM  * 10 / RECON_NZ          # 1.0 mm per slice
const WALL0, LUM0 = 76, 82
const AORTA = 28
const ADIPOSE_LO, ADIPOSE_HI = -190.0, -30.0       # the FAI definition's gate (Antonopoulos 2017)
const SHELL0 = Dict("healthy"=>88, "diseased"=>108)
fat_label(k, i) = 40 + 6 * (K - k) + i             # subring k (1 = innermost), vessel i (0-based)
shell_label(g, k) = SHELL0[g] + k - 1
vessel_solid_labels(vi) = (WALL0 + vi - 1, LUM0 + vi - 1)   # vi is 1-based into VESSELS

pcat_load(acq = "pcat_acq_shell.jls") = deserialize(joinpath(OUT, acq))
pcat_model() = TOML.parsefile(joinpath(@__DIR__, "wlp_model_70_150.toml"))

# ── truth HU per label, from the phantom's own stored ground-truth fractions ──────────
const WM = BS.XA.Materials.basis_water
const LM = BS.XA.Materials.basis_lipid
const PM = BS.XA.Material("p", 0.0, 0.0u"eV", 1.35u"g/cm^3",
    Dict{Int,Float64}(1=>0.066, 6=>0.534, 7=>0.170, 8=>0.220, 16=>0.010))
ρv(m) = BS.XA.val(m.density)
function wlpmat(fw, fl, fp)
    r = fw*ρv(WM) + fl*ρv(LM) + fp*ρv(PM)
    mf = (fw*ρv(WM)/r, fl*ρv(LM)/r, fp*ρv(PM)/r); c = Dict{Int,Float64}()
    for (m,x) in ((WM,mf[1]),(LM,mf[2]),(PM,mf[3])), (Z,f) in m.composition
        c[Z] = get(c,Z,0.0) + x*f
    end
    BS.XA.Material("x", 0.0, 0.0u"eV", r*u"g/cm^3", c)
end
huof(m, E) = (mw = BS.compute_μ_at_energy(WM, E); 1000*(BS.compute_μ_at_energy(m,E)-mw)/mw)
pcat_truth_hu(D, E = 70.0) = Dict(l => huof(wlpmat(f...), E) for (l,f) in D.gt)

# ── label grid on the recon lattice ──────────────────────────────────────────────────
function pcat_label_grid(D)
    stub = Dict{Int,Any}(Int(l) => BS.XA.Materials.water for l in unique(D.slab))
    BS.resample_to_recon(BS.Phantom(D.slab, stub, (VOXMM/10, VOXMM/10, VOXMM/10)),
                         D.geom, (RECON_N, RECON_N, RECON_NZ); method = :nearest)
end

"""
    pcat_slab_z_of_recon(D) -> Vector{Int}

Which SLAB slice fed each recon slice. Measured, not derived from the FOV arithmetic: a slab whose
every voxel carries its own slice index is pushed through the same nearest-neighbour resample, so the
answer is whatever the resampler actually did. Needed because the recon is a zoomed sub-FOV — slab
and recon share neither voxel pitch nor origin, so no coordinate can be compared between them
directly. Returns 0 where a recon slice drew from no slab slice.
"""
function pcat_slab_z_of_recon(D)
    nzs = size(D.slab, 3)
    nzs <= 255 || error("slab has $nzs slices; the index trick needs <= 255 to fit UInt8")
    idx = Array{UInt8}(undef, size(D.slab))
    for k in 1:nzs; idx[:,:,k] .= UInt8(k); end
    stub = Dict{Int,Any}(k => BS.XA.Materials.water for k in 0:nzs)
    r = BS.resample_to_recon(BS.Phantom(idx, stub, (VOXMM/10, VOXMM/10, VOXMM/10)),
                             D.geom, (RECON_N, RECON_N, RECON_NZ); method = :nearest)
    [let v = filter(>(0), Int.(vec(r[:,:,z]))); isempty(v) ? 0 : round(Int, median(v)) end
     for z in 1:RECON_NZ]
end

"""
    pcat_valid_z(D, m3; trim = 1) -> (ZR, plateau)

Slices where the myocardium sits on its own HU plateau. FDK truncation corrupts the edge slices, so
`trim` further drops that many slices from each end of the plateau (the 20-layer analysis uses 1,
the radial profile uses 0 — the difference is deliberate and belongs at the call site).
"""
function pcat_valid_z(D, m3; trim::Int = 1)
    nz = size(m3, 3)
    myo = [let i = findall(x -> 15 <= Int(x) <= 18, m3[:,:,z])
               isempty(i) ? -Inf : mean(Float64.(D.hu_lo[:,:,z])[i]) end for z in 1:nz]
    plateau = let v = sort(filter(isfinite, myo)); median(v[(length(v)÷2+1):end]) end
    gd = [z for z in 1:nz if isfinite(myo[z]) && abs(myo[z] - plateau) <= 8.0]
    ((minimum(gd)+trim):(maximum(gd)-trim), plateau)
end

# ── poly2 -> noise-weighted coupled Huber-TV -> simplex, once per slice ──────────────
include(joinpath(@__DIR__, "wlp_tv.jl"))

"Build the poly2 raw decoder and the gate/TV parameters out of a parsed model card."
function pcat_decoder(MODEL)
    cw = Float64.(MODEL["poly2"]["cw"]); cl = Float64.(MODEL["poly2"]["cl"])
    cp = Float64.(MODEL["poly2"]["cp"])
    raw = (a, b) -> begin
        bb = (1.0, a, b, a^2, b^2, a*b)
        f = (sum(cw .* bb), sum(cl .* bb), sum(cp .* bb)); s = sum(f)
        abs(s) < 1e-12 ? (NaN, NaN, NaN) : (f[1]/s, f[2]/s, f[3]/s)
    end
    (raw = raw, glo = Float64(MODEL["gate"]["soft_hu_lo"]), ghi = Float64(MODEL["gate"]["soft_hu_hi"]),
     iters = Int(MODEL["tv"]["iters"]), eps = Float64(MODEL["tv"]["eps"]))
end

"""
    pcat_decode(H70, H150, MODEL) -> (FW, FL, FP)

Per-slice poly2 decode, noise-weighted coupled Huber-TV on the lipid/protein pair, then one simplex
projection. NaN wherever the soft HU gate rejects the voxel. This is the delivered reader; do not
substitute a linear endpoint inversion (it is exact at truth HU and worse on real images, because
poly2 is what carries the acquisition chain's differential bias).
"""
function pcat_decode(H70, H150, MODEL)
    d = pcat_decoder(MODEL)
    FW = fill(NaN, size(H70)); FL = fill(NaN, size(H70)); FP = fill(NaN, size(H70))
    for z in axes(H70, 3)
        h70 = @view H70[:,:,z]; h150 = @view H150[:,:,z]
        rl = zeros(size(h70)); rp = zeros(size(h70))
        wl = zeros(size(h70)); wp = zeros(size(h70)); val = falses(size(h70))
        for i in eachindex(h70)
            (d.glo <= h70[i] <= d.ghi) || continue
            f0, σ = wlp_sigma_f(h70[i], h150[i], MODEL, d.raw)
            any(isnan, f0) && continue
            rl[i]=f0[2]; rp[i]=f0[3]; wl[i]=1/σ[2]^2; wp[i]=1/σ[3]^2; val[i]=true
        end
        tl, tp = wlp_tv!(rl, rp, wl, wp, val;
                         λ = wlp_lambda_map(MODEL, rl, rp, val), iters = d.iters, eps = d.eps)
        fwz = @view FW[:,:,z]; flz = @view FL[:,:,z]; fpz = @view FP[:,:,z]
        for i in eachindex(h70)
            val[i] || continue
            fwz[i], flz[i], fpz[i] = wlp_simplex(tl[i], tp[i])
        end
    end
    (FW, FL, FP)
end

# ── geometry helpers ─────────────────────────────────────────────────────────────────
"In-plane binary erosion by `n` 4-neighbour passes. Border pixels are cleared, not wrapped."
function erode2(mask::BitMatrix, n::Int)
    n <= 0 && return mask
    m = copy(mask)
    for _ in 1:n
        p = copy(m)
        @inbounds for j in 2:size(m,2)-1, i in 2:size(m,1)-1
            m[i,j] = p[i,j] & p[i-1,j] & p[i+1,j] & p[i,j-1] & p[i,j+1]
        end
        m[1,:] .= false; m[end,:] .= false; m[:,1] .= false; m[:,end] .= false
    end
    m
end

"""
    edt2(mask, px) -> distance

Chamfer (cityblock) in-plane distance to the nearest true pixel of `mask`, in mm. IN-PLANE ONLY:
pixels are 0.35 mm but slices are 1 mm, so an isotropic 3-D transform would fold a 3x coarser axis
into the radial coordinate.
"""
function edt2(mask::BitMatrix, px)
    n, m = size(mask); INF = 1e12
    d = [mask[i,j] ? 0.0 : INF for i in 1:n, j in 1:m]
    for j in 1:m, i in 2:n;      d[i,j] = min(d[i,j], d[i-1,j] + px); end
    for j in 1:m, i in n-1:-1:1; d[i,j] = min(d[i,j], d[i+1,j] + px); end
    for i in 1:n, j in 2:m;      d[i,j] = min(d[i,j], d[i,j-1] + px); end
    for i in 1:n, j in m-1:-1:1; d[i,j] = min(d[i,j], d[i,j+1] + px); end
    d
end

"""
    pcat_fat_erode(lab, n) -> BitArray

Erode the union of ALL adipose labels against non-fat tissue only. Shell-to-shell boundaries carry
no meaningful contrast (adjacent shells differ by ~0.005 in lipid fraction), so eroding between them
would only throw pixels away.
"""
function pcat_fat_erode(lab, n::Int)
    allfat = ((lab .>= 40) .& (lab .<= 75)) .| ((lab .>= 88) .& (lab .<= 127)) .| (lab .== 29)
    out = falses(size(allfat))
    for z in axes(allfat, 3); out[:,:,z] = erode2(BitMatrix(allfat[:,:,z]), n); end
    (mask = out, n_before = count(allfat), n_after = count(out))
end

"Lin's concordance correlation coefficient."
ccc(x, y) = begin
    mx, my = mean(x), mean(y)
    2mean((x .- mx) .* (y .- my)) / (mean((x .- mx).^2) + mean((y .- my).^2) + (mx - my)^2)
end

"Standard error of the mean — NOT the per-voxel standard deviation. Both get reported; never merge."
sem(v) = std(v) / sqrt(length(v))
