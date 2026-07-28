# 1. Export the reconstructed CT as .raw so it can be opened in a viewer.
# 2. Answer whether the subring ROI should be eroded off the boundary, by measuring it.
#
# Two erosion variants, because they are NOT the same thing here:
#   per-subring : erode each subring's own mask. Also strips the boundaries BETWEEN adjacent
#                 subrings — but neighbouring subrings differ by only ~0.01 in lipid fraction,
#                 so that contamination is negligible and this just throws pixels away.
#   whole-cuff  : erode the union of all fat, then intersect with each subring. Strips only the
#                 voxels near NON-fat tissue (vessel wall inside, myocardium/background outside),
#                 which is where the +40 HU contamination actually comes from.
import BasisSimulator as BS
using Statistics: mean, std, cor
using Printf: @printf
using Serialization: deserialize
using Unitful: @u_str

const OUT = joinpath(@__DIR__, "pcat_ct")
const ACQ = get(ENV, "PCAT_ACQ", "pcat_acq_tissue.jls")
const D = deserialize(joinpath(OUT, ACQ))
const K, VOXMM = 6, 0.5
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
fat_label(k, i) = 40 + 6 * (K - k) + i
const RECON_N, RECON_FOV_CM, RECON_NZ = 512, 18.0, 40
const PX_MM = RECON_FOV_CM * 10 / RECON_N

# ── 1. export ─────────────────────────────────────────────────────────────────────────
stub = Dict{Int, Any}(Int(l) => BS.XA.Materials.water for l in unique(D.slab))
ph_cpu = BS.Phantom(D.slab, stub, (VOXMM / 10, VOXMM / 10, VOXMM / 10))
m3 = BS.resample_to_recon(ph_cpu, D.geom, (RECON_N, RECON_N, RECON_NZ); method = :nearest)
const SLICE_MM = 4.0 * 10 / RECON_NZ

for (nm, arr) in (("vmi070keV", Float32.(D.hu_lo)), ("vmi150keV", Float32.(D.hu_hi)))
    p = joinpath(OUT, "pcat_$(nm)_$(RECON_N)x$(RECON_N)x$(RECON_NZ)_float32.raw")
    write(p, arr); println("wrote $p  ($(round(filesize(p)/1e6, digits=2)) MB)")
end
let p = joinpath(OUT, "pcat_labels_recon_$(RECON_N)x$(RECON_N)x$(RECON_NZ)_uint8.raw")
    write(p, UInt8.(m3)); println("wrote $p")
end
let p = joinpath(OUT, "pcat_phantom_slab_640x640x$(size(D.slab,3))_uint8.raw")
    write(p, D.slab); println("wrote $p  (0.5mm phantom slab, z $(D.z0)..$(D.z0+size(D.slab,3)-1))")
end
println("recon grid $(RECON_N)^2 @ $(RECON_FOV_CM) cm FOV = $(round(PX_MM, digits=4)) mm/px, " *
        "$(RECON_NZ) slices of $(SLICE_MM) mm\n")

# ── 2. erosion experiment ─────────────────────────────────────────────────────────────
midz = size(m3, 3) ÷ 2 + 1
lab = m3[:, :, midz]
hlo = Float64.(D.hu_lo[:, :, midz])

median_(v) = (s = sort(collect(v)); n = length(s); iseven(n) ? (s[n÷2] + s[n÷2+1]) / 2 : s[(n+1)÷2])

function erode2(mask::BitMatrix, n::Int)
    n <= 0 && return mask
    m = copy(mask)
    for _ in 1:n
        p = copy(m)
        @inbounds for j in 2:size(m, 2)-1, i in 2:size(m, 1)-1
            m[i, j] = p[i, j] & p[i-1, j] & p[i+1, j] & p[i, j-1] & p[i, j+1]
        end
        m[1, :] .= false; m[end, :] .= false; m[:, 1] .= false; m[:, end] .= false
    end
    m
end

# theoretical HU of each subring's own material
const WM = BS.XA.Materials.basis_water
const LM = BS.XA.Materials.basis_lipid
const PM = BS.XA.Material("p", 0.0, 0.0u"eV", 1.35u"g/cm^3",
    Dict{Int, Float64}(1 => 0.066, 6 => 0.534, 7 => 0.170, 8 => 0.220, 16 => 0.010))
ρv(m) = BS.XA.val(m.density)
function wlpmat(fw, fl, fp)
    r = fw * ρv(WM) + fl * ρv(LM) + fp * ρv(PM)
    mf = (fw * ρv(WM) / r, fl * ρv(LM) / r, fp * ρv(PM) / r)
    c = Dict{Int, Float64}()
    for (m, wm) in ((WM, mf[1]), (LM, mf[2]), (PM, mf[3])), (Z, f) in m.composition
        c[Z] = get(c, Z, 0.0) + wm * f
    end
    BS.XA.Material("x", 0.0, 0.0u"eV", r * u"g/cm^3", c)
end
theo(m, E) = (mw = BS.compute_μ_at_energy(WM, E); 1000 * (BS.compute_μ_at_energy(m, E) - mw) / mw)

fatmask = falses(size(lab))
for (vi, _) in enumerate(VESSELS), k in 1:K
    fatmask .|= (lab .== UInt8(fat_label(k, vi - 1)))
end
println("PCAT pixels on the recon grid: $(count(fatmask))  " *
        "(1mm subring = $(round(1/PX_MM, digits=1)) pixels wide)\n")

println("="^94)
println("Does eroding the ROI off the boundary help?   (bias = measured VMI 70 keV minus true HU)")
println("="^94)
@printf("%-12s %6s %8s %9s %9s %9s %9s\n",
        "variant", "erode", "n_ROIs", "n_px_tot", "median_px", "mean_bias", "median_bias")
for variant in ("per-subring", "whole-cuff")
    for e in 0:3
        cuff_e = variant == "whole-cuff" ? erode2(fatmask, e) : nothing
        biases = Float64[]; ns = Int[]
        for (vi, v) in enumerate(VESSELS), k in 1:K
            l = fat_label(k, vi - 1)
            m0 = lab .== UInt8(l)
            count(m0) == 0 && continue
            m = variant == "per-subring" ? erode2(m0, e) : (m0 .& cuff_e)
            n = count(m); n < 10 && continue
            t = theo(wlpmat(D.gt[l]...), 70.0)
            push!(biases, mean(hlo[m]) - t); push!(ns, n)
        end
        isempty(biases) && (@printf("%-12s %6d %8d %9s %9s %9s %9s\n", variant, e, 0, "-", "-", "-", "-"); continue)
        @printf("%-12s %6d %8d %9d %9d %+9.2f %+9.2f\n",
                variant, e, length(biases), sum(ns), round(Int, median_(ns)),
                mean(biases), median_(biases))
    end
end

