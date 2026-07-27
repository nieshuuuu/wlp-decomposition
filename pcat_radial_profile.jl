# HU vs radial distance from the coronary wall — measured against the phantom's own truth.
#
# The profile is taken on a SIGNED distance: negative inside the wall/lumen, positive outward
# into the fat. Every voxel in the band is used regardless of label, so the wall -> fat -> pericardial
# fat transition is visible and the reconstruction's blur can be read off directly.
#
# The truth curve is the phantom's own material HU at each voxel, binned identically — i.e. the
# profile an infinite-resolution scanner would return. The gap between the two curves IS the
# point-spread contamination, resolved as a function of depth.
import BasisSimulator as BS
import CairoMakie as CM
using Statistics: mean, std, median
using Serialization: deserialize
using Unitful: @u_str
using Printf: @printf

const OUT = joinpath(@__DIR__, "pcat_ct")
const D = deserialize(joinpath(OUT, "pcat_acq.jls"))
const K, VOXMM = 6, 0.5
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
const VESSEL_GROUP = Dict("rca1" => "healthy", "lcx" => "healthy", "lad1" => "healthy",
                          "lad2" => "diseased", "rca2" => "diseased", "lad3" => "diseased")
fat_label(k, i) = 40 + 6 * (K - k) + i
const RECON_N, RECON_NZ, RECON_FOV_CM, RECON_Z_CM = 512, 40, 18.0, 4.0
const PX_MM = RECON_FOV_CM * 10 / RECON_N
const WALL0, LUM0 = 76, 82

# ── materials -> theoretical HU per label (the truth curve) ───────────────────────────
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
huof(m, E) = (mw = BS.compute_μ_at_energy(WM, E); 1000 * (BS.compute_μ_at_energy(m, E) - mw) / mw)

function truth_lut(E)
    M = BS.XA.Materials
    lut = Dict{Int, Float64}()
    base = Dict(0 => M.air, 1 => M.softtissue, 2 => M.cartilage, 4 => M.muscle, 6 => M.lung,
                10 => M.liver, 11 => M.kidney, 12 => M.spleen, 29 => M.adipose, 28 => M.wholeblood)
    for l in (3, 5, 8, 9); base[l] = M.corticalbone; end
    for l in 15:18; base[l] = M.heart; end
    for l in 19:22; base[l] = M.wholeblood; end
    for i in 0:5; base[WALL0 + i] = M.muscle; base[LUM0 + i] = M.wholeblood; end
    for (l, m) in base; lut[l] = huof(m, E); end
    for (i, v) in enumerate(VESSELS), k in 1:K
        lut[fat_label(k, i - 1)] = huof(wlpmat(D.gt[fat_label(k, i - 1)]...), E)
    end
    lut
end

# ── recon grid + valid z (myocardium plateau; FDK truncation kills the edge slices) ───
stub = Dict{Int, Any}(Int(l) => BS.XA.Materials.water for l in unique(D.slab))
m3 = BS.resample_to_recon(BS.Phantom(D.slab, stub, (VOXMM/10, VOXMM/10, VOXMM/10)),
                          D.geom, (RECON_N, RECON_N, RECON_NZ); method = :nearest)
nz = size(m3, 3)
myo_hu = [let i = findall(x -> 15 <= Int(x) <= 18, m3[:, :, z])
              isempty(i) ? -Inf : mean(Float64.(D.hu_lo[:, :, z])[i]) end for z in 1:nz]
plateau = median(filter(isfinite, myo_hu[(nz÷2):nz]))
good = [z for z in 1:nz if isfinite(myo_hu[z]) && abs(myo_hu[z] - plateau) <= 8.0]
ZR = minimum(good):maximum(good)
lab = m3[:, :, ZR]
H70 = Float64.(D.hu_lo[:, :, ZR]); H150 = Float64.(D.hu_hi[:, :, ZR])
@info "valid z = $ZR; in-plane $(round(PX_MM, digits=4)) mm, slices $(RECON_Z_CM*10/RECON_NZ) mm"

# ── signed in-plane distance from each vessel's wall+lumen solid ──────────────────────
# In-plane only: pixels are 0.352 mm but slices are 1 mm, so an isotropic 3D transform would
# mix a 3x coarser axis into the radial coordinate.
function edt2(mask::BitMatrix, px)
    n, m = size(mask); INF = 1e12
    d = [mask[i, j] ? 0.0 : INF for i in 1:n, j in 1:m]
    for j in 1:m, i in 2:n;   d[i, j] = min(d[i, j], d[i-1, j] + px); end
    for j in 1:m, i in n-1:-1:1; d[i, j] = min(d[i, j], d[i+1, j] + px); end
    for i in 1:n, j in 2:m;   d[i, j] = min(d[i, j], d[i, j-1] + px); end
    for i in 1:n, j in m-1:-1:1; d[i, j] = min(d[i, j], d[i, j+1] + px); end
    d                                              # chamfer (cityblock) — adequate at this scale
end

const BINW = PX_MM                                  # one pixel per bin
const RMIN, RMAX = -3.0, 9.0
const EDGES = RMIN:BINW:RMAX
nb = length(EDGES) - 1
centers = [(EDGES[i] + EDGES[i+1]) / 2 for i in 1:nb]

lut70 = truth_lut(70.0); lut150 = truth_lut(150.0)

profiles = Dict{String, Any}()
for (vi, v) in enumerate(VESSELS)
    sm = zeros(nb, 4); cnt = zeros(Int, nb)          # meas70, meas150, true70, true150
    sq = zeros(nb)
    for z in axes(lab, 3)
        L = @view lab[:, :, z]
        h70z = @view H70[:, :, z]; h150z = @view H150[:, :, z]
        solid = BitMatrix((L .== UInt8(WALL0 + vi - 1)) .| (L .== UInt8(LUM0 + vi - 1)))
        count(solid) < 5 && continue
        depth_in = edt2(.!solid, PX_MM)      # 0 outside, grows with depth INTO wall/lumen
        dist_out = edt2(solid, PX_MM)        # 0 inside,  grows OUTWARD into the fat
        sgn = dist_out .- depth_in           # positive = outward into the fat
        for idx in eachindex(L)
            r = sgn[idx]
            (RMIN <= r < RMAX) || continue
            b = clamp(floor(Int, (r - RMIN) / BINW) + 1, 1, nb)
            l = Int(L[idx])
            haskey(lut70, l) || continue
            cnt[b] += 1
            sm[b, 1] += h70z[idx];  sm[b, 2] += h150z[idx]
            sm[b, 3] += lut70[l];   sm[b, 4] += lut150[l]
            sq[b] += h70z[idx]^2
        end
    end
    ok = cnt .>= 30
    profiles[v] = (r = centers[ok], n = cnt[ok],
                   m70 = sm[ok, 1] ./ cnt[ok], m150 = sm[ok, 2] ./ cnt[ok],
                   t70 = sm[ok, 3] ./ cnt[ok], t150 = sm[ok, 4] ./ cnt[ok],
                   sd70 = sqrt.(max.(sq[ok] ./ cnt[ok] .- (sm[ok, 1] ./ cnt[ok]) .^ 2, 0.0)))
end

println("\nHU vs radial distance (70 keV), measured / truth")
@printf("%6s", "r_mm"); for v in VESSELS; @printf("%18s", v); end; println()
for (i, r) in enumerate(centers)
    (-1.0 <= r <= 7.0) || continue
    abs(r - round(r * 2) / 2) < BINW / 2 || continue
    @printf("%6.2f", r)
    for v in VESSELS
        p = profiles[v]; j = findfirst(x -> abs(x - r) < BINW / 2, p.r)
        j === nothing ? @printf("%18s", "-") : @printf("%9.1f /%7.1f", p.m70[j], p.t70[j])
    end
    println()
end

# ── figure ────────────────────────────────────────────────────────────────────────────
COL = Dict("rca1" => CM.RGBf(.85, .20, .18), "rca2" => CM.RGBf(.55, .35, .30),
           "lad1" => CM.RGBf(.55, .45, .75), "lad2" => CM.RGBf(.20, .45, .80),
           "lad3" => CM.RGBf(.75, .55, .10), "lcx" => CM.RGBf(.20, .65, .35))
fig = CM.Figure(size = (1560, 760))
for (col, (E, mk, tk, lbl)) in enumerate(((70, :m70, :t70, "70 keV"), (150, :m150, :t150, "150 keV")))
    ax = CM.Axis(fig[1, col];
        title = "$lbl — measured (solid) vs phantom truth (dashed)", titlesize = 18,
        xlabel = "signed radial distance from the coronary wall (mm)",
        ylabel = col == 1 ? "HU" : "", xticks = -2:1:8)
    CM.vlines!(ax, 0.0; color = :gray50, linestyle = :dot)
    CM.text!(ax, 0.02, 5; text = "wall surface", rotation = π/2, fontsize = 11, color = :gray40)
    for v in VESSELS
        haskey(profiles, v) || continue
        p = profiles[v]; isempty(p.r) && continue
        CM.lines!(ax, p.r, getproperty(p, mk); color = COL[v], linewidth = 2.4, label = v)
        CM.lines!(ax, p.r, getproperty(p, tk); color = (COL[v], 0.55), linewidth = 1.6,
                  linestyle = :dash)
    end
    col == 2 && CM.axislegend(ax; position = :rb, framevisible = false, labelsize = 12)
end
CM.Label(fig[0, :],
    "PCAT radial HU profile — the gap between solid and dashed is the reconstruction's " *
    "point-spread contamination, resolved by depth";
    fontsize = 17, font = :bold)
CM.save(joinpath(OUT, "pcat_radial_HU_profile.png"), fig; px_per_unit = 2)
println("\nfigure -> ", joinpath(OUT, "pcat_radial_HU_profile.png"))

open(joinpath(OUT, "pcat_radial_profile.csv"), "w") do io
    println(io, "vessel,group,r_mm,n,measured_HU70,truth_HU70,measured_HU150,truth_HU150,sd_HU70")
    for v in VESSELS, (i, r) in enumerate(profiles[v].r)
        p = profiles[v]
        println(io, "$v,$(VESSEL_GROUP[v]),$r,$(p.n[i]),$(p.m70[i]),$(p.t70[i]),$(p.m150[i]),$(p.t150[i]),$(p.sd70[i])")
    end
end
println("csv -> ", joinpath(OUT, "pcat_radial_profile.csv"))
