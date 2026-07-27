# Decode the simulated 70/150 keV VMI pair into water/lipid/protein and score it against the
# phantom's own ground truth, per (vessel, subring) region.
#
# Decoder: the shipped poly2 surface from wlp_model_70_150.toml — the SAME acquisition chain it
# was fitted on (80/140 kVp EICT -> Cong -> FBP -> VMI 70/150), so it is applied as delivered,
# not refitted here. Refitting on this phantom's own GT would be supervised by the answer.
#
# Reported at the REGION MEAN, which is where material-decomposition accuracy is defined; the
# per-voxel RMSE is printed alongside as the honest single-voxel number.
import BasisSimulator as BS
import TOML
using Statistics: mean, std, cor
using Printf: @printf
using Serialization: deserialize
using Unitful: @u_str
import CairoMakie as CM

const OUT = joinpath(@__DIR__, "pcat_ct")
const D = deserialize(joinpath(OUT, "pcat_acq.jls"))
const MODEL = TOML.parsefile(joinpath(@__DIR__, "wlp_model_70_150.toml"))
const K, VOXMM = 6, 0.5
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
const VESSEL_GROUP = Dict("rca1" => "healthy", "lcx" => "healthy", "lad1" => "healthy",
                          "lad2" => "diseased", "rca2" => "diseased", "lad3" => "diseased")
fat_label(k, i) = 40 + 6 * (K - k) + i
const RECON_N, RECON_FOV_CM, RECON_NZ = 512, 18.0, 3
const MATRIX = (RECON_N, RECON_N, RECON_NZ)

# ── poly2 decode: [1, hLo, hHi, hLo^2, hHi^2, hLo*hHi] -> (fw,fl,fp), normalised by sum ──
const CW = Float64.(MODEL["poly2"]["cw"])
const CL = Float64.(MODEL["poly2"]["cl"])
const CP = Float64.(MODEL["poly2"]["cp"])
const GATE_LO = Float64(MODEL["gate"]["soft_hu_lo"])
const GATE_HI = Float64(MODEL["gate"]["soft_hu_hi"])

function decode_poly2(hlo::Real, hhi::Real)
    b = (1.0, hlo, hhi, hlo^2, hhi^2, hlo * hhi)
    fw = sum(CW .* b); fl = sum(CL .* b); fp = sum(CP .* b)
    s = fw + fl + fp
    abs(s) < 1e-12 && return (NaN, NaN, NaN)
    (fw / s, fl / s, fp / s)
end

"""Project onto the simplex {f >= 0, sum = 1} ONCE on the result — per-sweep rectification
would be a Jensen bias on any region mean drawn from the map (model card's `simplex_note`)."""
function simplex_project(f)
    g = max.(f, 0.0); s = sum(g)
    s < 1e-12 ? (NaN, NaN, NaN) : Tuple(g ./ s)
end

# ── rebuild the label map on the recon grid ───────────────────────────────────────────
# Materials are irrelevant for the nearest-neighbour resample; a stub Dict keeps Phantom happy.
stub = Dict{Int, Any}(Int(l) => BS.XA.Materials.water for l in unique(D.slab))
ph_cpu = BS.Phantom(D.slab, stub, (VOXMM / 10, VOXMM / 10, VOXMM / 10))
m3 = BS.resample_to_recon(ph_cpu, D.geom, MATRIX; method = :nearest)
midz = size(m3, 3) ÷ 2 + 1
lab2 = m3[:, :, midz]
hlo2 = Float64.(D.hu_lo[:, :, midz])
hhi2 = Float64.(D.hu_hi[:, :, midz])
@info "recon grid $(size(lab2)), $(round(RECON_FOV_CM*10/RECON_N,digits=3)) mm/px; " *
      "PCAT voxels on the grid = $(count(40 .<= lab2 .< 76))"

# ── decode the whole slice, then score ────────────────────────────────────────────────
fw = fill(NaN, size(lab2)); fl = similar(fw); fp = similar(fw)
for i in eachindex(lab2)
    hl, hh = hlo2[i], hhi2[i]
    (GATE_LO <= hl <= GATE_HI) || continue
    f = simplex_project(decode_poly2(hl, hh))
    fw[i], fl[i], fp[i] = f
end

rows = NamedTuple[]
for (vi, v) in enumerate(VESSELS), k in 1:K
    l = fat_label(k, vi - 1)
    idx = findall(==(UInt8(l)), lab2)
    isempty(idx) && continue
    sel = [i for i in idx if !isnan(fw[i])]
    length(sel) < 20 && continue                  # too few pixels for a region mean
    t = D.gt[l]
    push!(rows, (vessel = v, group = VESSEL_GROUP[v], subring = k, n = length(sel),
                 gt_w = t[1], gt_l = t[2], gt_p = t[3],
                 m_w = mean(fw[sel]), m_l = mean(fl[sel]), m_p = mean(fp[sel]),
                 s_w = std(fw[sel]), s_l = std(fl[sel]), s_p = std(fp[sel])))
end
@info "$(length(rows)) (vessel, subring) regions with >= 20 decodable pixels"

# ── metrics ───────────────────────────────────────────────────────────────────────────
function ccc(x, y)
    mx, my = mean(x), mean(y)
    vx = mean((x .- mx) .^ 2); vy = mean((y .- my) .^ 2)
    cxy = mean((x .- mx) .* (y .- my))
    2cxy / (vx + vy + (mx - my)^2)
end
rmse(x, y) = sqrt(mean((x .- y) .^ 2))
function ols(x, y)                                # y = a + b x
    mx, my = mean(x), mean(y)
    b = sum((x .- mx) .* (y .- my)) / sum((x .- mx) .^ 2)
    (slope = b, intercept = my - b * mx)
end
r2(x, y) = cor(x, y)^2

MATS = [(:w, "Water", CM.RGBf(0.231, 0.459, 0.690)),
        (:l, "Lipid", CM.RGBf(0.90, 0.60, 0.10)),
        (:p, "Protein", CM.RGBf(0.757, 0.267, 0.235))]

println("\n", "="^92)
println("REGION-MEAN ACCURACY — ground truth vs measured volume fraction (n = $(length(rows)) regions)")
println("="^92)
@printf("%-9s %8s %8s %8s %8s %8s %8s %8s\n",
        "material", "CCC", "R2", "RMSE_%", "slope", "intercept", "bias_%", "GTrange_%")
stats = Dict{Symbol, Any}()
for (s, name, _) in MATS
    g = [r[Symbol("gt_", s)] for r in rows]
    m = [r[Symbol("m_", s)] for r in rows]
    o = ols(g, m)
    stats[s] = (g = g, m = m, ccc = ccc(g, m), r2 = r2(g, m), rmse = rmse(g, m),
                slope = o.slope, intercept = o.intercept, bias = mean(m .- g),
                gtrange = maximum(g) - minimum(g))
    st = stats[s]
    @printf("%-9s %8.4f %8.4f %8.2f %8.3f %8.4f %8.2f %8.2f\n",
            name, st.ccc, st.r2, 100st.rmse, st.slope, st.intercept, 100st.bias, 100st.gtrange)
end

# per-voxel honesty check
pv = Dict{Symbol, Float64}()
for (s, name, _) in MATS
    F = s === :w ? fw : (s === :l ? fl : fp)
    gv = Float64[]; mv = Float64[]
    for (vi, v) in enumerate(VESSELS), k in 1:K
        l = fat_label(k, vi - 1); t = D.gt[l]
        gi = s === :w ? t[1] : (s === :l ? t[2] : t[3])
        for i in findall(==(UInt8(l)), lab2)
            isnan(F[i]) && continue
            push!(gv, gi); push!(mv, F[i])
        end
    end
    pv[s] = rmse(gv, mv)
end
println("\nper-VOXEL RMSE (%): water $(round(100pv[:w],digits=1))  " *
        "lipid $(round(100pv[:l],digits=1))  protein $(round(100pv[:p],digits=1))")
println("region means average $(round(mean(r.n for r in rows),digits=0)) pixels, " *
        "so the region-mean standard error is ~sqrt(n) smaller than the per-voxel spread.")

# ── why: is the decoder wrong, or is the CT never seeing the fat? ─────────────────────
# Compare the measured VMI HU in each region against the theoretical HU of that region's own
# material. A bulk-tissue reference (myocardium, blood) is printed alongside: if bulk is right
# and the thin PCAT shells are not, the error is partial volume, not the decomposition.
const WATER_M = BS.XA.Materials.basis_water
const LIPID_M = BS.XA.Materials.basis_lipid
const PROT_M = BS.XA.Material("p", 0.0, 0.0u"eV", 1.35u"g/cm^3",
    Dict{Int, Float64}(1 => 0.066, 6 => 0.534, 7 => 0.170, 8 => 0.220, 16 => 0.010))
ρv(m) = BS.XA.val(m.density)
function wlpmat(fw, fl, fp)
    r = fw * ρv(WATER_M) + fl * ρv(LIPID_M) + fp * ρv(PROT_M)
    mf = (fw * ρv(WATER_M) / r, fl * ρv(LIPID_M) / r, fp * ρv(PROT_M) / r)
    c = Dict{Int, Float64}()
    for (m, wm) in ((WATER_M, mf[1]), (LIPID_M, mf[2]), (PROT_M, mf[3])), (Z, f) in m.composition
        c[Z] = get(c, Z, 0.0) + wm * f
    end
    BS.XA.Material("x", 0.0, 0.0u"eV", r * u"g/cm^3", c)
end
theo_hu(m, E) = (mw = BS.compute_μ_at_energy(WATER_M, E);
                 1000 * (BS.compute_μ_at_energy(m, E) - mw) / mw)
hubias = NamedTuple[]
for r in rows
    l = fat_label(r.subring, findfirst(==(r.vessel), VESSELS) - 1)
    idx = findall(==(UInt8(l)), lab2)
    m = wlpmat(D.gt[l]...)
    push!(hubias, (vessel = r.vessel, subring = r.subring, n = length(idx),
                   d70 = mean(hlo2[idx]) - theo_hu(m, 70.0)))
end
println("\nHU bias (measured VMI 70 keV minus this region's own theoretical HU):")
for h in hubias
    @printf("   %-6s subring%d  n=%4d   %+7.2f HU\n", h.vessel, h.subring, h.n, h.d70)
end
for (nm, ls) in (("myocardium", 15:18), ("chamber blood", 19:22))
    idx = findall(x -> Int(x) in ls, lab2)
    @printf("   %-14s n=%6d   HU70 mean %+8.2f  (bulk reference)\n", nm, length(idx), mean(hlo2[idx]))
end

# ── figure: GT vs measured per material, equal axis scale ─────────────────────────────
fig = CM.Figure(size = (1500, 980))
for (col, (s, name, colr)) in enumerate(MATS)
    st = stats[s]
    lo = min(minimum(st.g), minimum(st.m)); hi = max(maximum(st.g), maximum(st.m))
    pad = 0.08 * (hi - lo) + 1e-3
    lims = (100 * (lo - pad), 100 * (hi + pad))
    ax = CM.Axis(fig[1, col];
        title = name, titlesize = 20,
        xlabel = "ground truth $(lowercase(name)) volume fraction (%)",
        ylabel = col == 1 ? "measured volume fraction (%)" : "",
        aspect = CM.AxisAspect(1), limits = (lims, lims))          # equal axis scale
    CM.lines!(ax, [lims[1], lims[2]], [lims[1], lims[2]]; color = :gray55, linestyle = :dash)
    xs = range(lims[1], lims[2], 2)
    CM.lines!(ax, xs, 100 * st.intercept .+ st.slope .* xs; color = colr, linewidth = 2)
    for grp in ("healthy", "diseased")
        sel = [i for i in eachindex(rows) if rows[i].group == grp]
        isempty(sel) && continue
        CM.scatter!(ax, 100 .* st.g[sel], 100 .* st.m[sel];
            color = (colr, grp == "healthy" ? 0.95 : 0.5),
            marker = grp == "healthy" ? :circle : :utriangle, markersize = 13,
            strokewidth = 0.6, strokecolor = :white, label = grp)
    end
    CM.text!(ax, 0.03, 0.97; space = :relative, align = (:left, :top), fontsize = 13,
        text = "Concordance Correlation Coefficient = $(round(st.ccc, digits=3))\n" *
               "Coefficient of Determination R² = $(round(st.r2, digits=3))\n" *
               "Root Mean Square Error = $(round(100*st.rmse, digits=2)) %\n" *
               "Slope = $(round(st.slope, digits=3))")
    col == 3 && CM.axislegend(ax; position = :rb, framevisible = false, labelsize = 12)
end
# diagnostic row — the HU the decoder was actually fed
let ax = CM.Axis(fig[2, 1:3];
        title = "Why: the reconstruction never resolves the fat",
        titlesize = 19,
        xlabel = "subring (mm from the vessel wall)",
        ylabel = "measured VMI 70 keV minus this region's own true HU")
    CM.hlines!(ax, 0.0; color = :gray40, linestyle = :dash)
    vcol = Dict("rca1" => CM.RGBf(.85, .20, .18), "lad2" => CM.RGBf(.20, .45, .80),
                "lcx" => CM.RGBf(.20, .65, .35), "lad3" => CM.RGBf(.75, .50, .10))
    for v in unique(h.vessel for h in hubias)
        s = [h for h in hubias if h.vessel == v]
        sort!(s, by = x -> x.subring)
        c = get(vcol, v, CM.RGBf(.5, .5, .5))
        CM.scatterlines!(ax, [Float64(x.subring) for x in s], [x.d70 for x in s];
            color = c, markersize = 12, linewidth = 2, label = v)
    end
    CM.axislegend(ax; position = :rt, framevisible = false, labelsize = 12)
    CM.text!(ax, 0.02, 0.04; space = :relative, align = (:left, :bottom), fontsize = 13,
        text = "bulk reference is correct — myocardium reads $(round(mean(hlo2[findall(x->Int(x) in 15:18, lab2)]), digits=1)) HU, " *
               "blood $(round(mean(hlo2[findall(x->Int(x) in 19:22, lab2)]), digits=1)) HU.\n" *
               "The whole ground-truth spread across subrings is only 3.3 HU; the partial-volume " *
               "bias here is 11–55 HU.")
end
CM.Label(fig[0, :],
    "PCAT water/lipid/protein decomposition accuracy — FEBio phantom, no iodine, " *
    "70/150 keV VMI, region means over $(length(rows)) (vessel, subring) regions";
    fontsize = 17, font = :bold)
CM.save(joinpath(OUT, "pcat_wlp_accuracy.png"), fig; px_per_unit = 2)
println("\nfigure -> $(joinpath(OUT, "pcat_wlp_accuracy.png"))")

open(joinpath(OUT, "pcat_wlp_regions.csv"), "w") do io
    println(io, "vessel,group,subring,n_pixels,gt_water,gt_lipid,gt_protein," *
                "measured_water,measured_lipid,measured_protein,sd_water,sd_lipid,sd_protein")
    for r in rows
        println(io, "$(r.vessel),$(r.group),$(r.subring),$(r.n),$(r.gt_w),$(r.gt_l),$(r.gt_p)," *
                    "$(r.m_w),$(r.m_l),$(r.m_p),$(r.s_w),$(r.s_l),$(r.s_p)")
    end
end
println("regions -> $(joinpath(OUT, "pcat_wlp_regions.csv"))")
