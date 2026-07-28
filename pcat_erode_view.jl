# What the geometric partial-volume erosion actually removes, at zoom.
#
# The 20-layer analysis scores adipose voxels that survive an erosion of the union of ALL fat
# (FEBio subrings 40-75, distance shells 88-127, and the pericardial compartment 29). Eroding the
# UNION means shell-to-shell boundaries cost nothing — only the fat/non-fat surface retreats —
# and that is the thing this figure is for: seeing where the retreat happens and what it costs.
#
# Columns: CT at 70 keV | ground-truth labels | the surviving ROI at each erosion depth.
# Rows: one healthy vessel and one diseased vessel, because the diseased cuff runs against lung
# and is thin, so it is where deep erosion is most expensive.
#
# Usage: julia --project=. pcat_erode_view.jl [z_clinical] [erode_list]
#        julia --project=. pcat_erode_view.jl 12 0,2,4,8
import BasisSimulator as BS
import CairoMakie as CM
import TOML
using Statistics: mean, median
using Serialization: deserialize
using Printf: @printf, @sprintf
include(joinpath(@__DIR__, "pcat_orient.jl"))

const OUT = joinpath(@__DIR__, "pcat_ct")
const D = deserialize(joinpath(OUT, get(ENV, "PCAT_ACQ", "pcat_acq_shell.jls")))
const K, VOXMM, NSHELL = 6, 0.5, 20
const RECON_N, RECON_NZ, RECON_FOV_CM = 512, 40, 18.0
const PX_MM = RECON_FOV_CM * 10 / RECON_N
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
const VGROUP = Dict("rca1"=>"healthy","lcx"=>"healthy","lad1"=>"healthy",
                    "lad2"=>"diseased","rca2"=>"diseased","lad3"=>"diseased")
const WALL0, LUM0 = 76, 82
fat_label(k, i) = 40 + 6 * (K - k) + i
const SHELL0 = Dict("healthy"=>88, "diseased"=>108)

const ZPICK  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0        # 0 = auto (middle of valid z)
const ERODES = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ',')) : [0, 2, 4, 8]
const HALF   = 78                                                  # crop half-width, px (~27 mm)

# ── label grid + valid z window, identical to pcat_20layer.jl ────────────────────────
stub = Dict{Int,Any}(Int(l)=>BS.XA.Materials.water for l in unique(D.slab))
m3 = BS.resample_to_recon(BS.Phantom(D.slab, stub, (VOXMM/10,VOXMM/10,VOXMM/10)),
                          D.geom, (RECON_N,RECON_N,RECON_NZ); method=:nearest)
nz = size(m3,3)
myo = [let i = findall(x->15<=Int(x)<=18, m3[:,:,z])
           isempty(i) ? -Inf : mean(Float64.(D.hu_lo[:,:,z])[i]) end for z in 1:nz]
pl = let v = sort(filter(isfinite, myo)); median(v[(length(v)÷2+1):end]) end
gd = [z for z in 1:nz if isfinite(myo[z]) && abs(myo[z]-pl) <= 8.0]
ZR = (minimum(gd)+1):(maximum(gd)-1)
lab = m3[:,:,ZR]; H70 = Float64.(D.hu_lo[:,:,ZR])
@info "valid z $ZR (myocardium plateau $(round(pl,digits=1)) HU)"

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
allfat(sl) = ((sl .>= 40) .& (sl .<= 75)) .| ((sl .>= 88) .& (sl .<= 127)) .| (sl .== 29)

# ── pick the slice and the two vessels ───────────────────────────────────────────────
"""Centroid (in recon px) of a vessel's lumen on slice z, or nothing if it is not there."""
function lumen_centre(sl, vi)
    idx = findall(==(LUM0 + vi), Int.(sl))
    isempty(idx) && return nothing
    (mean(c -> c[1], idx), mean(c -> c[2], idx), length(idx))
end

zsel = ZPICK > 0 ? ZPICK : (first(ZR) + last(ZR)) ÷ 2 - first(ZR) + 1
zsel = clamp(zsel, 1, size(lab,3))
sl = lab[:,:,zsel]

# best-populated healthy and diseased vessel on this slice
picks = Tuple{String,Int,Float64,Float64}[]
for grp in ("healthy", "diseased")
    best = nothing
    for (i, v) in enumerate(VESSELS)
        VGROUP[v] == grp || continue
        c = lumen_centre(sl, i-1)
        c === nothing && continue
        (best === nothing || c[3] > best[2][3]) && (best = (v, c))
    end
    best === nothing && continue
    push!(picks, (best[1], findfirst(==(best[1]), VESSELS)-1, best[2][1], best[2][2]))
end
isempty(picks) && error("no coronary lumen on the selected slice — try another z")
for p in picks
    @printf("row: %-6s (%s) centred at recon px (%.0f, %.0f) on clinical z %d\n",
            p[1], VGROUP[p[1]], p[3], p[4], zsel)
end

# ── figure ───────────────────────────────────────────────────────────────────────────
disp(A) = reverse(A; dims = 2)                       # anterior up
ncol = 2 + length(ERODES)
fig = CM.Figure(size = (300*ncol + 120, 300*length(picks) + 130))

for (r, (vname, vi, cx, cy)) in enumerate(picks)
    i0 = clamp(round(Int, cx) - HALF, 1, RECON_N - 2HALF)
    j0 = clamp(round(Int, cy) - HALF, 1, RECON_N - 2HALF)
    ii, jj = i0:(i0+2HALF), j0:(j0+2HALF)
    grp = VGROUP[vname]
    ct  = H70[ii, jj, zsel]
    slc = Int.(sl[ii, jj])
    fat = BitMatrix(allfat(slc))

    # ground truth: shell distance in mm (grown subrings ARE shells 1-6), NaN elsewhere
    gt = fill(NaN, size(slc))
    for c in CartesianIndices(slc)
        L = slc[c]
        if 88 <= L <= 127
            gt[c] = (L - SHELL0[L >= 108 ? "diseased" : "healthy"]) + 1
        elseif 40 <= L <= 75
            gt[c] = K - (L - 40) ÷ 6                       # subring k = distance in mm
        end
    end
    wall = BitMatrix([76 <= L <= 81 for L in slc])
    lum  = BitMatrix([82 <= L <= 87 for L in slc])

    ax = CM.Axis(fig[r, 1]; aspect = CM.DataAspect(),
        title = r == 1 ? "CT, 70 keV" : "", titlesize = 15,
        ylabel = "$vname ($grp)", ylabelsize = 14)
    CM.heatmap!(ax, disp(ct); colormap = :grays, colorrange = (-200, 150))
    CM.hidedecorations!(ax; label = false); CM.hidespines!(ax)

    ax2 = CM.Axis(fig[r, 2]; aspect = CM.DataAspect(),
        title = r == 1 ? "ground truth: distance shell (mm)" : "", titlesize = 15)
    CM.heatmap!(ax2, disp(ct); colormap = :grays, colorrange = (-200, 150))
    hm = CM.heatmap!(ax2, disp(gt); colormap = :viridis, colorrange = (1, NSHELL), nan_color = :transparent)
    CM.contour!(ax2, disp(Float64.(wall)); levels = [0.5], color = (:orangered, 0.9), linewidth = 1.4)
    CM.contour!(ax2, disp(Float64.(lum));  levels = [0.5], color = (:cyan, 0.9), linewidth = 1.4)
    CM.hidedecorations!(ax2); CM.hidespines!(ax2)
    r == 1 && CM.Colorbar(fig[r, 2, CM.Right()], hm; width = 9, ticklabelsize = 10)

    for (c, e) in enumerate(ERODES)
        m = erode2(fat, e)
        # count only the scored labels (shells + subrings), not the whole pericardial compartment
        scored = m .& BitMatrix([(40 <= L <= 75) || (88 <= L <= 127) for L in slc])
        base   = fat .& BitMatrix([(40 <= L <= 75) || (88 <= L <= 127) for L in slc])
        keep   = 100 * count(scored) / max(count(base), 1)
        axe = CM.Axis(fig[r, 2+c]; aspect = CM.DataAspect(), titlesize = 15,
            title = r == 1 ? @sprintf("erode %d px = %.2f mm", e, e*PX_MM) : "",
            xlabel = @sprintf("%d px kept (%.0f%%)", count(scored), keep), xlabelsize = 12)
        CM.heatmap!(axe, disp(ct); colormap = :grays, colorrange = (-200, 150))
        ov = fill(NaN, size(slc)); ov[scored] .= 1.0
        CM.heatmap!(axe, disp(ov); colormap = CM.cgrad([:gold, :gold]), colorrange = (0, 1), nan_color = :transparent)
        CM.contour!(axe, disp(Float64.(base)); levels = [0.5], color = (:deepskyblue, 0.85), linewidth = 1.2)
        CM.hidedecorations!(axe; label = false); CM.hidespines!(axe)
    end
end

CM.Label(fig[0, :], "Geometric partial-volume erosion at zoom — clinical z $zsel, " *
    "$(round(PX_MM, digits=3)) mm/px.  gold = surviving scored ROI, blue outline = " *
    "un-eroded scored adipose, orange = vessel wall, cyan = lumen.  " *
    "Erosion is applied to the union of ALL fat, so shell-to-shell borders cost nothing.";
    fontsize = 15, font = :bold)

path = joinpath(OUT, "pcat_erode_view.png")
CM.save(path, fig; px_per_unit = 1.6)
println("\nfigure -> ", path)
