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

# 3-D helpers for the --rois mode, matching pcat_20layer.jl exactly
const ERODE_SHOW = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
const ALLFAT3 = allfat(lab)
"""Erode the union of ALL adipose by n voxels, per slice (pcat_20layer.jl's geometric PV rule)."""
function fat_eroded(n::Int)
    Ee = falses(size(ALLFAT3))
    for z in axes(ALLFAT3, 3); Ee[:, :, z] = erode2(BitMatrix(ALLFAT3[:, :, z]), n); end
    Ee
end
"""Labels of one (group, shell): the distance shell, plus the FEBio subrings for k <= K."""
function shell_labels(g, k)
    labs = Int[SHELL0[g] + k - 1]
    k <= K && append!(labs, [fat_label(k, i-1) for (i,v) in enumerate(VESSELS) if VGROUP[v]==g])
    labs
end
"""CartesianIndices of the (group, shell) ring over the whole valid volume."""
function ROI(g, k; eroded = true, E = nothing)
    labs = Set(shell_labels(g, k))
    idx = [c for c in CartesianIndices(lab) if Int(lab[c]) in labs]
    eroded ? [c for c in idx if E[c]] : idx
end

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

disp(A) = reverse(A; dims = 2)                       # anterior up

# ══ MODE: --rois ═══════════════════════════════════════════════════════════════════
# Where a data point actually comes from, and what the erosion is doing to it.
#
# One scored data point is ONE (group, shell) ring: every voxel in the volume carrying that
# shell's labels, pooled over the group's three vessels and all valid z slices. Not a small
# circular ROI, not a sphere, not a sample. The panels below show individual rings on the CT so
# that is visible, with the surviving subset separated from the part erosion removes.
#
# The erosion itself is `erode2`: N iterations of 4-neighbour binary erosion of the union of ALL
# adipose. Iterating a plus-shaped structuring element N times is a threshold on the CITY-BLOCK
# distance to the nearest non-fat voxel, so a voxel survives iff that distance exceeds N.
#
# Why that is a defensible rule: it reads ONLY the label map. It never looks at the measured HU,
# the decoded fractions, or the truth, so it cannot preferentially discard voxels that disagree
# with the answer — which is exactly what gating on measured HU would do. It is one integer, it is
# monotone in that integer, and the survival count per shell is deterministic and printable.
#
# Where it is NOT clean, stated rather than hidden: city-block distance is anisotropic, so the
# clearance is N px along the axes but only N/sqrt(2) along the diagonals; the measured Euclidean
# clearance is reported below. And it does not repair the boundary zone, it deletes it — where the
# fat cuff is thinner than 2N px the ring disappears instead of being corrected.
if "--rois" in ARGS
    SHOW = [3, 8, 14, 20]
    E = fat_eroded(ERODE_SHOW)
    println("\nring provenance, erode $ERODE_SHOW px = $(round(ERODE_SHOW*PX_MM, digits=2)) mm")
    @printf("%-9s %5s %10s %10s %8s\n", "group", "shell", "n_total", "n_kept", "kept%")
    for g in ("healthy", "diseased"), k in SHOW
        tot = ROI(g, k; eroded = false); kept = ROI(g, k; eroded = true, E = E)
        @printf("%-9s %5d %10d %10d %7.0f%%\n", g, k, length(tot), length(kept),
                100 * length(kept) / max(length(tot), 1))
    end
    # exact Euclidean clearance of surviving voxels, brute force on a window, subsampled
    let W = ERODE_SHOW + 2, samp = Float64[]
        nf = .!ALLFAT3
        idx = [c for c in CartesianIndices(ALLFAT3) if E[c]]
        for c in idx[1:97:end]
            i, j, z = c.I; best = Inf
            for dj in -W:W, di in -W:W
                ii, jj = i + di, j + dj
                (1 <= ii <= size(ALLFAT3,1) && 1 <= jj <= size(ALLFAT3,2)) || continue
                nf[ii, jj, z] && (best = min(best, sqrt(Float64(di^2 + dj^2))))
            end
            isfinite(best) && push!(samp, best * PX_MM)
        end
        sort!(samp)
        @printf("\nEuclidean clearance of surviving voxels (n=%d sampled): min %.2f mm, 5th pct %.2f mm, median %.2f mm\n",
                length(samp), samp[1], samp[max(1, round(Int, 0.05*length(samp)))], samp[length(samp)÷2])
        @printf("  city-block threshold was %d px = %.2f mm; along the diagonals that is only %.2f mm — the low end of the 2-3 mm boundary-artifact zone.\n",
                ERODE_SHOW, ERODE_SHOW*PX_MM, ERODE_SHOW*PX_MM/sqrt(2))
    end

    fig = CM.Figure(size = (300*(1+length(SHOW)) + 120, 300*length(picks) + 150))
    for (r, (vname, vi, cx, cy)) in enumerate(picks)
        i0 = clamp(round(Int, cx) - HALF, 1, RECON_N - 2HALF)
        j0 = clamp(round(Int, cy) - HALF, 1, RECON_N - 2HALF)
        ii, jj = i0:(i0+2HALF), j0:(j0+2HALF)
        g = VGROUP[vname]
        ct = H70[ii, jj, zsel]; slc = Int.(sl[ii, jj])
        ax = CM.Axis(fig[r, 1]; aspect = CM.DataAspect(), titlesize = 14,
            title = r == 1 ? "CT, 70 keV" : "", ylabel = "$vname ($g)", ylabelsize = 13)
        CM.heatmap!(ax, disp(ct); colormap = :grays, colorrange = (-200, 150))
        CM.hidedecorations!(ax; label = false); CM.hidespines!(ax)
        for (c, k) in enumerate(SHOW)
            labs = Set(shell_labels(g, k))
            ring = BitMatrix([L in labs for L in slc])
            kept = ring .& BitMatrix(E[ii, jj, zsel])
            nt = length(ROI(g, k; eroded = false)); nk = length(ROI(g, k; eroded = true, E = E))
            axk = CM.Axis(fig[r, 1+c]; aspect = CM.DataAspect(), titlesize = 14,
                title = r == 1 ? "shell $k  (one data point)" : "",
                xlabel = @sprintf("%d of %d voxels kept (%.0f%%)", nk, nt, 100nk/max(nt,1)),
                xlabelsize = 11)
            CM.heatmap!(axk, disp(ct); colormap = :grays, colorrange = (-200, 150))
            rm = fill(NaN, size(slc)); rm[ring .& .!kept] .= 1.0
            kp = fill(NaN, size(slc)); kp[kept] .= 1.0
            CM.heatmap!(axk, disp(rm); colormap = CM.cgrad([:firebrick,:firebrick]),
                        colorrange = (0,1), nan_color = :transparent)
            CM.heatmap!(axk, disp(kp); colormap = CM.cgrad([:gold,:gold]),
                        colorrange = (0,1), nan_color = :transparent)
            CM.hidedecorations!(axk; label = false); CM.hidespines!(axk)
        end
    end
    CM.Label(fig[0, :], "One data point = one (group, shell) ring, pooled over the group's three " *
        "vessels and all $(length(ZR)) valid z slices — shown here on one slice (clinical z $zsel). " *
        "gold = survives the $(ERODE_SHOW) px ($(round(ERODE_SHOW*PX_MM,digits=2)) mm) erosion and is scored; " *
        "dark red = removed by it. Erosion thresholds the city-block distance to non-fat, computed " *
        "from LABELS only — never from the measured HU, the decode, or the truth.";
        fontsize = 14, font = :bold, word_wrap = true)
    path = joinpath(OUT, "pcat_roi_provenance.png")
    CM.save(path, fig; px_per_unit = 1.6)
    println("\nfigure -> ", path)
    exit()
end

# ── figure ───────────────────────────────────────────────────────────────────────────
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
