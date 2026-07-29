# Figures for pcat_rca1_proximal.jl. Included from it, so it sees rows/srows/excl/stats/sect and the
# geometry arrays directly — this is a figure body, not a reusable module.
#
# DISPLAY ORIENTATION. +y is POSTERIOR in the array (measured in section 1), so every axial panel
# sets yreversed = true to put ANTERIOR UP. +x is patient left, which is already image-right in the
# radiological convention, so x is not flipped.

const CSEC = [CM.RGBf(.22,.45,.72), CM.RGBf(.30,.62,.76), CM.RGBf(.40,.70,.55), CM.RGBf(.62,.72,.33),
              CM.RGBf(.85,.65,.20), CM.RGBf(.84,.45,.20), CM.RGBf(.76,.27,.24), CM.RGBf(.55,.35,.62)]
const CLIN = CM.RGBf(.20,.45,.80)

# ══ figure 1 — the 20-layer format, restricted to proximal rca1 ══════════════════════
fig = CM.Figure(size = (1620, 900))
ax1 = CM.Axis(fig[1,1:2]; title = "radial HU profile — proximal rca1 only, phantom vs its own truth",
    titlesize = 17, xlabel = "shell (mm from the coronary)", ylabel = "HU at 70 keV", xticks = 1:2:20)
SHELL_MIN > 1 && CM.vspan!(ax1, 0.5, SHELL_MIN - 0.5; color = (:gray60, 0.16))
SHELL_MIN > 1 && CM.text!(ax1, SHELL_MIN - 0.6, 1.0; space = :relative, align = (:right, :top),
    text = "not scored\n(shell < $SHELL_MIN)", fontsize = 9, color = :gray45, offset = (0,-4))
let rr = [r for r in rows if !isnan(r.hu_gate)], sh = [Float64(r.shell) for r in rr]
    CM.lines!(ax1, sh, [r.truthHU for r in rr]; color = CLIN, linewidth = 2, linestyle = :dash,
              label = "truth — the shell's nominal composition")
    CM.scatterlines!(ax1, sh, [r.hu_gate for r in rr]; color = CLIN, linewidth = 2.6, markersize = 8,
                     label = "measured — clinical FAI ([$(Int(ADIPOSE_LO)), $(Int(ADIPOSE_HI))] HU gate)")
end
CM.axislegend(ax1; position = :rt, framevisible = false, labelsize = 11)

ax2 = CM.Axis(fig[1,3]; title = "adipose pixels per shell", titlesize = 17,
    xlabel = "shell (mm)", ylabel = "pixels in the ROI", xticks = 1:2:20, yscale = log10)
CM.scatterlines!(ax2, [Float64(r.shell) for r in rows], [Float64(max(r.n_gate,1)) for r in rows];
                 color = :gray35, markersize = 8, label = "HU-gated")
let rr = [r for r in rows if !isnan(r.m_l)]
    isempty(rr) || CM.scatterlines!(ax2, [Float64(r.shell) for r in rr],
        [Float64(max(r.n_pv,1)) for r in rr]; color = CLIN, markersize = 8, label = "MMD, eroded")
end
CM.axislegend(ax2; position = :rt, framevisible = false, labelsize = 11)

for (col, (s, nm, c)) in enumerate(((:w,"Water",CM.RGBf(.23,.46,.69)),
                                    (:l,"Lipid",CM.RGBf(.90,.60,.10)),
                                    (:p,"Protein",CM.RGBf(.76,.27,.24))))
    g, m = stats[s].g, stats[s].m
    exd = [r for r in excl if !isnan(r.m_l)]
    ge = [r[Symbol("gt_",s)] for r in exd]; me = [r[Symbol("m_",s)] for r in exd]
    allg = vcat(g, ge); allm = vcat(m, me)
    isempty(allg) && continue
    lo = min(minimum(allg), minimum(allm)); hi = max(maximum(allg), maximum(allm))
    pad = 0.08*(hi-lo) + 1e-3; L = (100*(lo-pad), 100*(hi+pad))
    ax = CM.Axis(fig[2,col]; title = nm, titlesize = 16, aspect = CM.AxisAspect(1), limits = (L,L),
        xlabel = "ground truth $(lowercase(nm)) (%)", ylabel = col == 1 ? "measured (%)" : "")
    CM.lines!(ax, [L[1],L[2]], [L[1],L[2]]; color = :gray55, linestyle = :dash)
    isempty(ge) || CM.scatter!(ax, 100 .* ge, 100 .* me; color = :transparent, marker = :circle,
                               markersize = 11, strokewidth = 1.1, strokecolor = (c,0.75))
    isempty(g) || CM.scatter!(ax, 100 .* g, 100 .* m; color = (c,0.95), marker = :circle,
                              markersize = 11, strokewidth = 0.5, strokecolor = :white)
    isempty(g) || CM.text!(ax, 0.03, 0.97; space = :relative, align = (:left,:top), fontsize = 12,
        text = "CCC = $(round(ccc(g,m),digits=3))\nRMSE = $(round(100sqrt(mean((m.-g).^2)),digits=2)) %" *
               "\nbias = $(round(100mean(m.-g),digits=2)) %")
    col == 3 && CM.text!(ax, 0.97, 0.03; space = :relative, align = (:right,:bottom), fontsize = 10,
        color = :gray40, text = "hollow = excluded\n(shell < $SHELL_MIN or n < $NMIN_STAT)")
end
CM.Label(fig[0,:], "Proximal RCA1 only — FAI segment requested $(Int(PROX_LO))-$(Int(PROX_HI)) mm " *
    "from the ostium, ACTUALLY COVERED $(round(ARC_OK[1],digits=1))-$(round(ARC_OK[2],digits=1)) mm " *
    "($(round(Int,COVER))% of it, the recon z window is the limit); erode $ERODE px = " *
    "$(round(ERODE*PX_MM,digits=2)) mm; shells >= $SHELL_MIN with n_PV >= $NMIN_STAT";
    fontsize = 15, font = :bold)
CM.save(joinpath(OUT, "pcat_rca1_proximal$(TAG).png"), fig; px_per_unit = 2)
println("\nfigure -> ", joinpath(OUT, "pcat_rca1_proximal$(TAG).png"))

# ══ figure 2 — the 8 azimuthal sectors ══════════════════════════════════════════════
f2 = CM.Figure(size = (1500, 860))
SLAB_ID = PZ[max(1, length(PZ) ÷ 2)]                       # a representative proximal slice

# (a) the sector map, so the reader can see what "sector 5" means anatomically.
# ORIENTATION: +y is POSTERIOR in the array, so the plot's Y coordinate is -y and anterior is UP.
# Doing it by negating the data rather than with yreversed, because a later limits! call silently
# undoes yreversed and the panel comes out upside down.
axm = CM.Axis(f2[1,1]; title = "(a) the ROI on recon slice $SLAB_ID  (arc $(round(ARC[SLAB_ID],digits=1)) mm)",
    titlesize = 13, aspect = CM.DataAspect(),
    xlabel = "patient left  →  (mm from the vessel centre)",
    ylabel = "anterior  →  (mm)")
let sel = findall(c -> c[3] == SLAB_ID, CartesianIndices(lab))
    solid = [c for c in sel if Int(lab[c]) in SOLID_RCA1]
    roi = [c for c in roi_std if c[3] == SLAB_ID]
    # everything is drawn RELATIVE to the vessel centroid, so the axes read as depth rather than as
    # meaningless absolute recon coordinates
    x0 =  mean(c[1] for c in solid)*PX_MM
    y0 = -mean(c[2] for c in solid)*PX_MM
    for i in 1:NSECT
        pts = [c for c in roi if sector_of(THETA[c]) == i]
        isempty(pts) && continue
        CM.scatter!(axm, [c[1]*PX_MM - x0 for c in pts], [-c[2]*PX_MM - y0 for c in pts];
                    color = CSEC[i], markersize = 3.2)
    end
    CM.scatter!(axm, [c[1]*PX_MM - x0 for c in solid], [-c[2]*PX_MM - y0 for c in solid];
                color = :gray20, markersize = 3.2)
    ao = [c for c in sel if Int(lab[c]) == AORTA]
    isempty(ao) || CM.scatter!(axm, [c[1]*PX_MM - x0 for c in ao], [-c[2]*PX_MM - y0 for c in ao];
                               color = (:firebrick,0.35), markersize = 2.2)
    if !isempty(roi) && !isempty(solid)
        cx, cY = 0.0, 0.0
        rr = 2.5 * DIAM[SLAB_ID]
        for i in 1:NSECT                          # sector centre directions, in PLOT coordinates
            θ = 45.0*(i-1)
            CM.text!(axm, cx + rr*sind(θ), cY + rr*cosd(θ); text = SECT_LABEL[i],
                     align = (:center,:center), fontsize = 12, color = CSEC[i], font = :bold)
        end
        w = 3.2 * DIAM[SLAB_ID]
        CM.limits!(axm, cx-w, cx+w, cY-w, cY+w)
    end
end

# (b) FAI per sector against that sector's own truth
axf = CM.Axis(f2[1,2]; title = "(b) FAI per sector — measured vs that sector's OWN truth",
    titlesize = 13, xlabel = "sector (centre direction)", ylabel = "HU at 70 keV",
    xticks = (1:NSECT, SECT_LABEL))
let x = Float64.(1:NSECT), fv = [s.fai for s in sect], tv = [s.truthHU for s in sect],
    ev = [s.fai_sem for s in sect]
    CM.lines!(axf, x, tv; color = :gray35, linestyle = :dash, linewidth = 2, label = "truth")
    CM.scatter!(axf, x, tv; color = :gray35, markersize = 9)
    CM.errorbars!(axf, x, fv, ev, ev; color = CLIN, whiskerwidth = 8)
    CM.scatterlines!(axf, x, fv; color = CLIN, markersize = 10, linewidth = 2.4,
                     label = "measured FAI (bars = standard error of the mean)")
end
CM.axislegend(axf; position = :lb, framevisible = true, backgroundcolor = (:white,0.85), labelsize = 10)

# (c) the decoded fractions per sector
axc = CM.Axis(f2[2,1]; title = "(c) decoded volume fractions per sector — measured vs truth",
    titlesize = 13, xlabel = "sector (centre direction)", ylabel = "volume fraction",
    xticks = (1:NSECT, SECT_LABEL))
for (s, nm, c) in ((:w,"water",CM.RGBf(.23,.46,.69)), (:l,"lipid",CM.RGBf(.90,.60,.10)),
                   (:p,"protein",CM.RGBf(.76,.27,.24)))
    x = Float64.(1:NSECT)
    CM.lines!(axc, x, [ss[Symbol("gt_",s)] for ss in sect]; color = (c,0.55), linestyle = :dash,
              linewidth = 2)
    CM.scatterlines!(axc, x, [ss[Symbol("m_",s)] for ss in sect]; color = c, markersize = 9,
                     linewidth = 2.2, label = nm)
end
CM.axislegend(axc; position = :rc, framevisible = false, labelsize = 10)
CM.text!(axc, 0.03, 0.97; space = :relative, align = (:left,:top), fontsize = 10, color = :gray40,
         text = "dashed = truth, solid = measured")

# (d) how many voxels each sector actually has — a thin sector is a noisy sector
axn = CM.Axis(f2[2,2]; title = "(d) ROI size and mean radial depth per sector", titlesize = 13,
    xlabel = "sector (centre direction)", ylabel = "voxels in the ROI",
    xticks = (1:NSECT, SECT_LABEL))
CM.barplot!(axn, Float64.(1:NSECT), Float64.([s.n_gate for s in sect]);
            color = [(CSEC[i], 0.75) for i in 1:NSECT], strokewidth = 0.5)
let axr = CM.Axis(f2[2,2]; ylabel = "mean radial depth (mm)", yaxisposition = :right,
                  xticksvisible = false, xticklabelsvisible = false)
    CM.hidespines!(axr); CM.hidexdecorations!(axr)
    CM.scatterlines!(axr, Float64.(1:NSECT), [s.rmean for s in sect]; color = :black,
                     markersize = 8, linewidth = 1.8, linestyle = :dot)
    CM.linkxaxes!(axn, axr)
end
CM.text!(axn, 0.03, 0.97; space = :relative, align = (:left,:top), fontsize = 10, color = :gray40,
         text = "bars = gated voxel count (left axis)\ndotted = mean radial depth (right axis)")

CM.Label(f2[0,:], "Proximal RCA1, 8 azimuthal sectors of the FAI ROI — cross-sectional (axial) " *
    "sectors, centre of sector 1 = ANTERIOR, increasing toward PATIENT LEFT; arc " *
    "$(round(ARC_OK[1],digits=1))-$(round(ARC_OK[2],digits=1)) mm from the ostium, " *
    "ROI = gated voxels within one vessel diameter ($(round(DMED,digits=2)) mm)";
    fontsize = 14, font = :bold)
CM.save(joinpath(OUT, "pcat_rca1_sectors$(TAG).png"), f2; px_per_unit = 2)
println("figure -> ", joinpath(OUT, "pcat_rca1_sectors$(TAG).png"))
