# 20-layer radial analysis, matching the Oxford construct.
#
# The phantom now has ground truth across the whole 1-20 mm range: the FEBio-grown PCAT carries
# per-vessel labels 40-75 (which ARE shells 1-6 by construction) and the pericardial fat within
# 20 mm was carved into distance shells 88-127 (healthy 88-107, diseased 108-127). Composition
# depends on distance and group only, so a voxel 3 mm out has the same truth whether or not FEBio
# happened to grow fat there.
#
# TWO SEPARATE LINES, deliberately not mixed:
#   MMD      — geometric partial-volume avoidance (erode the fat mask), NO adipose gate.
#              Selecting voxels by measured HU and then scoring that measurement is circular.
#   clinical — the [-190,-30] adipose gate, which is part of the FAI definition, reported
#              alongside so the two can be compared but never averaged together.
import CairoMakie as CM
using Statistics: mean, std, cor, median
using Printf: @printf
include(joinpath(@__DIR__, "pcat_common.jl"))   # grid, valid z, decode, geometry, ccc, sem

const D = pcat_load(get(ENV, "PCAT_ACQ", "pcat_acq_shell.jls"))
const MODEL = pcat_model()

# Three rules, all measured on this phantom (see the erosion sweep and the same-region control):
#
# ERODE = 8 px = 2.81 mm. The in-plane edge response of this recon has FWHM ~1.0-1.1 mm (measured
#   as the edge spread function across fat|myocardium, myocardium|blood and fat|lung, and identical
#   at both keV), but the fat plateau does not recover until ~6-8 px: the BOUNDARY-ARTIFACT zone is
#   2-3 mm, because the two-basis decomposition is nonlinear across an interface and the FBP ramp
#   overshoots. That zone, not the FWHM, is what has to be cleared. At erode 2 (0.70 mm) the shipped
#   lipid bias is -7.03 pp; at erode 8 it is -0.95 pp. Scored on the SAME 36 regions so the region
#   set cannot flatter the comparison: -6.51 -> -0.95 pp, CCC 0.375 -> 0.744.
#
# SHELL_MIN = 3. Erosion retreats from the vessel WALL outward as well as from lung/myocardium
#   inward, so shells 1-2 cannot survive it: shell 1 sits 1 mm from a +43 HU muscle wall, inside the
#   2-3 mm artifact zone. Healthy shell 1 keeps 110 of 2381 voxels and still reads -8.90 pp;
#   diseased shell 1 keeps none. These layers are not measurable at this resolution — they are
#   TABULATED but excluded from the accuracy statistics rather than silently dropped.
#
# NMIN_STAT = 500. The sparse outer diseased shells that do survive (n = 73-376) flip sign to
#   +3.2 .. +6.1 pp. That is the standard error of the ROI mean, not a bias. Regions below the floor
#   are tabulated with their standard error and excluded from the statistics.
const ERODE_DEF, SHELL_MIN_DEF, NMIN_STAT_DEF = 8, 3, 500
const ERODE      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : ERODE_DEF
const SHELL_MIN  = parse(Int, get(ENV, "PCAT_SHELL_MIN", string(SHELL_MIN_DEF)))
const NMIN_STAT  = parse(Int, get(ENV, "PCAT_NMIN_STAT", string(NMIN_STAT_DEF)))
const NMIN_TABLE = 30                      # below this a region is not even reported
# Non-default settings write to their own filenames, so a sweep never clobbers the canonical run.
const TAG = (ERODE, SHELL_MIN, NMIN_STAT) == (ERODE_DEF, SHELL_MIN_DEF, NMIN_STAT_DEF) ? "" :
            "_e$(ERODE)_s$(SHELL_MIN)_n$(NMIN_STAT)"
const CW = Float64.(MODEL["poly2"]["cw"]); const CL = Float64.(MODEL["poly2"]["cl"])
const CP = Float64.(MODEL["poly2"]["cp"])
const GLO = Float64(MODEL["gate"]["soft_hu_lo"]); const GHI = Float64(MODEL["gate"]["soft_hu_hi"])
const LAMBDA = Float64(MODEL["tv"]["lambda"]); const TVIT = Int(MODEL["tv"]["iters"])
const TVEPS = Float64(MODEL["tv"]["eps"])

const truthHU = pcat_truth_hu(D, 70.0)

# ── grid + valid z (drop the plateau's first and last slice) ─────────────────────────
m3 = pcat_label_grid(D)
ZR, plateau = pcat_valid_z(D, m3; trim = 1)
lab = m3[:,:,ZR]; H70 = Float64.(D.hu_lo[:,:,ZR]); H150 = Float64.(D.hu_hi[:,:,ZR])
@info "myocardium plateau $(round(plateau,digits=1)) HU; using z $ZR (plateau minus first/last)"

FW, FL, FP = pcat_decode(H70, H150, MODEL)

# ── geometric partial-volume avoidance ───────────────────────────────────────────────
fe = pcat_fat_erode(lab, ERODE); FATE = fe.mask
@info "PV erosion $ERODE vox: $(fe.n_before) -> $(fe.n_after) adipose pixels"

# ── per (shell, group) statistics ────────────────────────────────────────────────────
rows = NamedTuple[]
for g in ("healthy","diseased"), k in 1:NSHELL
    labs = Int[shell_label(g,k)]
    k <= K && append!(labs, [fat_label(k, i-1) for (i,v) in enumerate(VESSELS) if VGROUP[v]==g])
    idx = [c for c in CartesianIndices(lab) if Int(lab[c]) in labs]
    isempty(idx) && continue
    pv  = [c for c in idx if FATE[c] && !isnan(FW[c])]                  # MMD: geometric only
    gt  = [c for c in idx if ADIPOSE_LO <= H70[c] <= ADIPOSE_HI]        # clinical: HU gate
    # Each arm stands or falls on ITS OWN voxel count. Testing only the MMD count used to drop the
    # whole row, which truncated the clinical FAI curve at 17 mm on the diseased side even though
    # its gated ROI still held 1861-2096 voxels there — the erosion is not part of the FAI
    # definition, so it must not decide what the FAI curve shows.
    (length(pv) < NMIN_TABLE && length(gt) < NMIN_TABLE) && continue
    hasmmd = length(pv) >= NMIN_TABLE
    t = D.gt[labs[1]]
    # Standard error of the ROI mean = per-voxel standard deviation / sqrt(N). These are two
    # different statistics and both are reported; never collapse them into one "sigma". The TV
    # stage correlates neighbouring voxels, so this standard error is a LOWER bound on the true
    # uncertainty of the region mean.
    mm(f) = hasmmd ? f() : NaN            # MMD fields are NaN when only the clinical arm survives
    push!(rows, (group=g, shell=k, n_geom=length(idx), n_pv=length(pv), n_gate=length(gt),
                 gt_w=t[1], gt_l=t[2], gt_p=t[3], truthHU=truthHU[labs[1]],
                 m_w=mm(()->mean(FW[pv])), m_l=mm(()->mean(FL[pv])), m_p=mm(()->mean(FP[pv])),
                 sd_l=mm(()->std(FL[pv])), sem_w=mm(()->sem(FW[pv])),
                 sem_l=mm(()->sem(FL[pv])), sem_p=mm(()->sem(FP[pv])),
                 hu_pv=mm(()->mean(H70[pv])),
                 hu_gate=isempty(gt) ? NaN : mean(H70[gt]),
                 instat = hasmmd && k >= SHELL_MIN && length(pv) >= NMIN_STAT))
end

println("\n", "="^104)
println("20-LAYER RADIAL ANALYSIS — MMD (geometric PV avoidance, no HU gate) | clinical (HU-gated)")
println("="^104)
@printf("%-9s %5s %8s %8s %8s %9s %9s %9s   %8s %8s %+8s %8s %5s\n",
        "group","shell","n_geom","n_PV","n_gate","truth_HU","HU_PV","HU_gate",
        "GT_lip%","ms_lip%","d_lip%","SEM_lip%","stat")
for r in rows
    @printf("%-9s %5d %8d %8d %8d %9.1f %9.1f %9.1f   %8.1f %8.1f %+8.1f %8.2f %5s\n",
            r.group, r.shell, r.n_geom, r.n_pv, r.n_gate, r.truthHU, r.hu_pv, r.hu_gate,
            100r.gt_l, 100r.m_l, 100*(r.m_l - r.gt_l), 100r.sem_l, r.instat ? "yes" : "--")
end

# Regions are TABULATED above and only then filtered for the statistics, so nothing is dropped
# silently — the excluded ones are named here with the rule that excluded them.
srows = [r for r in rows if r.instat]
excl  = [r for r in rows if !r.instat]
isempty(excl) || println("\nexcluded from the accuracy statistics (still tabulated above):")
for r in excl
    why = isnan(r.m_l) ? "MMD ROI has only $(r.n_pv) voxels after erosion — clinical arm still scored ($(r.n_gate) gated voxels)" :
          r.shell < SHELL_MIN ? "shell < $SHELL_MIN (inside the wall's boundary-artifact zone)" :
                                "n_PV $(r.n_pv) < $NMIN_STAT (standard error of the ROI mean $(round(100r.sem_l,digits=2)) pp)"
    @printf("  %-9s shell %2d — %s\n", r.group, r.shell, why)
end
nmiss = 2*NSHELL - length(rows)
nmiss > 0 && @printf("  %d of %d (group, shell) regions had fewer than %d voxels and are not reported at all\n",
                     nmiss, 2*NSHELL, NMIN_TABLE)
isempty(srows) && error("no region survives shell >= $SHELL_MIN and n_PV >= $NMIN_STAT — loosen the floors")

println("\nMMD accuracy — geometric PV avoidance only, erode $ERODE px ($(round(ERODE*PX_MM,digits=2)) mm)")
println("  reported over $(length(srows)) of $(length(rows)) tabulated regions " *
        "(shell >= $SHELL_MIN and n_PV >= $NMIN_STAT)")
@printf("%-9s %8s %8s %8s %8s %9s %10s\n","material","CCC","R2","RMSE_%","bias_%","GTrange_%","medSEM_%")
stats = Dict{Symbol,Any}()
for (s,nm) in ((:w,"Water"),(:l,"Lipid"),(:p,"Protein"))
    g = [r[Symbol("gt_",s)] for r in srows]; m = [r[Symbol("m_",s)] for r in srows]
    stats[s] = (g=g, m=m)
    @printf("%-9s %8.4f %8.4f %8.2f %+8.2f %9.2f %10.2f\n", nm, ccc(g,m), cor(g,m)^2,
            100sqrt(mean((m.-g).^2)), 100mean(m.-g), 100*(maximum(g)-minimum(g)),
            100median([r[Symbol("sem_",s)] for r in srows]))
end
println("  medSEM = median standard error of the ROI mean (per-voxel standard deviation / sqrt(N));")
println("  TV correlates neighbouring voxels, so it is a LOWER bound on the region mean's uncertainty.")
mrows = [r for r in rows if !isnan(r.m_l)]      # rows whose MMD arm exists at all
if length(srows) < length(mrows)
    println("\nfor reference, the same statistics over ALL $(length(mrows)) regions with an MMD ROI:")
    for (s,nm) in ((:w,"Water"),(:l,"Lipid"),(:p,"Protein"))
        g = [r[Symbol("gt_",s)] for r in mrows]; m = [r[Symbol("m_",s)] for r in mrows]
        @printf("%-9s %8.4f %8.4f %8.2f %+8.2f\n", nm, ccc(g,m), cor(g,m)^2,
                100sqrt(mean((m.-g).^2)), 100mean(m.-g))
    end
end
huerr_pv   = [r.hu_pv - r.truthHU for r in srows]
huerr_gate = [r.hu_gate - r.truthHU for r in srows if !isnan(r.hu_gate)]
@printf("\nHU bias vs truth:  geometric-PV %+.2f HU   |   clinical-gated %+.2f HU\n",
        mean(huerr_pv), mean(huerr_gate))

# ── figure ───────────────────────────────────────────────────────────────────────────
fig = CM.Figure(size = (1620, 900))
COLG = Dict("healthy"=>CM.RGBf(.20,.45,.80), "diseased"=>CM.RGBf(.85,.20,.18))
ax1 = CM.Axis(fig[1,1:2]; title="radial HU profile, 20 layers — phantom vs Oxford clinical",
    titlesize=17, xlabel="shell (mm from the coronary)", ylabel="HU at 70 keV", xticks=1:2:20)
# Shells below SHELL_MIN are still PLOTTED — nothing is dropped silently — but they do not enter
# the accuracy statistics, so the band is shaded to say which part of the curve is not scored.
SHELL_MIN > 1 && CM.vspan!(ax1, 0.5, SHELL_MIN - 0.5; color = (:gray60, 0.16))
SHELL_MIN > 1 && CM.text!(ax1, SHELL_MIN - 0.6, 1.0; space = :relative, align = (:right, :top),
    text = "not scored\n(shell < $SHELL_MIN)", fontsize = 9, color = :gray45,
    offset = (0, -4))
for g in ("healthy","diseased")
    rr = [r for r in rows if r.group==g && !isnan(r.hu_gate)]; isempty(rr) && continue
    sh = [Float64(r.shell) for r in rr]
    CM.lines!(ax1, sh, [r.truthHU for r in rr]; color=COLG[g], linewidth=2, linestyle=:dash)
    CM.scatterlines!(ax1, sh, [r.hu_gate for r in rr]; color=COLG[g], linewidth=2.6, markersize=8)
end
# This panel is the CLINICAL FAI construct, so only the HU-gated curve belongs on it. The MMD ROI's
# own mean HU is not an MMD deliverable (its deliverables are the fractions below) and plotting it
# here mixes the two lines the analysis deliberately keeps apart; the number is still printed.
#
# Two encodings are in play — colour says WHICH GROUP, line style says WHICH QUANTITY — so the
# legend is split the same way. A single flat list interleaves the two and reads as noise.
CM.Legend(fig[1,1:2],
    [[CM.LineElement(color=COLG["healthy"],  linewidth=3),
      CM.LineElement(color=COLG["diseased"], linewidth=3)],
     [CM.LineElement(color=:gray25, linewidth=2, linestyle=:dash),
      [CM.LineElement(color=:gray25, linewidth=2.6),
       CM.MarkerElement(color=:gray25, marker=:circle, markersize=8)]]],
    [["healthy", "diseased"],
     ["truth — the shell's nominal composition",
      "measured — clinical FAI ([$(Int(ADIPOSE_LO)), $(Int(ADIPOSE_HI))] HU gate)"]],
    ["colour = vessel group", "line = quantity"];
    tellwidth=false, tellheight=false, halign=:right, valign=:top, margin=(8,8,8,8),
    framevisible=false, labelsize=11, titlesize=11, titlefont=:bold, patchsize=(28,12),
    groupgap=14, rowgap=2)
ax2 = CM.Axis(fig[1,3]; title="adipose pixels per shell", titlesize=17,
    xlabel="shell (mm)", ylabel="pixels in the ROI", xticks=1:2:20, yscale=log10)
for g in ("healthy","diseased")
    rr = [r for r in rows if r.group==g]; isempty(rr) && continue
    rr = [r for r in rr if !isnan(r.m_l)]
    isempty(rr) && continue
    CM.scatterlines!(ax2, [Float64(r.shell) for r in rr], [Float64(max(r.n_pv,1)) for r in rr];
                     color=COLG[g], markersize=8, label="$g")
end
CM.axislegend(ax2; position=:rt, framevisible=false, labelsize=11)
for (col,(s,nm,c)) in enumerate(((:w,"Water",CM.RGBf(.23,.46,.69)),
                                 (:l,"Lipid",CM.RGBf(.90,.60,.10)),
                                 (:p,"Protein",CM.RGBf(.76,.27,.24))))
    g, m = stats[s].g, stats[s].m
    exd = [r for r in excl if !isnan(r.m_l)]        # nothing to plot where the MMD arm is absent
    ge = [r[Symbol("gt_",s)] for r in exd]; me = [r[Symbol("m_",s)] for r in exd]
    allg = vcat(g, ge); allm = vcat(m, me)
    lo = min(minimum(allg),minimum(allm)); hi = max(maximum(allg),maximum(allm))
    pad = 0.08*(hi-lo)+1e-3; L = (100*(lo-pad), 100*(hi+pad))
    ax = CM.Axis(fig[2,col]; title=nm, titlesize=16, aspect=CM.AxisAspect(1), limits=(L,L),
        xlabel="ground truth $(lowercase(nm)) (%)", ylabel=col==1 ? "measured (%)" : "")
    CM.lines!(ax, [L[1],L[2]], [L[1],L[2]]; color=:gray55, linestyle=:dash)
    # Excluded regions are drawn hollow rather than removed — the statistics quote srows, the plot
    # still shows what was left out and where it sits.
    for grp in ("healthy","diseased")
        se = [i for i in eachindex(exd) if exd[i].group==grp]
        isempty(se) && continue
        CM.scatter!(ax, 100 .* ge[se], 100 .* me[se]; color=:transparent,
            marker = grp=="healthy" ? :circle : :utriangle, markersize=11,
            strokewidth=1.1, strokecolor=(c,0.75))
    end
    for grp in ("healthy","diseased")
        sel = [i for i in eachindex(srows) if srows[i].group==grp]
        isempty(sel) && continue
        CM.scatter!(ax, 100 .* g[sel], 100 .* m[sel]; color=(c, grp=="healthy" ? 0.95 : 0.5),
            marker = grp=="healthy" ? :circle : :utriangle, markersize=11,
            strokewidth=0.5, strokecolor=:white)
    end
    CM.text!(ax, 0.03, 0.97; space=:relative, align=(:left,:top), fontsize=12,
        text="CCC = $(round(ccc(g,m),digits=3))\nRMSE = $(round(100sqrt(mean((m.-g).^2)),digits=2)) %\nbias = $(round(100mean(m.-g),digits=2)) %")
    if col == 3
        CM.Legend(fig[2,col],
            [[CM.MarkerElement(color=(c,0.95), marker=:circle, markersize=11),
              CM.MarkerElement(color=(c,0.5),  marker=:utriangle, markersize=11)],
             [CM.MarkerElement(color=(c,0.95), marker=:circle, markersize=11),
              CM.MarkerElement(color=:transparent, marker=:circle, markersize=11,
                               strokewidth=1.1, strokecolor=(c,0.75))]],
            [["healthy", "diseased"],
             ["in the statistics", "excluded (shell < $SHELL_MIN or n < $NMIN_STAT)"]],
            ["marker = group", "fill = scored?"];
            tellwidth=false, tellheight=false, halign=:right, valign=:bottom, margin=(6,6,6,6),
            framevisible=false, labelsize=10, titlesize=10, titlefont=:bold, rowgap=1, groupgap=8)
    end
end
CM.Label(fig[0,:], "PCAT 20-layer analysis — MMD, geometric partial-volume avoidance " *
    "(erode $ERODE px = $(round(ERODE*PX_MM, digits=2)) mm); statistics over shells >= $SHELL_MIN " *
    "with n_PV >= $NMIN_STAT; clinical FAI scored with the [$(Int(ADIPOSE_LO)), $(Int(ADIPOSE_HI))] HU gate";
    fontsize=16, font=:bold)
CM.save(joinpath(OUT,"pcat_20layer$(TAG).png"), fig; px_per_unit=2)
println("\nfigure -> ", joinpath(OUT,"pcat_20layer$(TAG).png"))

open(joinpath(OUT,"pcat_20layer$(TAG).csv"),"w") do io
    println(io,"group,shell_mm,n_geom,n_pv,n_gate,truth_HU,HU_pv,HU_gate,gt_water,gt_lipid,gt_protein,ms_water,ms_lipid,ms_protein,sem_water,sem_lipid,sem_protein,sd_lipid,in_statistics")
    for r in rows
        println(io,"$(r.group),$(r.shell),$(r.n_geom),$(r.n_pv),$(r.n_gate),$(r.truthHU),$(r.hu_pv),$(r.hu_gate),$(r.gt_w),$(r.gt_l),$(r.gt_p),$(r.m_w),$(r.m_l),$(r.m_p),$(r.sem_w),$(r.sem_l),$(r.sem_p),$(r.sd_l),$(r.instat)")
    end
end
println("csv -> ", joinpath(OUT,"pcat_20layer$(TAG).csv"))
