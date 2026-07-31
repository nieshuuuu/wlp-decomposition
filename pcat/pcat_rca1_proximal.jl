# Proximal RCA1 only — the segment the FAI is actually defined on — plus an 8-sector azimuthal
# breakdown of the same ROI.
#
# THE DEFINITION (Antonopoulos 2017 / Oikonomou 2018, as summarised in EHJ-CI 2022;23:e526 and
# Radiology CTI 2021;3:e200563): pericoronary FAI is the mean attenuation of adipose voxels
# ([-190, -30] HU) lying within a radial distance from the OUTER VESSEL WALL equal to the vessel's
# own DIAMETER. For the RCA the tracing is the proximal 40 mm, with the most proximal 10 mm excluded
# to keep the aortic wall out of the ROI — i.e. 10-50 mm measured from the ostium.
#
# WHY THE 10 mm EXCLUSION STILL APPLIES HERE even with no iodine anywhere. It is not a contrast
# artefact rule: it is there because the aortic wall's own attenuation leaks into the fat gate. In
# this phantom the aorta is whole blood (+56 HU) and the vessel wall is muscle (+43 HU), so against
# fat at -75 HU the interface contrast is still ~130 HU and the partial-volume zone is 2-3 mm wide.
# The 0-50 mm variant is reported alongside so the size of that effect is measured, not assumed.
#
# GEOMETRY IS TAKEN FROM THE PHANTOM, NOT THE RECON. The recon window is only 40 mm of z and the RCA
# ostium falls just outside it (closest approach to the aorta at the top slice is 3.98 mm, still not
# touching), so arc length cannot be referenced to the ostium from the recon alone. The ostium and
# the centreline arc length are measured on the 0.5 mm slab, then mapped onto the recon slices.
#
# Usage: julia --project=. pcat_rca1_proximal.jl [ERODE]
#   PCAT_PROX_LO / PCAT_PROX_HI  arc-length window in mm from the ostium (default 10 / 50)
#   PCAT_SHELL_MIN, PCAT_NMIN_STAT, PCAT_ACQ  as in pcat_20layer.jl
import CairoMakie as CM
using Statistics: mean, std, cor, median
using Printf: @printf
include(joinpath(@__DIR__, "pcat_common.jl"))

const D = pcat_load(get(ENV, "PCAT_ACQ", "pcat_acq_shell.jls"))
const MODEL = pcat_model()
const ERODE     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
const SHELL_MIN = parse(Int, get(ENV, "PCAT_SHELL_MIN", "3"))
const NMIN_STAT = parse(Int, get(ENV, "PCAT_NMIN_STAT", "200"))
const NMIN_TABLE = 30
const PROX_LO = parse(Float64, get(ENV, "PCAT_PROX_LO", "10.0"))
const PROX_HI = parse(Float64, get(ENV, "PCAT_PROX_HI", "50.0"))
const RCA1 = 1                                   # index into VESSELS
const NSECT = 8
const SECT_LABEL = ["A", "AL", "L", "PL", "P", "PR", "R", "AR"]
const TAG = "_e$(ERODE)_p$(Int(PROX_LO))-$(Int(PROX_HI))"

# ── 1. anatomical axes, measured from the phantom rather than assumed ─────────────────
const S3 = D.slab
let bone = findall(x -> Int(x) in (3,5,8,9), S3), heart = findall(x -> 15 <= Int(x) <= 22, S3),
    liver = findall(x -> Int(x) == 10, S3), spleen = findall(x -> Int(x) == 12, S3)
    by, hy = mean(getindex.(bone,2)), mean(getindex.(heart,2))
    lx, sx = mean(getindex.(liver,1)), mean(getindex.(spleen,1))
    by > hy || error("expected +y posterior (bone behind the heart); got bone y=$by heart y=$hy")
    sx > lx || error("expected +x patient-left (spleen left of liver); got spleen x=$sx liver x=$lx")
    @printf("anatomical axes: +y = POSTERIOR (bone y %.0f vs heart %.0f), +x = PATIENT LEFT (spleen x %.0f vs liver %.0f)\n", by, hy, sx, lx)
end
# θ = 0 at ANTERIOR, increasing toward PATIENT LEFT. Sector i is centred on 45(i-1) degrees.
azimuth(dx, dy) = mod(atand(dx, -dy), 360.0)
sector_of(θ) = mod(round(Int, θ / 45), NSECT) + 1

# ── 2. the ostium and the centreline arc length, on the 0.5 mm slab ──────────────────
const SOLID_RCA1 = vessel_solid_labels(RCA1)
cenS = Dict{Int,Tuple{Float64,Float64}}(); daoS = Dict{Int,Float64}()
for z in axes(S3, 3)
    L = @view S3[:,:,z]
    r = findall(x -> Int(x) in SOLID_RCA1, L); isempty(r) && continue
    cenS[z] = (mean(getindex.(r,1)), mean(getindex.(r,2)))
    a = findall(x -> Int(x) == AORTA, L); isempty(a) && continue
    daoS[z] = minimum(VOXMM*sqrt((ri[1]-ai[1])^2 + (ri[2]-ai[2])^2) for ri in r, ai in a)
end
const ZO = argmin(daoS)                          # ostium = closest approach to the aorta
const ZSLAB = sort(collect(keys(cenS)))
arcS = Dict{Int,Float64}(ZO => 0.0)
for z in ZO+1:maximum(ZSLAB)
    (haskey(cenS,z) && haskey(arcS,z-1)) || continue
    (x1,y1), (x2,y2) = cenS[z-1], cenS[z]
    arcS[z] = arcS[z-1] + sqrt((VOXMM*(x2-x1))^2 + (VOXMM*(y2-y1))^2 + VOXMM^2)
end
for z in ZO-1:-1:minimum(ZSLAB)
    (haskey(cenS,z) && haskey(arcS,z+1)) || continue
    (x1,y1), (x2,y2) = cenS[z+1], cenS[z]
    arcS[z] = arcS[z+1] + sqrt((VOXMM*(x2-x1))^2 + (VOXMM*(y2-y1))^2 + VOXMM^2)
end
@printf("ostium at slab z=%d (aorta gap %.2f mm); rca1 spans %.1f mm of centreline\n", ZO, daoS[ZO], maximum(values(arcS)))
# tangent tilt off z — the justification for treating axial planes as cross sections
const TILT = let t = Float64[]
    for i in 2:length(ZSLAB)
        (x1,y1), (x2,y2) = cenS[ZSLAB[i-1]], cenS[ZSLAB[i]]
        push!(t, atand(VOXMM*sqrt((x2-x1)^2+(y2-y1)^2), VOXMM*(ZSLAB[i]-ZSLAB[i-1])))
    end
    sort(t)
end
@printf("centreline tilt off the z axis: median %.1f deg, 90th percentile %.1f deg (axial planes stand in for cross sections to within this)\n", TILT[length(TILT)÷2], TILT[max(1,round(Int, 0.9length(TILT)))])

# ── 3. recon grid, valid z, decode ───────────────────────────────────────────────────
m3 = pcat_label_grid(D)
ZR, plateau = pcat_valid_z(D, m3; trim = 1)
lab = m3[:,:,ZR]; H70 = Float64.(D.hu_lo[:,:,ZR]); H150 = Float64.(D.hu_hi[:,:,ZR])
@info "myocardium plateau $(round(plateau,digits=1)) HU; using recon z $ZR"
FW, FL, FP = pcat_decode(H70, H150, MODEL)
const truthHU = pcat_truth_hu(D, 70.0)
fe = pcat_fat_erode(lab, ERODE); FATE = fe.mask
@info "PV erosion $ERODE vox: $(fe.n_before) -> $(fe.n_after) adipose pixels"

# ── 4. per-recon-slice rca1 geometry ─────────────────────────────────────────────────
# Arc length comes from the MEASURED recon->slab slice map (pcat_slab_z_of_recon), not from matching
# centroids: slab and recon share neither voxel pitch nor origin, so no coordinate is comparable
# between them. Everything else here is per-slice and stays in recon coordinates.
const SLABZ = pcat_slab_z_of_recon(D)          # recon slice -> slab slice, measured
@printf("recon slice 1..%d maps to slab z %d..%d\n", RECON_NZ, SLABZ[1], SLABZ[end])
nzr = size(lab, 3)
ARC   = fill(NaN, nzr)                          # arc length from the ostium, mm
DIAM  = fill(NaN, nzr)                          # in-plane equivalent lumen diameter, mm
RDIST = fill(NaN, size(lab))                    # signed in-plane distance from the rca1 solid
THETA = fill(NaN, size(lab))                    # azimuth about the rca1 centroid, degrees
NEAREST = falses(size(lab))                     # rca1 is the closest of the six vessels
for z in 1:nzr
    L = @view lab[:,:,z]
    solid = BitMatrix([Int(x) in SOLID_RCA1 for x in L])
    count(solid) < 5 && continue
    r = findall(solid)
    cx, cy = mean(getindex.(r,1)), mean(getindex.(r,2))
    ARC[z] = get(arcS, SLABZ[ZR[z]], NaN)
    nlum = count(x -> Int(x) == LUM0 + RCA1 - 1, L)
    DIAM[z] = 2*sqrt(nlum * PX_MM^2 / π)
    dout = edt2(solid, PX_MM); din = edt2(.!solid, PX_MM)
    RDIST[:,:,z] = dout .- din
    for j in axes(L,2), i in axes(L,1)
        THETA[i,j,z] = azimuth(i - cx, j - cy)
    end
    dmin = fill(Inf, size(L))                   # distance to the nearest OTHER vessel
    for vi in 1:length(VESSELS)
        vi == RCA1 && continue
        sv = BitMatrix([Int(x) in vessel_solid_labels(vi) for x in L])
        count(sv) < 5 && continue
        dmin .= min.(dmin, edt2(sv, PX_MM))
    end
    NEAREST[:,:,z] = dout .< dmin
end
const NSL = count(!isnan, ARC)
@printf("rca1 present on %d of %d valid recon slices; arc length %.1f .. %.1f mm from the ostium\n", NSL, nzr, minimum(filter(!isnan, ARC)), maximum(filter(!isnan, ARC)))
@printf("in-plane equivalent lumen diameter: median %.2f mm (range %.2f .. %.2f)\n", median(filter(!isnan, DIAM)), minimum(filter(!isnan, DIAM)), maximum(filter(!isnan, DIAM)))

"Slices whose arc length falls in [lo, hi] mm from the ostium."
prox_slices(lo, hi) = [z for z in 1:nzr if !isnan(ARC[z]) && lo <= ARC[z] <= hi]
const PZ = prox_slices(PROX_LO, PROX_HI)
isempty(PZ) && error("no recon slice falls in the $(PROX_LO)-$(PROX_HI) mm window; " *
                     "available arc lengths are $(round.(filter(!isnan,ARC),digits=1))")
# The FAI segment is 40 mm long. This recon is 40 mm of z of which only the myocardium-plateau
# slices survive, and the centreline is tilted, so the requested window is generally NOT fully
# covered. Say so with a number rather than quietly reporting a short segment as if it were 10-50.
const ARC_OK = extrema(ARC[PZ])
const COVER = 100 * (ARC_OK[2] - ARC_OK[1]) / (PROX_HI - PROX_LO)
COVER < 99 && @warn "the $(Int(PROX_LO))-$(Int(PROX_HI)) mm FAI segment is only " *
    "$(round(COVER, digits=0))% covered: the recon reaches $(round(ARC_OK[1],digits=1))-" *
    "$(round(ARC_OK[2],digits=1)) mm from the ostium. Everything below is that sub-segment. " *
    "Extending it needs a taller recon z window in pcat_ct_sim.jl (RECON_Z_CM), not a code change here."
@printf("\nPROXIMAL WINDOW %.0f-%.0f mm: recon slices %s (%d slices, %.1f mm of z)\n", PROX_LO, PROX_HI, string(extrema(PZ)), length(PZ), length(PZ)*DZ_MM)

# proximal + belongs-to-rca1 mask, reused by both analyses
PROX = falses(size(lab))
for z in PZ; PROX[:,:,z] .= true; end
const OWN = PROX .& NEAREST

# ── 5. per-shell statistics, restricted to proximal rca1 ─────────────────────────────
# The shell index comes from the LABEL (which is what truth is), never from the geometric distance —
# so the shell number and the ground-truth composition can never disagree. The rca1 restriction is a
# spatial mask on top of it.
rows = NamedTuple[]
for k in 1:NSHELL
    labs = Int[shell_label("healthy", k)]
    k <= K && push!(labs, fat_label(k, RCA1 - 1))
    idx = [c for c in CartesianIndices(lab) if Int(lab[c]) in labs && OWN[c]]
    isempty(idx) && continue
    pv = [c for c in idx if FATE[c] && !isnan(FW[c])]
    gt = [c for c in idx if ADIPOSE_LO <= H70[c] <= ADIPOSE_HI]
    (length(pv) < NMIN_TABLE && length(gt) < NMIN_TABLE) && continue
    hasmmd = length(pv) >= NMIN_TABLE
    t = D.gt[labs[1]]
    mm(f) = hasmmd ? f() : NaN
    push!(rows, (shell=k, n_geom=length(idx), n_pv=length(pv), n_gate=length(gt),
                 gt_w=t[1], gt_l=t[2], gt_p=t[3], truthHU=truthHU[labs[1]],
                 m_w=mm(()->mean(FW[pv])), m_l=mm(()->mean(FL[pv])), m_p=mm(()->mean(FP[pv])),
                 sem_w=mm(()->sem(FW[pv])), sem_l=mm(()->sem(FL[pv])), sem_p=mm(()->sem(FP[pv])),
                 sd_l=mm(()->std(FL[pv])), hu_pv=mm(()->mean(H70[pv])),
                 hu_gate=isempty(gt) ? NaN : mean(H70[gt]),
                 instat = hasmmd && k >= SHELL_MIN && length(pv) >= NMIN_STAT))
end
isempty(rows) && error("no shell survived the proximal rca1 mask")

println("\n", "="^96)
println("PROXIMAL RCA1 ONLY — $(Int(PROX_LO))-$(Int(PROX_HI)) mm from the ostium, erode $ERODE px")
println("="^96)
@printf("%6s %8s %8s %8s %9s %9s %9s   %8s %8s %+8s %8s %5s\n",
        "shell","n_geom","n_PV","n_gate","truth_HU","HU_PV","HU_gate",
        "GT_lip%","ms_lip%","d_lip%","SEM_lip%","stat")
for r in rows
    @printf("%6d %8d %8d %8d %9.1f %9.1f %9.1f   %8.1f %8.1f %+8.1f %8.2f %5s\n",
            r.shell, r.n_geom, r.n_pv, r.n_gate, r.truthHU, r.hu_pv, r.hu_gate,
            100r.gt_l, 100r.m_l, 100*(r.m_l-r.gt_l), 100r.sem_l, r.instat ? "yes" : "--")
end
srows = [r for r in rows if r.instat]; excl = [r for r in rows if !r.instat]
isempty(excl) || println("\nexcluded from the statistics (still tabulated):")
for r in excl
    why = isnan(r.m_l) ? "MMD ROI has only $(r.n_pv) voxels after erosion" :
          r.shell < SHELL_MIN ? "shell < $SHELL_MIN (wall boundary-artifact zone)" :
                                "n_PV $(r.n_pv) < $NMIN_STAT"
    @printf("  shell %2d — %s\n", r.shell, why)
end
stats = Dict{Symbol,Any}()
if length(srows) >= 3
    @printf("\n%-9s %8s %8s %8s %8s %9s %10s\n",
            "material","CCC","R2","RMSE_%","bias_%","GTrange_%","medSEM_%")
    for (s,nm) in ((:w,"Water"),(:l,"Lipid"),(:p,"Protein"))
        g = [r[Symbol("gt_",s)] for r in srows]; m = [r[Symbol("m_",s)] for r in srows]
        stats[s] = (g=g, m=m)
        @printf("%-9s %8.4f %8.4f %8.2f %+8.2f %9.2f %10.2f\n", nm, ccc(g,m), cor(g,m)^2,
                100sqrt(mean((m.-g).^2)), 100mean(m.-g), 100*(maximum(g)-minimum(g)),
                100median([r[Symbol("sem_",s)] for r in srows]))
    end
    hg = [r.hu_gate - r.truthHU for r in srows if !isnan(r.hu_gate)]
    @printf("HU bias vs truth:  geometric-PV %+.2f HU  |  clinical-gated %+.2f HU\n", mean([r.hu_pv - r.truthHU for r in srows]), mean(hg))
else
    for s in (:w,:l,:p); stats[s] = (g=Float64[], m=Float64[]); end
    println("\nfewer than 3 shells survive — no accuracy statistics for this window")
end

# ── 6. the FAI itself, by the definition, and per azimuthal sector ───────────────────
# ROI = adipose-gated voxels with 0 < r <= the vessel's own in-plane diameter, inside the proximal
# window and closer to rca1 than to any other vessel.
const DMED = median(filter(!isnan, DIAM))
faiROI(mask) = [c for c in CartesianIndices(lab) if mask[c] &&
                0 < RDIST[c] <= DIAM[c[3]] && ADIPOSE_LO <= H70[c] <= ADIPOSE_HI]
function fai_report(lo, hi, tag)
    zz = prox_slices(lo, hi); isempty(zz) && return nothing
    m = falses(size(lab)); for z in zz; m[:,:,z] .= true; end
    roi = faiROI(m .& NEAREST)
    isempty(roi) && return nothing
    tv = [truthHU[Int(lab[c])] for c in roi if haskey(truthHU, Int(lab[c]))]
    @printf("  %-22s n = %6d   FAI = %+7.2f HU   truth %+7.2f HU   bias %+6.2f HU\n", tag, length(roi), mean(H70[roi]), isempty(tv) ? NaN : mean(tv), isempty(tv) ? NaN : mean(H70[roi]) - mean(tv))
    roi
end
println("\nFAI by the definition (gated voxels within one vessel diameter, median D = " *
        "$(round(DMED,digits=2)) mm):")
roi_std = fai_report(PROX_LO, PROX_HI, "$(Int(PROX_LO))-$(Int(PROX_HI)) mm (standard)")
fai_report(0.0, PROX_HI, "0-$(Int(PROX_HI)) mm (no exclusion)")
fai_report(0.0, PROX_LO, "0-$(Int(PROX_LO)) mm (the excluded bit)")

sect = NamedTuple[]
for i in 1:NSECT
    roi = [c for c in roi_std if sector_of(THETA[c]) == i]
    mmd = [c for c in CartesianIndices(lab) if OWN[c] && 0 < RDIST[c] <= DIAM[c[3]] &&
           sector_of(THETA[c]) == i && FATE[c] && !isnan(FW[c])]
    tv = [truthHU[Int(lab[c])] for c in roi if haskey(truthHU, Int(lab[c]))]
    gtw = [D.gt[Int(lab[c])][1] for c in mmd if haskey(D.gt, Int(lab[c]))]
    gtl = [D.gt[Int(lab[c])][2] for c in mmd if haskey(D.gt, Int(lab[c]))]
    gtp = [D.gt[Int(lab[c])][3] for c in mmd if haskey(D.gt, Int(lab[c]))]
    push!(sect, (i=i, lab=SECT_LABEL[i], n_gate=length(roi), n_pv=length(mmd),
                 fai=isempty(roi) ? NaN : mean(H70[roi]),
                 fai_sem=isempty(roi) ? NaN : sem(H70[roi]),
                 truthHU=isempty(tv) ? NaN : mean(tv),
                 m_w=isempty(mmd) ? NaN : mean(FW[mmd]), m_l=isempty(mmd) ? NaN : mean(FL[mmd]),
                 m_p=isempty(mmd) ? NaN : mean(FP[mmd]),
                 gt_w=isempty(gtw) ? NaN : mean(gtw), gt_l=isempty(gtl) ? NaN : mean(gtl),
                 gt_p=isempty(gtp) ? NaN : mean(gtp),
                 rmean=isempty(roi) ? NaN : mean(RDIST[c] for c in roi)))
end
println("\n8 azimuthal sectors of the same ROI — sector 1 (A) is centred on ANTERIOR, " *
        "increasing toward PATIENT LEFT")
@printf("%3s %5s %8s %8s %9s %9s %8s   %8s %8s %+8s\n",
        "sec","dir","n_gate","n_PV","FAI_HU","truth_HU","d_HU","GT_lip%","ms_lip%","d_lip%")
for s in sect
    @printf("%3d %5s %8d %8d %9.2f %9.2f %+8.2f   %8.2f %8.2f %+8.2f\n",
            s.i, s.lab, s.n_gate, s.n_pv, s.fai, s.truthHU, s.fai - s.truthHU,
            100s.gt_l, 100s.m_l, 100*(s.m_l - s.gt_l))
end
let f = [s.fai for s in sect if !isnan(s.fai)], t = [s.truthHU for s in sect if !isnan(s.truthHU)]
    @printf("\nsector spread: measured FAI %.2f HU peak-to-peak (sd %.2f), TRUTH %.2f HU peak-to-peak (sd %.2f)\n", maximum(f)-minimum(f), std(f), maximum(t)-minimum(t), std(t))
    println("  Truth varies between sectors because the FEBio fat pad is asymmetric, so part of any " *
            "measured\n  azimuthal spread is real composition, not measurement error. Both are printed above.")
end

# ── 7. figures ───────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "pcat_rca1_proximal_figs.jl"))

open(joinpath(OUT, "pcat_rca1_proximal$(TAG).csv"), "w") do io
    println(io, "# proximal rca1 only, $(Int(PROX_LO))-$(Int(PROX_HI)) mm from the ostium " *
                "(ostium at slab z=$ZO), erode $ERODE px")
    println(io, "shell_mm,n_geom,n_pv,n_gate,truth_HU,HU_pv,HU_gate,gt_water,gt_lipid,gt_protein," *
                "ms_water,ms_lipid,ms_protein,sem_water,sem_lipid,sem_protein,sd_lipid,in_statistics")
    for r in rows
        println(io, "$(r.shell),$(r.n_geom),$(r.n_pv),$(r.n_gate),$(r.truthHU),$(r.hu_pv)," *
                    "$(r.hu_gate),$(r.gt_w),$(r.gt_l),$(r.gt_p),$(r.m_w),$(r.m_l),$(r.m_p)," *
                    "$(r.sem_w),$(r.sem_l),$(r.sem_p),$(r.sd_l),$(r.instat)")
    end
end
open(joinpath(OUT, "pcat_rca1_sectors$(TAG).csv"), "w") do io
    println(io, "# 8 azimuthal sectors, sector 1 centred on ANTERIOR increasing toward PATIENT LEFT")
    println(io, "sector,direction,n_gate,n_pv,fai_HU,fai_standard_error_of_the_mean,truth_HU," *
                "ms_water,ms_lipid,ms_protein,gt_water,gt_lipid,gt_protein,mean_radial_distance_mm")
    for s in sect
        println(io, "$(s.i),$(s.lab),$(s.n_gate),$(s.n_pv),$(s.fai),$(s.fai_sem),$(s.truthHU)," *
                    "$(s.m_w),$(s.m_l),$(s.m_p),$(s.gt_w),$(s.gt_l),$(s.gt_p),$(s.rmean)")
    end
end
println("\ncsv -> ", joinpath(OUT, "pcat_rca1_proximal$(TAG).csv"))
println("csv -> ", joinpath(OUT, "pcat_rca1_sectors$(TAG).csv"))
