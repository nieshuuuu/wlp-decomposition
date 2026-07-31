# CT sim of the FEBio PCAT phantom + water/lipid/protein decomposition accuracy.
#
#   phantom : act_kpaP40_K6 (FEBio-grown PCAT, 6 x 1mm subrings, 0.5mm isotropic)
#   materials: NO IODINE anywhere — blood pool and coronary lumen are whole blood.
#             Each fat subring gets the Oxford PCAT radial composition at its own distance
#             (oxford_wlp_composition.csv), built as a water/lipid/protein volume mixture.
#   scan    : 80/140 kVp EICT -> Cong water/iodine -> FBP -> VMI at 70 and 150 keV,
#             the exact chain wlp_model_70_150.toml was fitted on. FOV zoomed to the heart.
#   decode  : poly2 surface from wlp_model_70_150.toml -> (f_w, f_l, f_p) maps
#   compare : region-mean GT vs measured per (vessel, subring) ROI, per material.
#
# Usage: julia --project=. pcat_ct_sim.jl [STAGE]     STAGE = all | sim | decode
import BasisSimulator as BS
import Metal, TOML, Random
using Unitful: @u_str
using Statistics: mean, std, cor, median
using Printf: @printf, @sprintf
using Serialization: serialize, deserialize

to_gpu(x) = Metal.functional() ? Metal.MtlArray(x) : x
const STAGE = length(ARGS) >= 1 ? ARGS[1] : "all"
const OUT = joinpath(@__DIR__, "pcat_ct")
mkpath(OUT)

# ── phantom geometry (SSoT: matches the raster written by gate_subrings.py) ────────────
const RAW = "/Users/shunie/Developer/PCATSim.jl/.worktrees/febio-fat-growth/" *
            "projects/pvat_study_001_febio/xcat/act_kpaP40_K6shell_640x640x348_uint8.raw"
const NX, NY, NZ = 640, 640, 348
const VOXMM = 0.5
const K = 6                                   # subrings
const VESSELS = ["rca1", "rca2", "lad1", "lad2", "lad3", "lcx"]
fat_label(k, i) = 40 + 6 * (K - k) + i        # subring k (1=innermost), vessel i (0-based)
const WALL0, LUM0 = 40 + 6K, 46 + 6K          # 76, 82

# ── Oxford PCAT composition: subring k takes the profile at distance k mm ──────────────
# The CSV is a radial gradient (1-20mm) of water/lipid/protein VOLUME fractions derived from
# single-energy 120kVp HU with a Woodard&White adipose prior. Healthy and diseased differ by
# only ~0.03 in f_l over 1-6mm, so a single group gives almost no dynamic range to regress
# against; both groups are used, assigned alternately per vessel, which is also the clinically
# meaningful contrast (FAI is a disease marker).
# TISSUE-domain composition, not the clinical CSV. The clinical numbers are already blurred by
# the scanner's point spread; assigning them as phantom TRUTH and then simulating a CT blurs a
# second time, so the simulated measurement can never sit on the clinical curve. This file is the
# profile that, blurred ONCE by this simulation's own point spread (sigma 1.05mm, FWHM 2.47mm),
# reproduces the clinical curve. Produced by pcat_deconv_design.jl. Set PCAT_USE_CLINICAL=1 to
# fall back to the raw clinical CSV for comparison.
const USE_CLINICAL = get(ENV, "PCAT_USE_CLINICAL", "0") == "1"
const CSV = USE_CLINICAL ?
    "/Volumes/Molloilab/Shu Nie/water-lipid-protein/oxford_wlp_composition.csv" :
    joinpath(@__DIR__, "pcat_ct", "oxford_deconvolved_composition.csv")
const VESSEL_GROUP = Dict("rca1" => "healthy", "lcx" => "healthy", "lad1" => "healthy",
                          "lad2" => "diseased", "rca2" => "diseased", "lad3" => "diseased")

function load_oxford()
    comp = Dict{Tuple{String, Int}, NTuple{3, Float64}}()
    hdr = nothing
    for ln in eachline(CSV)
        startswith(ln, "#") && continue
        f = split(strip(ln), ',')
        if hdr === nothing
            hdr = f; continue
        end
        ix(n) = findfirst(==(n), hdr)
        ix("group") === nothing && continue
        g = f[ix("group")]; d = round(Int, parse(Float64, f[ix("distance_mm")]))
        comp[(g, d)] = (parse(Float64, f[ix("water_volume_fraction")]),
                        parse(Float64, f[ix("lipid_volume_fraction")]),
                        parse(Float64, f[ix("protein_volume_fraction")]))
    end
    comp
end
const OXFORD = load_oxford()

# ── water / lipid / protein volume mixture -> XA.Material (from wlp_decomposition.jl) ──
const PROTEIN = BS.XA.Material("protein_WW1986", 0.0, 0.0u"eV", 1.35u"g/cm^3",
                               Dict{Int, Float64}(1 => 0.066, 6 => 0.534, 7 => 0.170,
                                                  8 => 0.220, 16 => 0.010))
const WATER = BS.XA.Materials.basis_water
const LIPID = BS.XA.Materials.basis_lipid
ρval(m) = BS.XA.val(m.density)
const ΡW, ΡL, ΡP = ρval(WATER), ρval(LIPID), ρval(PROTEIN)

function wlp_material(fw, fl, fp; name = "wlp")
    # trust boundary: fractions arrive from a CSV and become simulated matter + stored GT —
    # a negative fraction would build a physically meaningless material without complaint
    (all((fw, fl, fp) .>= 0.0) && fw + fl + fp ≈ 1.0) ||
        error("wlp_material: ($fw, $fl, $fp) is not a volume-fraction composition")
    ρ = fw * ΡW + fl * ΡL + fp * ΡP
    mf = (fw * ΡW / ρ, fl * ΡL / ρ, fp * ΡP / ρ)          # volume -> mass fractions
    comp = Dict{Int, Float64}()
    for (m, wm) in ((WATER, mf[1]), (LIPID, mf[2]), (PROTEIN, mf[3])), (Z, f) in m.composition
        comp[Z] = get(comp, Z, 0.0) + wm * f
    end
    BS.XA.Material(name, 0.0, 0.0u"eV", ρ * u"g/cm^3", comp)
end

# ── label -> material. NO IODINE ANYWHERE. ────────────────────────────────────────────
# XCAT activity labels for this phantom; anything unlisted falls back to soft tissue.
function build_materials()
    M = BS.XA.Materials
    mats = Dict{Int, Any}(
        0  => M.air,
        1  => M.softtissue,                       # body background
        2  => M.cartilage,
        3  => M.corticalbone, 5 => M.corticalbone, 8 => M.corticalbone, 9 => M.corticalbone,
        4  => M.muscle,
        6  => M.lung,
        7  => M.softtissue,
        10 => M.liver, 11 => M.kidney, 12 => M.spleen, 13 => M.softtissue, 14 => M.spleen,
        23 => M.softtissue, 24 => M.softtissue, 25 => M.softtissue,
        27 => M.softtissue, 28 => M.wholeblood,   # aorta — WHOLE BLOOD, no contrast
        29 => M.adipose,                          # pericardial FAT — 193.9 mL over 113 mm of z,
                                                  # far too large for a membrane; softtissue here
                                                  # (53 HU) was also ~3 HU from blood, which is
                                                  # what flattened the whole mediastinum.
    )
    for l in 15:18; mats[l] = M.heart; end                    # myocardium
    for l in 19:22; mats[l] = M.wholeblood; end               # chambers — NO IODINE
    for i in 0:5
        mats[WALL0 + i] = M.muscle                            # vessel wall
        mats[LUM0 + i]  = M.wholeblood                        # coronary lumen — NO IODINE
    end
    gt = Dict{Int, NTuple{3, Float64}}()                      # label -> true (fw, fl, fp)
    for (i, v) in enumerate(VESSELS), k in 1:K
        f = OXFORD[(VESSEL_GROUP[v], k)]
        l = fat_label(k, i - 1)
        mats[l] = wlp_material(f...; name = "pcat_$(v)_sub$(k)")
        gt[l] = f
    end
    # 20 distance shells in the pericardial fat (add_distance_shells.py). Composition is a
    # function of distance and group ONLY — the same rule the grown PCAT follows — so the
    # phantom finally has ground truth across the full 1-20mm range the Oxford paper reports.
    for (grp, base) in (("healthy", 88), ("diseased", 108)), k in 1:20
        f = OXFORD[(grp, k)]
        l = base + k - 1
        mats[l] = wlp_material(f...; name = "shell_$(grp)_$(k)mm")
        gt[l] = f
    end
    (mats = mats, gt = gt)
end

# ── load + crop. Keep the FULL transaxial body (attenuation/beam-hardening must be real);
#    crop only in z to a slab through the coronaries, and zoom the RECON FOV to the heart. ──
# Cover the WHOLE heart in z. The PCAT tree spans z 30-225 (98 mm) and the myocardium 37-232,
# so a thin slab samples almost none of it. The slab is the heart extent plus margin; how much of
# it the recon actually reconstructs is set by the beam collimation below.
# The slab only has to span what the beam actually traverses: the reconstructed z window plus
# the cone spread. At 40 mm collimation the cone half-angle is ~1.8 deg, so rays stay within a
# few mm of the recon window; a 2x margin is generous. Carrying the full 127 mm heart when only
# 40 mm is reconstructed just makes the projector walk voxels that never enter the image.
function load_phantom_slab(; margin = 8)
    v = Array{UInt8}(undef, NX, NY, NZ)
    read!(RAW, v)
    heart = (v .>= 15) .& (v .<= 22)
    fat = (v .>= 40) .& (v .< LUM0 + 6)
    zs = [k for k in 1:NZ if any(@view heart[:, :, k]) || any(@view fat[:, :, k])]
    zmid = (minimum(zs) + maximum(zs)) ÷ 2
    half = round(Int, RECON_Z_CM * 10 / VOXMM)          # 2x the recon z window
    z0 = max(zmid - half, minimum(zs) - margin, 1)
    z1 = min(zmid + half, maximum(zs) + margin, NZ)
    slab = v[:, :, z0:z1]
    @info "z slab $(z0):$(z1) = $(z1-z0+1) slices ($(round((z1-z0+1)*VOXMM, digits=1)) mm), " *
          "covers the whole heart; PCAT voxels in slab = $(count(slab .>= 40 .&& slab .< LUM0+6))"
    (slab = slab, z0 = z0)
end

# ── acquisition: identical chain to wlp_decomposition.jl run_acq, FOV zoomed ───────────
const SCANNER = BS.Scanner(source_to_isocenter = 625.6, source_to_detector = 1100.0,
    detector_rows = 256, detector_cols = 1300, detector_row_size = 0.625, detector_col_size = 0.6,
    focal_spot_width = 1.0, focal_spot_length = 1.0, target_angle = 10.0,
    flat_filter_material = :aluminum, flat_filter_thickness = 2.5, bowtie_filter = :none,
    detector_material = :lumex, detector_depth = 3.0, fill_factor_row = 0.9,
    fill_factor_col = 0.9, electronic_noise = 0, detection_gain = 10.0)
const ELO, EHI = 70.0, 150.0
const RECON_N = 512
const RECON_FOV_CM = 18.0                    # ZOOMED to the heart: 180mm / 512 = 0.352 mm/px
const RECON_Z_CM = 4.0                       # 40 mm of z — half the heart, for a first look
const RECON_NZ = 40                          # 1 mm slices — real cardiac CT rarely resolves 0.5 mm

function run_acq(pg; views = 984, collimation = 40.0, seed = 1234, zmed = 1)
    matrix = (RECON_N, RECON_N, RECON_NZ)
    plow  = BS.CTProtocol(kVp = 80,  mA = 407 * 0.65, views = views, rotation_time = 0.5,
                          collimation_mm = collimation, additional_filters = [("Al", 4.5)])
    phigh = BS.CTProtocol(kVp = 140, mA = 405 * 0.35, views = views, rotation_time = 0.5,
                          collimation_mm = collimation, additional_filters = [("Al", 4.5)])
    so = BS.SimOptions(fidelity = :eict, use_noise = true, use_fill_factor = false,
                       use_optical_crosstalk = false, use_scatter = false,
                       projector = :dd_fast, seed = seed)
    ro = BS.ReconOptions(matrix_size = matrix, fov_cm = RECON_FOV_CM, z_cm = RECON_Z_CM)
    _sim(p) = begin
        ws = BS.create_eict_workspace(SCANNER, p, so, ro, pg)
        BS.simulate!(ws, pg, p, so)
        r = (sino = Array(ws.sinogram), geom = ws.geom); ws = nothing; GC.gc(true); r
    end
    slo = _sim(plow); shi = _sim(phigh)
    iod = BS.XA.Elements.Iodine; wat = BS.XA.Materials.water
    eL, ŵL = BS.resolve_source_spectrum_full(so, plow;  scanner = SCANNER, geom = slo.geom, phantom = pg)
    eH, ŵH = BS.resolve_source_spectrum_full(so, phigh; scanner = SCANNER, geom = shi.geom, phantom = pg)
    μρ(m, e) = Float32[Float32(BS.compute_mass_μ_at_energy(m, Float64(E))) for E in e]
    basis = (ŵ_L = ŵL, p_L = μρ(iod, eL), q_L = μρ(wat, eL),
             ŵ_H = ŵH, p_H = μρ(iod, eH), q_H = μρ(wat, eH))
    slo_g = to_gpu(Float32.(slo.sino)); shi_g = to_gpu(Float32.(shi.sino))
    sy = similar(slo_g); fill!(sy, 0f0); sc = similar(slo_g); fill!(sc, 0f0)
    cws = BS.create_cong_workspace(slo_g, basis)
    BS.apply_cong!(cws, sy, sc, slo_g, shi_g; water_basis = (a = 0f0, c = 1f0))
    siod = Array(sy); swat = Array(sc)
    slo_g = shi_g = sy = sc = cws = nothing; GC.gc(true)
    _fbp(s) = begin
        g = to_gpu(Float32.(s))
        ws = BS.create_fdk_recon_workspace(g, slo.geom, matrix; filter = BS.SoftFilter())
        r = Array(BS.reconstruct!(ws, g, slo.geom)); ws = g = nothing; GC.gc(true); Float32.(r)
    end
    viod = _fbp(siod); vwat = _fbp(swat)
    if zmed > 0
        viod = BS.apply_median_z(viod; adjacent_slices = zmed)
        vwat = BS.apply_median_z(vwat; adjacent_slices = zmed)
    end
    ciod = viod .* 1000f0
    (hu_lo = BS.synth_vmi_2basis(vwat, ciod; energy_keV = ELO),
     hu_hi = BS.synth_vmi_2basis(vwat, ciod; energy_keV = EHI), geom = slo.geom)
end

# ══ STAGE: simulate ═══════════════════════════════════════════════════════════════════
const SIMCACHE = joinpath(OUT, USE_CLINICAL ? "pcat_acq.jls" : "pcat_acq_shell.jls")
if STAGE in ("all", "sim") || !isfile(SIMCACHE)
    @info "building phantom + materials (no iodine; " * (USE_CLINICAL ? "CLINICAL" : "TISSUE-domain") * " WLP subrings from $(basename(CSV)))"
    ph = load_phantom_slab()
    mg = build_materials()
    for l in unique(ph.slab)
        haskey(mg.mats, Int(l)) || (mg.mats[Int(l)] = BS.XA.Materials.softtissue)
    end
    @info "materials: $(length(mg.mats)) labels, $(length(mg.gt)) PCAT subring materials"
    let μw70 = BS.compute_μ_at_energy(WATER, ELO), μw150 = BS.compute_μ_at_energy(WATER, EHI)
        println("  label -> material audit (theoretical HU):")
        for (l, nm) in ((1,"background"),(4,"muscle"),(6,"lung"),(15,"myocardium"),(19,"chamber blood"),
                        (28,"aorta"),(29,"pericardial fat"),(WALL0,"vessel wall"),(LUM0,"coronary lumen"))
            m = mg.mats[l]
            @printf("    %3d %-16s %-22s HU70 %+8.2f  HU150 %+8.2f\n", l, nm, m.name,
                    1000*(BS.compute_μ_at_energy(m,ELO)-μw70)/μw70,
                    1000*(BS.compute_μ_at_energy(m,EHI)-μw150)/μw150)
        end
    end
    for k in 1:K
        f = OXFORD[("healthy", k)]
        m = wlp_material(f...)
        μw = BS.compute_μ_at_energy(WATER, ELO)
        hu70 = 1000 * (BS.compute_μ_at_energy(m, ELO) - μw) / μw
        μw2 = BS.compute_μ_at_energy(WATER, EHI)
        hu150 = 1000 * (BS.compute_μ_at_energy(m, EHI) - μw2) / μw2
        @printf("  subring%d healthy fw=%.4f fl=%.4f fp=%.4f -> HU70=%+7.2f HU150=%+7.2f\n",
                k, f..., hu70, hu150)
    end
    phantom = BS.Phantom(to_gpu(ph.slab), mg.mats, (VOXMM / 10, VOXMM / 10, VOXMM / 10))
    @info "simulating 80/140 kVp, $(RECON_N)^2 @ $(RECON_FOV_CM) cm FOV ($(round(RECON_FOV_CM*10/RECON_N,digits=3)) mm/px)"
    t = @elapsed acq = run_acq(phantom; seed = 4242)
    @info "acquisition done in $(round(t/60,digits=1)) min"
    serialize(SIMCACHE, (hu_lo = Array(acq.hu_lo), hu_hi = Array(acq.hu_hi),
                         geom = acq.geom, slab = ph.slab, z0 = ph.z0, gt = mg.gt))
    @info "cached -> $SIMCACHE"
end
println("stage `sim` complete — run with STAGE=decode for the accuracy analysis")
