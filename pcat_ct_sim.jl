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
            "projects/pvat_study_001_febio/xcat/act_kpaP40_K6_640x640x348_uint8.raw"
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
const CSV = "/Volumes/Molloilab/Shu Nie/water-lipid-protein/oxford_wlp_composition.csv"
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
        29 => M.softtissue,                       # pericardium
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
    (mats = mats, gt = gt)
end

# ── load + crop. Keep the FULL transaxial body (attenuation/beam-hardening must be real);
#    crop only in z to a slab through the coronaries, and zoom the RECON FOV to the heart. ──
function load_phantom_slab(; nz_slab = 48)
    v = Array{UInt8}(undef, NX, NY, NZ)
    read!(RAW, v)
    fatmask = (v .>= 40) .& (v .< LUM0 + 6)
    zc = [count(@view fatmask[:, :, k]) for k in 1:NZ]
    zbest = argmax([sum(@view zc[max(k - nz_slab ÷ 2, 1):min(k + nz_slab ÷ 2, NZ)]) for k in 1:NZ])
    z0 = clamp(zbest - nz_slab ÷ 2, 1, NZ - nz_slab + 1)
    slab = v[:, :, z0:(z0 + nz_slab - 1)]
    @info "z slab $(z0):$(z0+nz_slab-1) (richest PCAT), fat voxels in slab = $(count(slab .>= 40 .&& slab .< LUM0+6))"
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
const RECON_NZ = 3

function run_acq(pg; views = 984, collimation = 2.5, seed = 1234, zmed = 1)
    matrix = (RECON_N, RECON_N, RECON_NZ)
    plow  = BS.CTProtocol(kVp = 80,  mA = 407 * 0.65, views = views, rotation_time = 0.5,
                          collimation_mm = collimation, additional_filters = [("Al", 4.5)])
    phigh = BS.CTProtocol(kVp = 140, mA = 405 * 0.35, views = views, rotation_time = 0.5,
                          collimation_mm = collimation, additional_filters = [("Al", 4.5)])
    so = BS.SimOptions(fidelity = :eict, use_noise = true, use_fill_factor = false,
                       use_optical_crosstalk = false, use_scatter = false,
                       projector = :dd_fast, seed = seed)
    ro = BS.ReconOptions(matrix_size = matrix, fov_cm = RECON_FOV_CM, z_cm = 0.1875)
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
const SIMCACHE = joinpath(OUT, "pcat_acq.jls")
if STAGE in ("all", "sim") || !isfile(SIMCACHE)
    @info "building phantom + materials (no iodine; Oxford WLP subrings)"
    ph = load_phantom_slab()
    mg = build_materials()
    for l in unique(ph.slab)
        haskey(mg.mats, Int(l)) || (mg.mats[Int(l)] = BS.XA.Materials.softtissue)
    end
    @info "materials: $(length(mg.mats)) labels, $(length(mg.gt)) PCAT subring materials"
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
