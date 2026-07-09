# export_raw.jl — dump the wlp CT scans as ImageJ-openable raws, grouped by role into
#   data/recon/calibration/  (noise-measurement sims — the σ ladder is fit on these)
#   data/recon/test/         (held-out circular + sector sims)
# Each scan is the FULL reconstructed z-stack (512×512×3), not just the mid slice.
# Reads the wlp_*_cache_$(PTAG).jls caches (regenerate them by running the notebook).
# Run:  julia --startup-file=no export_raw.jl   ·   WLP_PAIR_TAG=40_70 to pick another pair.
#
# Byte layout: column-major Float32/UInt8, little-endian, NO dim2-reverse — the notebook figures
# use yreversed=true, so a direct write already matches them (ImageJ row 0 = top). A 512×512×3
# array writes as 3 consecutive frames (ImageJ: Number of images = 3). The 3 z-slices are the
# zmed=1 cross-z-median recon of a z-uniform (extruded) phantom, so they are near-identical
# denoised planes. Insert geometry is fixed per family, so all circular sims share one label
# stack and all sector sims share another; label_comps.csv gives per-label (fw,fl,fp) truth.

using Serialization, Printf

const PTAG = get(ENV, "WLP_PAIR_TAG", "70_150")
const LOK, HIK = split(PTAG, "_")
const ROD0 = 8
const RECON = joinpath(@__DIR__, "data", "recon")

function cache(name)
    f = joinpath(@__DIR__, "wlp_$(name)_cache_$(PTAG).jls")
    isfile(f) ? deserialize(f) : error("missing cache $f — run the notebook to regenerate")
end

dims(M) = "$(size(M,1))x$(size(M,2))x$(size(M,3))"          # 2D → …x1, 3D → …x3
w_f32(dir, base, kev, M) = write(joinpath(dir, "$(base)_vmi$(kev)keV_$(dims(M))_float32.raw"), Array{Float32}(M))
w_u8(dir, base, M)       = write(joinpath(dir, "$(base)_$(dims(M))_uint8.raw"), Array{UInt8}(M))
scan(dir, base, s)       = (w_f32(dir, base, LOK, s.hu40); w_f32(dir, base, HIK, s.hu70))
labels(dir, base, m2, nz) = w_u8(dir, base, repeat(m2, 1, 1, nz))   # z-uniform ⇒ repeat 2D map to match stack

function comps_csv(dir, rows)                     # rows :: Vector{Tuple{name, comps}}
    open(joinpath(dir, "label_comps.csv"), "w") do io
        println(io, "scan,label,f_water,f_lipid,f_protein")
        for (name, comps) in rows, (k, c) in enumerate(comps)
            @printf(io, "%s,%d,%.4f,%.4f,%.4f\n", name, ROD0 - 1 + k, c[1], c[2], c[3])
        end
    end
end

rm(RECON; recursive=true, force=true)
CAL = joinpath(RECON, "calibration"); TST = joinpath(RECON, "test"); mkpath(CAL); mkpath(TST)

# ── calibration / noise-measurement: 4 σ-ladder sims + the delivered-map thorax ──
Dm = cache("sim"); crows = Tuple{String,Any}[]; NZ = size(Dm.map70v, 3)
for (i, s) in enumerate(Dm.calsims)
    scan(CAL, "noise_sim$i", s); push!(crows, ("noise_sim$i", s.comps))
end
scan(CAL, "deliveredmap", (hu40=Dm.map40v, hu70=Dm.map70v)); push!(crows, ("deliveredmap", Dm.mcomps))
labels(CAL, "labels", Dm.map_m2, NZ)              # shared circular-geometry inserts (labels 8..20)
comps_csv(CAL, crows)

# ── test / held-out: 5 circular + 4 sector + the sector delivered-map thorax ──
Dt = cache("test"); DS = cache("sect"); trows = Tuple{String,Any}[]
for (i, s) in enumerate(Dt.testsims)
    scan(TST, "circular_sim$i", s); push!(trows, ("circular_sim$i", s.comps))
end
for (i, s) in enumerate(DS.sectsims)
    scan(TST, "sector_sim$i", s); push!(trows, ("sector_sim$i", s.comps))
end
scan(TST, "sector_deliveredmap", (hu40=DS.smap40v, hu70=DS.smap70v)); push!(trows, ("sector_deliveredmap", DS.scomps))
labels(TST, "labels_circular", Dm.map_m2, NZ)     # circular sims share the calibration label stack
labels(TST, "labels_sector",   DS.smap_m2, NZ)    # sector-geometry inserts (labels 8..23)
comps_csv(TST, trows)

open(joinpath(RECON, "README_imagej.txt"), "w") do io
    print(io, """
ImageJ → File → Import → Raw…
  Image type       = 32-bit Real   (*_float32.raw)   |   8-bit  (*_labels*_uint8.raw)
  Width            = nx   (first number in _<nx>x<ny>x<nz>_)
  Height           = ny
  Number of images = nz   (= $(NZ) here; each file is the full reconstructed z-stack)
  ☑ Little-endian byte order

Column-major, written directly (NOT dim2-reversed): the notebook figures use yreversed=true,
so ImageJ row 0 = top matches them. Each file holds $(NZ) z-slices as consecutive frames. The
phantom is z-uniform (2D inserts extruded along z) and the recon applies a zmed=1 cross-z
median, so the $(NZ) slices are near-identical denoised planes — scroll z to confirm.

Layout — grouped by role:
  calibration/  noise-measurement sims the σ(HU) ladder is fit on:
                  noise_sim1..4   (+ deliveredmap = the delivered-decomposition-map thorax)
                labels_*.raw  = shared insert stack (labels 8..20)
  test/         held-out validation:
                  circular_sim1..5  ·  sector_sim1..4  (+ sector_deliveredmap)
                labels_circular = circular inserts (8..20)  ·  labels_sector = sector inserts (8..23)

VMI keV pair: $(LOK) / $(HIK).  label_comps.csv (per folder) → each scan's per-label
(f_water, f_lipid, f_protein) ground truth. Circular scans use labels_circular, sector scans
use labels_sector.
""")
end

# round-trip check: byte layout must reconstruct the source stack exactly
p = joinpath(CAL, "noise_sim1_vmi$(LOK)keV_512x512x$(NZ)_float32.raw")
back = reshape(reinterpret(Float32, read(p)), 512, 512, NZ)
@assert back == Array{Float32}(Dm.calsims[1].hu40) "raw round-trip mismatch — byte layout wrong"

ncal = 2 * (length(Dm.calsims) + 1); ntst = 2 * (length(Dt.testsims) + length(DS.sectsims) + 1)
println("round-trip OK ($(NZ)-slice stacks).  calibration/: $ncal scans + 1 label + csv   test/: $ntst scans + 2 labels + csv")
