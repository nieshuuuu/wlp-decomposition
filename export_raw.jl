# export_raw.jl — dump the wlp calibration + validation VMI CTs (and label rasters) as
# ImageJ-openable raws under data/recon/. Reads the existing wlp_*_$(PTAG).jls caches
# (no GPU re-sim needed). Run: `julia --startup-file=no export_raw.jl`
#
# Byte layout: column-major Float32/UInt8, little-endian, NO dim2-reverse. The notebook
# figures use yreversed=true, so a direct write already matches them (ImageJ row 0 = top =
# dim2 index 1). ponytail: this is why we don't reuse src save_vmi_raw — that one reverses
# dims=2 for the reference notebook, whose figures are NOT yreversed.

using Serialization, Printf

const PTAG = get(ENV, "WLP_PAIR_TAG", "70_150")      # cache tag = "$(low)_$(high)" keV
const LOK, HIK = split(PTAG, "_")                    # keV strings for filenames
const ROD0 = 8                                       # first insert label (SSoT: notebook)
const OUT  = joinpath(@__DIR__, "data", "recon"); mkpath(OUT)

function cache(name)
    f = joinpath(@__DIR__, "wlp_$(name)_cache_$(PTAG).jls")
    isfile(f) ? deserialize(f) : (@warn "missing cache" f; nothing)
end

dims(M) = "$(size(M,1))x$(size(M,2))x1"
function w_f32(variant, kev, M)
    n = "$(variant)_vmi$(kev)keV_$(dims(M))_float32.raw"
    write(joinpath(OUT, n), Array{Float32}(M)); n
end
function w_u8(variant, M)
    n = "$(variant)_labels_$(dims(M))_uint8.raw"
    write(joinpath(OUT, n), Array{UInt8}(M)); n
end

# label → (fw,fl,fp) truth table, so the uint8 raster is decodable outside Julia
function w_comps(variant, comps)
    open(joinpath(OUT, "$(variant)_label_comps.csv"), "w") do io
        println(io, "label,f_water,f_lipid,f_protein")
        for (k, c) in enumerate(comps)
            @printf(io, "%d,%.4f,%.4f,%.4f\n", ROD0 - 1 + k, c[1], c[2], c[3])
        end
    end
end

emitted = String[]
function variant(name, lo, hi, labels, comps)
    push!(emitted, w_f32(name, LOK, lo), w_f32(name, HIK, hi), w_u8(name, labels))
    comps === nothing || w_comps(name, comps)
end

# ── calibration / delivered-map thorax ──
Dm = cache("sim")
Dm === nothing || variant("calib", Dm.map40, Dm.map70, Dm.map_m2, Dm.mcomps)

# ── sector validation thorax (held-out shape) ──
DS = cache("sect")
DS === nothing || variant("sector", DS.smap40, DS.smap70, DS.smap_m2, DS.scomps)

# README with ImageJ import recipe
open(joinpath(OUT, "README_imagej.txt"), "w") do io
    print(io, """
ImageJ → File → Import → Raw…
  Image type       = 32-bit Real   (*_float32.raw)   |   8-bit  (*_labels_*_uint8.raw)
  Width            = nx   (first number in _<nx>x<ny>x<nz>_)
  Height           = ny
  Number of images = nz
  ☑ Little-endian byte order

Column-major byte layout, written directly (NOT dim2-reversed): the notebook figures use
yreversed=true, so ImageJ row 0 = top matches the figures (spine down). If a scan looks
upside-down, the source figure convention changed — flip vertically in ImageJ.

Variants:  calib = calibration/delivered-map thorax · sector = held-out sector validation
           intfat_r<R>mm = integrated-HU size series (single centred fat insert, radius R mm)
VMI keV pair: $(LOK) / $(HIK).  Labels are 8-bit insert IDs; <variant>_label_comps.csv maps
each label → (f_water, f_lipid, f_protein) ground truth.
""")
end

# round-trip check: byte layout must reconstruct the source array exactly
if Dm !== nothing
    p = joinpath(OUT, "calib_vmi$(LOK)keV_512x512x1_float32.raw")
    back = reshape(reinterpret(Float32, read(p)), 512, 512)
    @assert back == Array{Float32}(Dm.map40) "raw round-trip mismatch — byte layout wrong"
    println("round-trip OK: $p")
end

println("wrote $(length(emitted)) raster(s) + CSVs + README to $OUT")
