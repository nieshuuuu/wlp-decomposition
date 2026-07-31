# Export the reconstructed CT in CLINICAL orientation, so a viewer shows the same thing the
# figures do: slice 1 is the most cranial and the index increases toward the feet.
#
# The simulator inherits XCAT's z convention, which increases toward the head. Everything is
# reversed along z here — the two VMI volumes AND the label map together — via the one shared
# `to_clinical_z` so the ground truth cannot decouple from the image.
#
# In-plane data is written unchanged; only the slice ORDER differs from the earlier export.
# Anterior is still low j, which is what ImageJ shows at the top of its display.
import BasisSimulator as BS
using Serialization: deserialize
using Statistics: mean, median
include(joinpath(@__DIR__, "pcat_orient.jl"))

const OUT = joinpath(@__DIR__, "pcat_ct")
const D = deserialize(joinpath(OUT, get(ENV, "PCAT_ACQ", "pcat_acq_shell.jls")))
const RECON_N, RECON_NZ = 512, 40
const VOXMM = 0.5

stub = Dict{Int,Any}(Int(l)=>BS.XA.Materials.water for l in unique(D.slab))
m3 = to_clinical_z(BS.resample_to_recon(BS.Phantom(D.slab, stub, (VOXMM/10,VOXMM/10,VOXMM/10)),
                                        D.geom, (RECON_N,RECON_N,RECON_NZ); method=:nearest))
h70 = to_clinical_z(Float32.(D.hu_lo))
h150 = to_clinical_z(Float32.(D.hu_hi))

# verify the flip actually landed: the liver is CAUDAL, so after flipping it must be at HIGH z
livz = [count(==(UInt8(10)), @view m3[:,:,z]) for z in 1:RECON_NZ]
first_half = sum(livz[1:RECON_NZ÷2]); second_half = sum(livz[RECON_NZ÷2+1:end])
println("liver voxels: first half (cranial) $first_half, second half (caudal) $second_half")
second_half > first_half ||
    error("clinical flip failed: liver should sit in the CAUDAL half after flipping")
println("clinical z confirmed — slice 1 is cranial, slice $RECON_NZ is caudal\n")

for (nm, A) in (("vmi070keV", h70), ("vmi150keV", h150))
    p = joinpath(OUT, "pcat_$(nm)_clinicalZ_$(RECON_N)x$(RECON_N)x$(RECON_NZ)_float32.raw")
    write(p, A); println("wrote $p  ($(round(filesize(p)/1e6, digits=2)) MB)")
end
let p = joinpath(OUT, "pcat_labels_clinicalZ_$(RECON_N)x$(RECON_N)x$(RECON_NZ)_uint8.raw")
    write(p, UInt8.(m3)); println("wrote $p")
end
println("\n512 x 512 x $RECON_NZ, 0.352 mm in-plane, 1 mm slices, slice 1 = most cranial")
