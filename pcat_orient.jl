# Single place where the reconstructed volume is put into clinical orientation.
#
# The simulator inherits XCAT's z convention, which increases toward the HEAD (verified with the
# liver: 26,091 voxels at recon z=1 falling to 0 by z=17, matching the phantom slab's own
# caudal->cranial profile). A clinical series is stored the other way round — slice 1 is the most
# cranial and the index increases toward the feet — so the volume is reversed along z ONCE, here,
# and everything downstream just reads ascending z as cranial->caudal.
#
# The label map MUST be reversed with the same call, in the same place, or the ground truth
# silently decouples from the image it is scored against.
#
# In-plane orientation is separate and unchanged: heatmap(M) draws M[x,y] with the first index
# horizontal and the second vertical, and low j is anterior (sternum j=29.5, vertebra j=454.2),
# so `disp` reverses the second index to put anterior at the top. Do NOT permutedims for that —
# it swaps the axes and rotates the slice 90 degrees.

"""Reverse a 3-D volume along z, turning the simulator's head-increasing index into the clinical
head-first order (slice 1 = most cranial)."""
to_clinical_z(A::AbstractArray{T,3}) where {T} = reverse(A; dims = 3)

"""Map a clinical slice index back to the simulator's own recon z, for cross-referencing the
un-flipped caches and any earlier output."""
sim_z(clin_z::Integer, nz::Integer) = nz - clin_z + 1

"""Anterior-up, patient-left on the viewer's right — standard radiological axial."""
disp(A::AbstractMatrix) = reverse(A; dims = 2)
