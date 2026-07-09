ImageJ → File → Import → Raw…
  Image type       = 32-bit Real   (*_float32.raw)   |   8-bit  (*_labels*_uint8.raw)
  Width            = nx   (first number in _<nx>x<ny>x<nz>_)
  Height           = ny
  Number of images = nz   (= 3 here; each file is the full reconstructed z-stack)
  ☑ Little-endian byte order

Column-major, written directly (NOT dim2-reversed): the notebook figures use yreversed=true,
so ImageJ row 0 = top matches them. Each file holds 3 z-slices as consecutive frames. The
phantom is z-uniform (2D inserts extruded along z) and the recon applies a zmed=1 cross-z
median, so the 3 slices are near-identical denoised planes — scroll z to confirm.

Layout — grouped by role:
  calibration/  noise-measurement sims the σ(HU) ladder is fit on:
                  noise_sim1..4   (+ deliveredmap = the delivered-decomposition-map thorax)
                labels_*.raw  = shared insert stack (labels 8..20)
  test/         held-out validation:
                  circular_sim1..5  ·  sector_sim1..4  (+ sector_deliveredmap)
                labels_circular = circular inserts (8..20)  ·  labels_sector = sector inserts (8..23)

VMI keV pair: 70 / 150.  label_comps.csv (per folder) → each scan's per-label
(f_water, f_lipid, f_protein) ground truth. Circular scans use labels_circular, sector scans
use labels_sector.
