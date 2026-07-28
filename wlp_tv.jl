# The second half of the delivered WLP reader: noise-weighted coupled Huber-TV on (f_l, f_p).
#
# From wlp_model_70_150.toml [tv]:
#   form   = "coupled Huber-TV on (f_l,f_p); den = w + Σ_nbr λ/max(‖∇f‖,eps), w = 1/σ_f²"
#   iters  = 25, eps = 0.04, simplex = "once"
#
# poly2 alone is a per-voxel map with no spatial term, so it passes CT noise straight through and
# amplifies it — the decoded map is speckle. The TV stage is what makes it a reader.
#
# Two things the model card is explicit about and this honours:
#   * lambda is NOT a fixed number: the card's [tv.lambda_model] evaluates λ per voxel at the raw
#     decode (`wlp_lambda_map`), and even its coefficients are chain-specific — refit on a phantom
#     for a new scanner/dose. Never hardcode a λ.
#   * the simplex projection runs ONCE on the result, never per sweep — per-sweep rectification
#     is a Jensen bias on any region mean drawn from the map.

using Statistics: mean

"""Per-voxel sigma on (f_w, f_l, f_p), propagated from the HU noise model through the poly2
Jacobian (evaluated numerically at each voxel's own HU pair)."""
function wlp_sigma_f(hlo, hhi, model, decode)
    sq_lo = Float64.(model["noise"]["sigma_quad_low"])
    sq_hi = Float64.(model["noise"]["sigma_quad_high"])
    ρ = Float64(model["noise"]["rho"])
    σlo = abs(sq_lo[1] * hlo^2 + sq_lo[2] * hlo + sq_lo[3])
    σhi = abs(sq_hi[1] * hhi^2 + sq_hi[2] * hhi + sq_hi[3])
    δ = 0.5
    f0 = decode(hlo, hhi)
    dl = (decode(hlo + δ, hhi) .- decode(hlo - δ, hhi)) ./ (2δ)   # ∂f/∂h_lo
    dh = (decode(hlo, hhi + δ) .- decode(hlo, hhi - δ)) ./ (2δ)   # ∂f/∂h_hi
    # var(f) = J Σ Jᵀ, with Σ = [[σlo², ρσloσhi], [ρσloσhi, σhi²]]
    v = ntuple(k -> dl[k]^2 * σlo^2 + dh[k]^2 * σhi^2 + 2ρ * σlo * σhi * dl[k] * dh[k], 3)
    (f0, map(x -> sqrt(max(x, 1e-12)), v))
end

"""λ(f̂) map from the model card's `[tv.lambda_model]`: log₁₀λ affine in (f_l, f_p), evaluated at
the raw decode clamped onto the simplex, log₁₀ clamped to the fitted range. Falls back to the
scalar `[tv.lambda]` (the pre-model behaviour) when the card predates the model."""
function wlp_lambda_map(model, fl0, fp0, valid)
    tv = model["tv"]
    haskey(tv, "lambda_model") || return Float64(tv["lambda"])
    c = Float64.(tv["lambda_model"]["coeff"])
    lo, hi = Float64.(tv["lambda_model"]["log10_clamp"])
    lam = zeros(Float64, size(fl0))
    for i in eachindex(fl0)
        valid[i] || continue
        l = clamp(fl0[i], 0.0, 1.0); p = clamp(fp0[i], 0.0, 1.0 - l)
        lam[i] = 10.0^clamp(c[1] + c[2] * l + c[3] * p, lo, hi)
    end
    lam
end

"""Noise-weighted coupled Huber-TV on (f_l, f_p), 4-neighbour, in-plane per slice.

`fl0`/`fp0` are the poly2 estimates, `wl`/`wp` the per-voxel weights 1/σ_f². `valid` marks
decodable voxels. `λ` is a scalar or a per-voxel matrix (`wlp_lambda_map`). Returns the
smoothed (f_l, f_p); f_w follows from the simplex closure.
"""
function wlp_tv!(fl0, fp0, wl, wp, valid; λ, iters = 25, eps = 0.04)
    fl = copy(fl0); fp = copy(fp0)
    n, m = size(fl)
    nl = similar(fl); np = similar(fp)
    for _ in 1:iters
        copyto!(nl, fl); copyto!(np, fp)
        @inbounds for j in 2:m-1, i in 2:n-1
            valid[i, j] || continue
            lam = λ isa AbstractArray ? Float64(λ[i, j]) : Float64(λ)
            numl = wl[i, j] * fl0[i, j]; nump = wp[i, j] * fp0[i, j]
            denl = wl[i, j];             denp = wp[i, j]
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ii, jj = i + di, j + dj
                valid[ii, jj] || continue
                # coupled gradient magnitude over BOTH fractions — this is what makes the
                # edge decision shared, so f_l and f_p keep a consistent boundary
                g = sqrt((fl[i, j] - fl[ii, jj])^2 + (fp[i, j] - fp[ii, jj])^2)
                c = lam / max(g, eps)
                numl += c * fl[ii, jj]; denl += c
                nump += c * fp[ii, jj]; denp += c
            end
            nl[i, j] = numl / denl
            np[i, j] = nump / denp
        end
        copyto!(fl, nl); copyto!(fp, np)
    end
    (fl, fp)
end

"""Simplex projection, applied ONCE (never inside the sweep) — the notebook's `smp`, exactly:
clamp (f_l, f_p) to ≥0, renormalize only if f_l+f_p > 1, f_w by closure. Every fraction lands
in [0,1] and they sum to 1. (The old signature took f_w too, then renormalized all three — a
bad f_w silently rescaled the reported f_l/f_p, and it disagreed with the notebook's numbers.)"""
function wlp_simplex(fl, fp)
    l = max(fl, 0.0); p = max(fp, 0.0)
    s = l + p
    s > 1.0 && (l /= s; p /= s)
    (1.0 - l - p, l, p)
end
