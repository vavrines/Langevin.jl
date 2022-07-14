# ============================================================
# The main functionality has been moved to KitBase.jl.
# ============================================================

"""
$(SIGNATURES)

Calculate L₁ norm of the polynomial basis for the Lasso filter
"""
function basis_norm(uq::AbstractUQ)
    nMoments = uq.nm + 1
    return [
        sum(@. abs(uq.op.quad.weights * uq.phiRan[:, j])) /
        (uq.t2Product[j-1, j-1] + 1.e-8) for j = 1:nMoments
    ]
end


"""
$(SIGNATURES)

Calculate adaptive strength for L₂ filter

*Xiao, Tianbai, and Martin Frank. "A stochastic kinetic scheme for multi-scale flow transport with uncertainty quantification." Journal of Computational Physics 437 (2021): 110337.*
"""
function adapt_filter_strength(u, λ, Δ, op::AbstractOrthoPoly)
    uRan = evaluatePCE(u, op.quad.nodes, op)
    δ = 0.5 * (maximum(uRan) - minimum(uRan))

    return λ * (exp(δ / Δ) - 1.0)
end
