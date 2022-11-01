# ============================================================
# Uncertainty Quantification Methods
# ============================================================

"""
$(SIGNATURES)

Transform collocation solution -> polynomial chaos
"""
function ran_chaos(ran::AV, uq::AbstractUQ)
    chaos = zeros(eltype(ran), uq.nm + 1)
    for j in eachindex(chaos)
        chaos[j] =
            sum(@. uq.weights * ran * uq.phiRan[:, j]) / (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return chaos
end

"""
$(SIGNATURES)
"""
function ran_chaos(ran::AV, op::AbstractOrthoPoly)
    phiRan = evaluate(collect(0:op.deg), op.quad.nodes, op)
    t2 = Tensor(2, op)

    chaos = zeros(eltype(ran), op.deg + 1)
    for j = 1:op.deg+1
        chaos[j] =
            sum(@. op.quad.weights * ran * phiRan[:, j]) / (t2.get([j - 1, j - 1]) + 1.e-7)
    end

    return chaos
end

"""
$(SIGNATURES)
"""
function ran_chaos(uRan::AM, idx::Integer, uq::AbstractUQ)
    if idx == 1
        uChaos = zeros(uq.nm + 1, axes(uRan, 2))
        for j in axes(uChaos, 2)
            uChaos[:, j] .= ran_chaos(uRan[:, j], uq)
        end
    elseif idx == 2
        uChaos = zeros(axes(uRan, 1), uq.nm + 1)
        for i in axes(uChaos, 1)
            uChaos[i, :] .= ran_chaos(uRan[i, :], uq)
        end
    end

    return uChaos
end

function ran_chaos(uRan::AA{T,3}, idx::Integer, uq::AbstractUQ) where {T}
    if idx == 1
        uChaos = zeros(uq.nm + 1, axes(uRan, 2), axes(uRan, 3))
        for k in axes(uChaos, 3)
            for j in axes(uChaos, 2)
                uChaos[:, j, k] .= ran_chaos(uRan[:, j, k], uq)
            end
        end
    elseif idx == 2
        uChaos = zeros(axes(uRan, 1), uq.nm + 1, axes(uRan, 3))
        for k in axes(uChaos, 3)
            for i in axes(uChaos, 1)
                uChaos[i, :, k] .= ran_chaos(uRan[i, :, k], uq)
            end
        end
    elseif idx == 3
        uChaos = zeros(uq.nm + 1, axes(uRan, 2), axes(uRan, 3))
        for k in axes(uChaos, 3)
            for j in axes(uChaos, 2)
                uChaos[:, j, k] .= ran_chaos(uRan[:, j, k], uq)
            end
        end
    end

    return uChaos
end

function ran_chaos(uRan::AA{T,4}, idx::Integer, uq::AbstractUQ) where {T}
    if idx == 1
        uChaos = zeros(uq.nr + 1, axes(uRan, 2), axes(uRan, 3), axes(uRan, 4))
        for l in axes(uChaos, 4), k in axes(uChaos, 3), j in axes(uChaos, 2)
            uChaos[:, j, k, l] .= ran_chaos(uRan[:, j, k, l], uq)
        end
    elseif idx == 2
        uChaos = zeros(axes(uRan, 1), uq.nr + 1, axes(uRan, 3), axes(uRan, 4))
        for l in axes(uChaos, 4), k in axes(uChaos, 3), i in axes(uChaos, 1)
            uChaos[i, :, k, l] .= ran_chaos(uRan[i, :, k, l], uq)
        end
    elseif idx == 3
        uChaos = zeros(axes(uRan, 1), axes(uRan, 2), uq.nr + 1, axes(uRan, 4))
        for l in axes(uChaos, 4), j in axes(uChaos, 2), i in axes(uChaos, 1)
            uChaos[i, j, :, l] .= ran_chaos(uRan[i, j, :, l], uq)
        end
    elseif idx == 4
        uChaos = zeros(axes(uRan, 1), axes(uRan, 2), axes(uRan, 3), uq.nr + 1)
        for k in axes(uChaos, 3), j in axes(uChaos, 2), i in axes(uChaos, 1)
            uChaos[i, j, k, :] .= ran_chaos(uRan[i, j, k, :], uq)
        end
    end

    return uChaos
end


"""
$(SIGNATURES)

Transform polynomial chaos -> collocation solution
"""
chaos_ran(chaos::AV, uq::AbstractUQ) = evaluatePCE(chaos, uq.points, uq.op)

"""
$(SIGNATURES)
"""
chaos_ran(chaos::AV, op::AbstractOrthoPoly) = evaluatePCE(chaos, op.quad.nodes, op)

"""
$(SIGNATURES)
"""
function chaos_ran(uChaos::AM, idx::Integer, uq::AbstractUQ)
    if idx == 1
        uRan = zeros(uq.nq, axes(uChaos, 2))
        for j in axes(uRan, 2)
            uRan[:, j] .= chaos_ran(uChaos[:, j], uq)
        end
    elseif idx == 2
        uRan = zeros(axes(uChaos, 1), uq.nq)
        for i in axes(uRan, 1)
            uRan[i, :] .= chaos_ran(uChaos[i, :], uq)
        end
    end

    return uRan
end

function chaos_ran(uChaos::AA{T,3}, idx::Integer, uq::AbstractUQ) where {T}
    if idx == 1
        uRan = zeros(uq.nq, axes(uChaos, 2), axes(uChaos, 3))
        for k in axes(uRan, 3)
            for j in axes(uRan, 2)
                uRan[:, j, k] .= chaos_ran(uChaos[:, j, k], uq)
            end
        end
    elseif idx == 2
        uRan = zeros(axes(uChaos, 1), uq.nq, axes(uChaos, 3))
        for k in axes(uRan, 3)
            for i in axes(uRan, 1)
                uRan[i, :, k] .= chaos_ran(uChaos[i, :, k], uq)
            end
        end
    elseif idx == 3
        uRan = zeros(axes(uChaos, 1), axes(uChaos, 2), uq.nq)
        for j in axes(uRan, 2)
            for i in axes(uRan, 1)
                uRan[i, j, :] .= chaos_ran(uChaos[i, j, :], uq)
            end
        end
    end

    return uRan
end


"""
$(SIGNATURES)

Transform λ -> T in polynomial chaos
"""
function lambda_tchaos(lambdaChaos::AV, mass, uq::AbstractUQ)
    lambdaRan = chaos_ran(lambdaChaos, uq)
    TRan = mass ./ lambdaRan
    TChaos = ran_chaos(TRan, uq)

    return TChaos
end


"""
$(SIGNATURES)

Transform T -> λ in polynomial chaos
"""
function t_lambdachaos(TChaos::AV, mass, uq::AbstractUQ)
    TRan = chaos_ran(TChaos, uq)
    lambdaRan = mass ./ TRan
    lambdaChaos = ran_chaos(lambdaRan, uq)

    return lambdaChaos
end


"""
$(SIGNATURES)

Calculate product of two polynomial chaos
"""
function chaos_product!(u::AV, a::AV, b::AV, uq::AbstractUQ)
    @assert length(u) == length(a) == length(b)

    L = uq.nm
    for m = 0:L
        u[m+1] = sum(
            a[j+1] * b[k+1] * uq.t3Product[j, k, m] / uq.t2Product[m, m] for j = 0:L for
            k = 0:L
        )
    end

    return nothing
end

"""
$(SIGNATURES)

Calculate product of two polynomial chaos
"""
function chaos_product(a::AV, b::AV, uq::AbstractUQ)
    u = similar(a)
    chaos_product!(u, a, b, uq)

    return u
end
