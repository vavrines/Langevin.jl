# ============================================================
# Uncertainty Quantification Methods
# ============================================================

"""
Calculate collocation -> polynomial chaos

"""
function ran_chaos(ran::AA{<:AbstractFloat,1}, uq::UQ1D)
    chaos = zeros(eltype(ran), uq.nr + 1)
    for j = 1:uq.nr+1
        chaos[j] =
            sum(@. uq.op.quad.weights * ran * uq.phiRan[:, j]) /
            (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return chaos
end

function ran_chaos(ran::AA{<:AbstractFloat,1}, uq::UQ2D)
    chaos = zeros(eltype(ran), uq.nm + 1)
    for j in eachindex(chaos)
        chaos[j] =
            sum(@. uq.weights * ran * uq.phiRan[:, j]) / (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return chaos
end

function ran_chaos(ran::AA{<:AbstractFloat,1}, op::AbstractOrthoPoly)

    phiRan = evaluate(collect(0:op.deg), op.quad.nodes, op)
    t2 = Tensor(2, op)

    chaos = zeros(eltype(ran), op.deg + 1)
    for j = 1:op.deg+1
        chaos[j] =
            sum(@. op.quad.weights * ran * phiRan[:, j]) / (t2.get([j - 1, j - 1]) + 1.e-7)
    end

    return chaos

end

function ran_chaos(uRan::AA{<:AbstractFloat,2}, idx::Integer, uq::AbstractUQ)

    if idx == 1

        uChaos = zeros(uq.nr + 1, axes(uRan, 2))
        for j in axes(uChaos, 2)
            uChaos[:, j] .= ran_chaos(uRan[:, j], uq)
        end

    elseif idx == 2

        uChaos = zeros(axes(uRan, 1), uq.nr + 1)
        for i in axes(uChaos, 1)
            uChaos[i, :] .= ran_chaos(uRan[i, :], uq)
        end

    end

    return uChaos

end

function ran_chaos(uRan::AA{<:AbstractFloat,3}, idx::Integer, uq::AbstractUQ)

    if idx == 1

        uChaos = zeros(uq.nr + 1, axes(uRan, 2), axes(uRan, 3))
        for k in axes(uChaos, 3)
            for j in axes(uChaos, 2)
                uChaos[:, j, k] .= ran_chaos(uRan[:, j, k], uq)
            end
        end

    elseif idx == 2

        uChaos = zeros(axes(uRan, 1), uq.nr + 1, axes(uRan, 3))
        for k in axes(uChaos, 3)
            for i in axes(uChaos, 1)
                uChaos[i, :, k] .= ran_chaos(uRan[i, :, k], uq)
            end
        end

    elseif idx == 3

        uChaos = zeros(uq.nr + 1, axes(uRan, 2), axes(uRan, 3))
        for k in axes(uChaos, 3)
            for j in axes(uChaos, 2)
                uChaos[:, j, k] .= ran_chaos(uRan[:, j, k], uq)
            end
        end

    end

    return uChaos

end

function ran_chaos(uRan::AA{<:AbstractFloat,4}, idx::Integer, uq::AbstractUQ)

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
Calculate polynomial chaos -> collocation

"""
chaos_ran(chaos::AA{<:AbstractFloat,1}, uq::UQ1D) =
    evaluatePCE(chaos, uq.op.quad.nodes, uq.op)

chaos_ran(chaos::AA{<:AbstractFloat,1}, uq::UQ2D) =
    evaluatePCE(chaos, uq.points, uq.op)

chaos_ran(chaos::AA{<:AbstractFloat,1}, op::AbstractOrthoPoly) =
    evaluatePCE(chaos, op.quad.nodes, op)

function chaos_ran(uChaos::AA{Float64,2}, idx::Integer, uq::AbstractUQ)

    if idx == 1

        uRan = zeros(uq.op.quad.Nquad, axes(uChaos, 2))
        for j in axes(uRan, 2)
            uRan[:, j] .= chaos_ran(uChaos[:, j], uq)
        end

    elseif idx == 2

        uRan = zeros(axes(uChaos, 1), uq.op.quad.Nquad)
        for i in axes(uRan, 1)
            uRan[i, :] .= chaos_ran(uChaos[i, :], uq)
        end

    end

    return uRan

end

function chaos_ran(uChaos::AA{Float64,3}, idx::Integer, uq::AbstractUQ)

    if idx == 1

        uRan = zeros(uq.op.quad.Nquad, axes(uChaos, 2), axes(uChaos, 3))
        for k in axes(uRan, 3)
            for j in axes(uRan, 2)
                uRan[:, j, k] .= chaos_ran(uChaos[:, j, k], uq)
            end
        end

    elseif idx == 2

        uRan = zeros(axes(uChaos, 1), uq.op.quad.Nquad, axes(uChaos, 3))
        for k in axes(uRan, 3)
            for i in axes(uRan, 1)
                uRan[i, :, k] .= chaos_ran(uChaos[i, :, k], uq)
            end
        end

    elseif idx == 3

        uRan = zeros(axes(uChaos, 1), axes(uChaos, 2), uq.op.quad.Nquad)
        for j in axes(uRan, 2)
            for i in axes(uRan, 1)
                uRan[i, j, :] .= chaos_ran(uChaos[i, j, :], uq)
            end
        end

    end

    return uRan

end


"""
Calculate λ -> T in polynomial chaos

"""
function lambda_tchaos(lambdaChaos::Array{<:AbstractFloat,1}, mass::Real, uq::AbstractUQ)
    lambdaRan = chaos_ran(lambdaChaos, uq)
    TRan = mass ./ lambdaRan
    TChaos = ran_chaos(TRan, uq)

    return TChaos
end


"""
Calculate T -> λ in polynomial chaos

"""
function t_lambdachaos(TChaos::Array{<:AbstractFloat,1}, mass::Real, uq::AbstractUQ)
    TRan = chaos_ran(TChaos, uq)
    lambdaRan = mass ./ TRan
    lambdaChaos = ran_chaos(lambdaRan, uq)

    return lambdaChaos

end
