# ============================================================
# Structures and Methods of Uncertainty Quantification
# ============================================================

export AbstractUQ
export UQ1D
export ran_chaos, chaos_ran
export lambda_tchaos, t_lambdachaos
export filter!

abstract type AbstractUQ end


"""
Struct of UQ setup

"""
struct UQ1D <: AbstractUQ

    method::AbstractString
    nr::Int
    nRec::Int
    opType::String
    op::AbstractOrthoPoly
    p1::Real
    p2::Real # parameters for random distribution
    phiRan::AbstractArray{<:AbstractFloat,2}
    t1::Tensor
    t2::Tensor
    t3::Tensor
    t1Product::AbstractArray # OffsetArray{Float64,1,Array{Float64,1}}
    t2Product::AbstractArray # OffsetArray{Float64,2,Array{Float64,2}}
    t3Product::AbstractArray # OffsetArray{Float64,3,Array{Float64,3}}
    pce::AbstractArray{<:AbstractFloat,1}
    pceSample::AbstractArray{<:AbstractFloat,1}

    function UQ1D(
        NR::Int,
        NREC::Int,
        P1::AbstractFloat,
        P2::AbstractFloat,
        TYPE = "uniform"::AbstractString,
        METHOD = "collocation"::AbstractString,
    )

        method = METHOD
        nr = NR
        nRec = NREC
        opType = TYPE

        if TYPE == "gauss"
            op = GaussOrthoPoly(nr, Nrec = nRec, addQuadrature = true)
        elseif TYPE == "uniform"
            # uniform ∈ [0,1]
            # op = Uniform01OrthoPoly(nr, Nrec=nRec, addQuadrature=true)

            # uniform ∈ [-1, 1]
            supp = (-1.0, 1.0)
            uni_meas = Measure("uni_meas", x -> 0.5, supp, true, Dict())
            op = OrthoPoly("uni_op", nr, uni_meas; Nrec = nRec)
        end

        p1 = P1
        p2 = P2

        phiRan = evaluate(collect(0:op.deg), op.quad.nodes, op)

        t1 = PolyChaos.Tensor(1, op) # < \phi_i >
        t2 = PolyChaos.Tensor(2, op) # < \phi_i \phi_j >
        t3 = PolyChaos.Tensor(3, op) # < \phi_i \phi_j \phi_k >

        t1Product = OffsetArray{Float64}(undef, 0:nr)
        t2Product = OffsetArray{Float64}(undef, 0:nr, 0:nr)
        t3Product = OffsetArray{Float64}(undef, 0:nr, 0:nr, 0:nr)
        for i = 0:nr
            t1Product[i] = t1.get([i])
        end
        for i = 0:nr
            for j = 0:nr
                t2Product[i, j] = t2.get([i, j])
            end
        end
        for i = 0:nr
            for j = 0:nr
                for k = 0:nr
                    t3Product[i, j, k] = t3.get([i, j, k])
                end
            end
        end

        if TYPE == "gauss"
            pce = [convert2affinePCE(p1, p2, op); zeros(nr - 1)]
        elseif TYPE == "uniform"
            # pce = [ convert2affinePCE(p1, p2, op); zeros(nr-1) ] # uniform ∈ [0, 1]
            pce = [[0.5 * (p1 + p2), 0.5 * (p2 - p1)]; zeros(nr - 1)] # uniform ∈ [-1, 1]
        end

        # pceSample = [1.0] # test
        # pceSample= samplePCE(2000, pce, op) # Monte-Carlo
        pceSample = evaluatePCE(pce, op.quad.nodes, op) # collocation

        # inner constructor
        new(
            method,
            nr,
            nRec,
            opType,
            op,
            p1,
            p2,
            phiRan,
            t1,
            t2,
            t3,
            t1Product,
            t2Product,
            t3Product,
            pce,
            pceSample,
        )

    end

end # struct


"""
Calculate collocation -> polynomial chaos

"""
function ran_chaos(ran::AbstractArray{<:AbstractFloat,1}, uq::AbstractUQ)

    chaos = zeros(eltype(ran), uq.nr + 1)
    for j = 1:uq.nr+1
        chaos[j] =
            sum(@. uq.op.quad.weights * ran * uq.phiRan[:, j]) /
            (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return chaos

end

function ran_chaos(ran::AbstractArray{<:AbstractFloat,1}, op::AbstractOrthoPoly)

    phiRan = evaluate(Vector(0:op.deg), op.quad.nodes, op)
    t2 = Tensor(2, op)

    chaos = zeros(eltype(ran), op.deg + 1)
    for j = 1:op.deg+1
        chaos[j] =
            sum(@. op.quad.weights * ran * phiRan[:, j]) /
            (t2.get([j - 1, j - 1]) + 1.e-7)
    end

    return chaos

end

function ran_chaos(uRan::AbstractArray{Float64,2}, idx::Int64, uq::AbstractUQ)

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

function ran_chaos(uRan::AbstractArray{Float64,3}, idx::Int64, uq::AbstractUQ)

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


"""
Calculate polynomial chaos -> collocation

"""
chaos_ran(chaos::AbstractArray{<:AbstractFloat,1}, uq::AbstractUQ) =
    evaluatePCE(chaos, uq.op.quad.nodes, uq.op)

chaos_ran(chaos::AbstractArray{<:AbstractFloat,1}, op::AbstractOrthoPoly) =
    evaluatePCE(chaos, op.quad.nodes, op)

function chaos_ran(uChaos::AbstractArray{Float64,2}, idx::Int64, uq::AbstractUQ)

    if idx == 1
        uRan = zeros(uq.op.quad.Nquad, axes(uChaos, 2))

        for j in axes(uRan, 2)
            uRan[:, j] .= evaluatePCE(uChaos[:, j], uq.op.quad.nodes, uq.op)
        end
    elseif idx == 2
        uRan = zeros(axes(uChaos, 1), uq.op.quad.Nquad)

        for i in axes(uRan, 1)
            uRan[i, :] .= evaluatePCE(uChaos[i, :], uq.op.quad.nodes, uq.op)
        end
    end

    return uRan

end

function chaos_ran(uChaos::AbstractArray{Float64,3}, idx::Int64, uq::AbstractUQ)

    if idx == 1
        uRan = zeros(uq.op.quad.Nquad, axes(uChaos, 2), axes(uChaos, 3))

        for k in axes(uRan, 3)
            for j in axes(uRan, 2)
                uRan[:, j, k] .=
                    evaluatePCE(uChaos[:, j, k], uq.op.quad.nodes, uq.op)
            end
        end
    elseif idx == 2
        uRan = zeros(axes(uChaos, 1), uq.op.quad.Nquad, axes(uChaos, 3))

        for k in axes(uRan, 3)
            for i in axes(uRan, 1)
                uRan[i, :, k] .=
                    evaluatePCE(uChaos[i, :, k], uq.op.quad.nodes, uq.op)
            end
        end
    elseif idx == 3
        uRan = zeros(axes(uChaos, 1), axes(uChaos, 2), uq.op.quad.Nquad)

        for j in axes(uRan, 2)
            for i in axes(uRan, 1)
                uRan[i, j, :] .=
                    evaluatePCE(uChaos[i, j, :], uq.op.quad.nodes, uq.op)
            end
        end
    end

    return uRan

end


"""
Calculate λ -> T in polynomial chaos

"""
function lambda_tchaos(
    lambdaChaos::Array{<:AbstractFloat,1},
    mass::Real,
    uq::AbstractUQ,
)

    lambdaRan = evaluatePCE(lambdaChaos, uq.op.quad.nodes, uq.op)
    TRan = mass ./ lambdaRan

    TChaos = zeros(typeof(lambdaChaos[1]), uq.nr + 1)
    for j = 1:uq.nr+1
        TChaos[j] =
            sum(@. uq.op.quad.weights * TRan * uq.phiRan[:, j]) /
            (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return TChaos

end


"""
Calculate T -> λ in polynomial chaos

"""
function t_lambdachaos(
    TChaos::Array{<:AbstractFloat,1},
    mass::Real,
    uq::AbstractUQ,
)

    TRan = evaluatePCE(TChaos, uq.op.quad.nodes, uq.op)
    lambdaRan = mass ./ TRan

    lambdaChaos = zeros(typeof(TChaos[1]), uq.nr + 1)
    for j = 1:uq.nr+1
        lambdaChaos[j] =
            sum(@. uq.op.quad.weights * lambdaRan * uq.phiRan[:, j]) /
            (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return lambdaChaos

end


"""
Filter function for polynomial chaos

"""
function filter!(u::AbstractArray{<:AbstractFloat,1}, λ::AbstractFloat)

    q0 = eachindex(u) |> first
    q1 = eachindex(u) |> last

    for i = q0+1:q1
        u[i] /= (1.0 + λ * i^2 * (i - 1)^2)
    end

end

function filter!(
    u::AbstractArray{<:AbstractFloat,2},
    λ::AbstractFloat,
    dim::Int,
)

    if dim == 1
        for j in axes(u, 2)
            filter!(u[:, j], λ)
        end
    elseif dim == 2
        for i in axes(u, 1)
            filter!(u[i, :], λ)
        end
    end

end

function filter!(
    u::AbstractArray{<:AbstractFloat,3},
    λ::AbstractFloat,
    dim::Int,
)

    if dim == 1
        for k in axes(u, 3), j in axes(u, 2)
            filter!(u[:, j, k], λ)
        end
    elseif dim == 2
        for k in axes(u, 3), i in axes(u, 1)
            filter!(u[i, :, k], λ)
        end
    elseif dim == 3
        for j in axes(u, 2), i in axes(u, 1)
            filter!(u[i, j, :], λ)
        end
    end

end
