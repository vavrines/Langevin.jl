# ============================================================
# Uncertainty Quantification Methods
# ============================================================

abstract type AbstractUQ end

"""
    struct UQ1D{
        A<:Integer,
        B<:AbstractString,
        C<:AbstractString,
        D<:AbstractOrthoPoly,
        E<:Union{AbstractVector,Tuple},
        F<:AbstractVector,
        G<:AbstractVector,
        H<:AbstractMatrix,
        I<:AbstractArray,
        J<:AbstractVector,
    } <: AbstractUQ
        nr::A
        nm::A
        nq::A
        method::B
        optype::C
        op::D
        p::E
        phiRan::F
        t1Product::G
        t2Product::H
        t3Product::I
        pce::J
        pceSample::J
    end

Struct of UQ setup

"""
struct UQ1D{
    A<:Integer,
    B<:AbstractString,
    C<:AbstractString,
    D<:AbstractOrthoPoly,
    E<:Union{AbstractVector,Tuple},
    F<:AbstractMatrix,
    G<:AbstractVector,
    H<:AbstractMatrix,
    I<:AbstractArray,
    J<:AbstractVector,
} <: AbstractUQ
    nr::A
    nm::A
    nq::A
    method::B
    optype::C
    op::D
    p::E
    phiRan::F
    t1Product::G
    t2Product::H
    t3Product::I
    pce::J
    pceSample::J
end

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
    optype = TYPE
    nm = nr # 1D

    if TYPE == "gauss"
        op = GaussOrthoPoly(nr, Nrec = nRec, addQuadrature = true)
    elseif TYPE == "uniform"
        # uniform ∈ [0,1]
        # op = Uniform01OrthoPoly(nr, Nrec=nRec, addQuadrature=true)

        # uniform ∈ [-1, 1]
        op = Uniform_11OrthoPoly(nr, Nrec=nRec, addQuadrature=true)
        #supp = (-1.0, 1.0)
        #uni_meas = Measure("uni_meas", x -> 0.5, supp, true, Dict())
        #op = OrthoPoly("uni_op", nr, uni_meas; Nrec = nRec)
    elseif TYPE == "custom"
        supp = (P1, P2)
        uni_meas = Measure("uni_meas", x -> 1 / (P2 - P1), supp, true, Dict())
        op = OrthoPoly("uni_op", nr, uni_meas; Nrec = nRec)
    else
        @warn "polynomial chaos unavailable"
    end

    nq = op.quad.Nquad

    p1 = P1
    p2 = P2
    p = [p1, p2]

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
    else
        pce = [[0.5 * (p1 + p2), 0.5 * (p2 - p1)]; zeros(nr - 1)] # uniform ∈ [-1, 1]
    end

    # pceSample = [1.0] # test
    # pceSample= samplePCE(2000, pce, op) # Monte-Carlo
    pceSample = evaluatePCE(pce, op.quad.nodes, op) # collocation

    return UQ1D(
        nr,
        nm,
        nq,
        method,
        optype,
        op,
        p,
        phiRan,
        t1Product,
        t2Product,
        t3Product,
        pce,
        pceSample,
    )
end


"""
    struct UQ2D{
        A<:Integer,
        B<:AbstractString,
        C<:Union{AbstractVector,Tuple},
        D<:AbstractVector,
        E<:AbstractArray{<:AbstractFloat,2},
        F<:AbstractVector,
        G<:AbstractMatrix,
        H<:AbstractArray,
        I<:AbstractMatrix,
        J<:AbstractVector,
    } <: AbstractUQ
        nr::A
        nm::A
        nq::A
        method::B
        optype::C
        op::MultiOrthoPoly
        p::D
        phiRan::E
        t1Product::F
        t2Product::G
        t3Product::H
        points::I
        weights::J
    end

Struct of UQ setup

"""
struct UQ2D{
    A<:Integer,
    B<:AbstractString,
    C<:Union{AbstractVector,Tuple},
    D<:AbstractVector,
    E<:AbstractArray{<:AbstractFloat,2},
    F<:AbstractVector,
    G<:AbstractMatrix,
    H<:AbstractArray,
    I<:AbstractMatrix,
    J<:AbstractVector,
} <: AbstractUQ
    nr::A
    nm::A
    nq::A
    method::B
    optype::C
    op::MultiOrthoPoly
    p::D
    phiRan::E
    t1Product::F
    t2Product::G
    t3Product::H
    points::I
    weights::J
end

function UQ2D(
    NR::Integer,
    NREC::Integer,
    P::AbstractVector,
    TYPE = ["uniform", "uniform"],
    METHOD = "collocation",
)

    ops = map(TYPE) do x
        if x == "uniform"
            return Uniform_11OrthoPoly(NR, Nrec = NREC, addQuadrature = true)
        elseif x == "gauss"
            return GaussOrthoPoly(NR, Nrec = NREC, addQuadrature = true)
        else
            throw("No default polynomials available")
        end
    end

    phi = MultiOrthoPoly(ops, 4)
    nm = size(phi.ind, 1) - 1

    t1 = PolyChaos.Tensor(1, phi)
    t2 = PolyChaos.Tensor(2, phi)
    t3 = PolyChaos.Tensor(3, phi)

    t1Product = OffsetArray{Float64}(undef, 0:nm)
    t2Product = OffsetArray{Float64}(undef, 0:nm, 0:nm)
    t3Product = OffsetArray{Float64}(undef, 0:nm, 0:nm, 0:nm)
    for i = 0:nm
        t1Product[i] = t1.get([i])
    end
    for i = 0:nm
        for j = 0:nm
            t2Product[i, j] = t2.get([i, j])
        end
    end
    for i = 0:nm
        for j = 0:nm
            for k = 0:nm
                t3Product[i, j, k] = t3.get([i, j, k])
            end
        end
    end

    p = [[P[1], P[2]], [P[3], P[4]]]

    nq = ops[1].quad.Nquad * ops[2].quad.Nquad
    weights = zeros(nq)
    points = zeros(nq, 2)
    for i = 1:ops[1].quad.Nquad, j = 1:ops[2].quad.Nquad
        idx = ops[1].quad.Nquad * (j - 1) + i

        points[idx, 1] = ops[1].quad.nodes[i]
        points[idx, 2] = ops[2].quad.nodes[j]
        weights[idx] = ops[1].quad.weights[i] * ops[2].quad.weights[j]
    end

    phiRan = evaluate(phi.ind, points, phi) |> permutedims

    return UQ2D(
        NR,
        nm,
        nq,
        METHOD,
        TYPE,
        phi,
        p,
        phiRan,
        t1Product,
        t2Product,
        t3Product,
        points,
        weights,
    )

end


"""
Calculate collocation -> polynomial chaos

"""
function ran_chaos(ran::AbstractArray{<:AbstractFloat,1}, uq::UQ1D)
    chaos = zeros(eltype(ran), uq.nr + 1)
    for j = 1:uq.nr+1
        chaos[j] =
            sum(@. uq.op.quad.weights * ran * uq.phiRan[:, j]) /
            (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return chaos
end

function ran_chaos(ran::AbstractArray{<:AbstractFloat,1}, uq::UQ2D)
    chaos = zeros(eltype(ran), uq.nm + 1)
    for j in eachindex(chaos)
        chaos[j] =
            sum(@. uq.weights * ran * uq.phiRan[:, j]) /
            (uq.t2Product[j-1, j-1] + 1.e-7)
    end

    return chaos
end

function ran_chaos(ran::AbstractArray{<:AbstractFloat,1}, op::AbstractOrthoPoly)

    phiRan = evaluate(collect(0:op.deg), op.quad.nodes, op)
    t2 = Tensor(2, op)

    chaos = zeros(eltype(ran), op.deg + 1)
    for j = 1:op.deg+1
        chaos[j] =
            sum(@. op.quad.weights * ran * phiRan[:, j]) / (t2.get([j - 1, j - 1]) + 1.e-7)
    end

    return chaos

end

function ran_chaos(uRan::AbstractArray{<:AbstractFloat,2}, idx::Integer, uq::AbstractUQ)

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

function ran_chaos(uRan::AbstractArray{<:AbstractFloat,3}, idx::Integer, uq::AbstractUQ)

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

function ran_chaos(uRan::AbstractArray{<:AbstractFloat,4}, idx::Integer, uq::AbstractUQ)

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
chaos_ran(chaos::AbstractArray{<:AbstractFloat,1}, uq::UQ1D) =
    evaluatePCE(chaos, uq.op.quad.nodes, uq.op)

chaos_ran(chaos::AbstractArray{<:AbstractFloat,1}, uq::UQ2D) =
    evaluatePCE(chaos, uq.points, uq.op)

chaos_ran(chaos::AbstractArray{<:AbstractFloat,1}, op::AbstractOrthoPoly) =
    evaluatePCE(chaos, op.quad.nodes, op)

function chaos_ran(uChaos::AbstractArray{Float64,2}, idx::Integer, uq::AbstractUQ)

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

function chaos_ran(uChaos::AbstractArray{Float64,3}, idx::Integer, uq::AbstractUQ)

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
