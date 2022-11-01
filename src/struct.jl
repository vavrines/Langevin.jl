abstract type AbstractUQ end

"""
$(TYPEDEF)

Struct of one-dimensional UQ setup

# Fields

$(FIELDS)
"""
struct UQ1D{
    A<:Integer,
    B<:AbstractString,
    D<:AbstractOrthoPoly,
    E<:Union{AV,Tuple},
    F<:AM,
    G<:AV,
    H<:AM,
    I<:AA,
    J<:AV,
    C<:AV,
} <: AbstractUQ
    nr::A
    nm::A
    nq::A
    method::B
    optype::B
    op::D
    p::E
    phiRan::F
    t1Product::G
    t2Product::H
    t3Product::I
    pce::J
    pceSample::J
    points::C
    weights::C
end

function UQ1D(
    NR::Int,
    NREC::Int,
    P1::Number,
    P2::Number,
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
        # z ∈ [0,1]
        # op = Uniform01OrthoPoly(nr, Nrec=nRec, addQuadrature=true)

        # z ∈ [-1, 1]
        op = Uniform_11OrthoPoly(nr, Nrec = nRec, addQuadrature = true)
        #supp = (-1.0, 1.0)
        #uni_meas = Measure("uni_meas", x -> 0.5, supp, true, Dict())
        #op = OrthoPoly("uni_op", nr, uni_meas; Nrec = nRec)
    elseif TYPE == "custom"
        supp = (P1, P2)
        uni_meas = Measure("uni_meas", x -> 1 / (P2 - P1), supp, true, Dict())
        op = OrthoPoly("uni_op", nr, uni_meas; Nrec = nRec)
    else
        throw("polynomial chaos not available")
    end

    nq = op.quad.Nquad
    points = op.quad.nodes
    weights = op.quad.weights

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

    return UQ1D{
        typeof(nr),
        typeof(method),
        typeof(op),
        typeof(p),
        typeof(phiRan),
        typeof(t1Product),
        typeof(t2Product),
        typeof(t3Product),
        typeof(pce),
        typeof(points),
    }(
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
        points,
        weights,
    )
end

UQ1D(; nr, nrec, uqp, optype, uqmethod) = UQ1D(nr, nrec, uqp[1], uqp[2], optype, uqmethod)


"""
$(TYPEDEF)

Struct of two-dimensional UQ setup

# Fields

$(FIELDS)

"""
struct UQ2D{
    A<:Integer,
    B<:AbstractString,
    C<:Union{AV,Tuple},
    D<:AV,
    E<:AA{<:AbstractFloat,2},
    F<:AV,
    G<:AM,
    H<:AA,
    I<:AM,
    J<:AV,
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
    P::AV,
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

UQ2D(; nr, nrec, uqp, optype, uqmethod) = UQ2D(nr, nrec, uqp, optype, uqmethod)
