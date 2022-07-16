# ============================================================
# Initialize Simulation
# ============================================================

"""
$(SIGNATURES)

Initialize solver
"""
function initialize(config::AbstractString, structure = :ctr)
    println("==============================================================")
    println("Langevin.jl: Stochastic Kinetic Modeling and Simulation")
    println("==============================================================")
    println("")
    println("initializeing solver:")
    println("")

    allowed = ["uqMethod", "nr", "nRec", "opType", "parameter1", "parameter2"]
    D = read_dict(config, allowed)

    nr = D["nr"]
    nRec = D["nRec"]
    opType = D["opType"]
    parameter1 = D["parameter1"]
    parameter2 = D["parameter2"]
    uqMethod = D["uqMethod"]

    ks = SolverSet(config)
    uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)

    if structure == :ctr
        ctr, face = init_fvm(ks, uq)
        return ks, ctr, face, uq, 0.0
    elseif structure == :sol
        sol, flux = init_sol(ks, uq)
        return ks, sol, flux, uq, 0.0
    else
        throw(ArgumentError("no data structure available for $structure"))
    end
end


"""
$(SIGNATURES)

Initialize solution structure
"""
init_sol(KS, uq::AbstractUQ) = init_sol(KS, KS.ps, uq)

"""
$(SIGNATURES)

1D initialization
"""
function init_sol(KS, ps::AbstractPhysicalSpace1D, uq::AbstractUQ)
    w0 = begin
        if uq.method == "galerkin"
            zeros(axes(KS.ib.fw(KS.ps.x0, KS.ib.p), 1), uq.nr + 1)
        else
            zeros(axes(KS.ib.fw(KS.ps.x0, KS.ib.p), 1), uq.op.quad.Nquad)
        end
    end
    prim0 = zero(w0)
    w = [deepcopy(w0) for i in axes(KS.ps.x, 1)]
    prim = deepcopy(w)
    facefw = [zero(w0) for i = 1:KS.ps.nx+1]

    for i in axes(w, 1)
        if uq.method == "galerkin"
            w0[:, 1] .= KS.ib.fw(KS.ps.x[i], KS.ib.p)
        else
            for j in axes(w0, 2)
                w0[:, j] .= KS.ib.fw(KS.ps.x[i], KS.ib.p)
            end
        end

        w[i] .= w0
        prim[i] .= uq_conserve_prim(w0, KS.gas.γ, uq)
    end

    if KS.set.space[3:4] == "1f"
        f0 = begin
            if KS.set.space[5:6] == "1v"
                uq_maxwellian(KS.vs.u, prim0, uq)
            elseif KS.set.space[5:6] == "3v"
                uq_maxwellian(KS.vs.u, KS.vs.v, KS.vs.w, prim0, uq)
            end
        end

        f = [deepcopy(f0) for i in axes(KS.ps.x, 1)]
        for i in axes(w, 1)
            f[i] .= begin
                if KS.set.space[5:6] == "1v"
                    uq_maxwellian(KS.vs.u, prim[i], uq)
                elseif KS.set.space[5:6] == "3v"
                    uq_maxwellian(KS.vs.u, KS.vs.v, KS.vs.w, prim[i], uq)
                end
            end
        end
        faceff = [zero(f0) for i = 1:KS.ps.nx+1]

        sol = Solution1D(w, prim, f)
        flux = Flux1D(facefw, faceff)

        return sol, flux
    elseif KS.set.space[3:4] == "2f"
        if KS.set.space[5:6] == "1v"
            h0 = uq_maxwellian(KS.vs.u, prim0, uq)
        elseif KS.set.space[5:6] == "2v"
            h0 = uq_maxwellian(KS.vs.u, KS.vs.v, prim0, uq)
        end

        h = [deepcopy(h0) for i in axes(KS.ps.x, 1)]
        b = deepcopy(h)
        if KS.set.space[5:6] == "1v"
            for i in axes(w, 1)
                h[i] .= uq_maxwellian(KS.vs.u, prim[i], uq)
                b[i] .= uq_energy_distribution(h[i], prim[i], KS.gas.K, uq)
            end
        elseif KS.set.space[5:6] == "2v"
            for i in axes(w, 1)
                h[i] .= uq_maxwellian(KS.vs.u, KS.vs.v, prim[i], uq)
                b[i] .= uq_energy_distribution(h[i], prim[i], KS.gas.K, uq)
            end
        end

        facefh = [zero(h0) for i = 1:KS.ps.nx+1]
        facefb = [zero(h0) for i = 1:KS.ps.nx+1]

        sol = Solution1D(w, prim, h, b)
        flux = Flux1D(facefw, facefh, facefb)

        return sol, flux
    end
end

"""
$(SIGNATURES)

2D initialization
"""
function init_sol(KS, ps::AbstractPhysicalSpace2D, uq::AbstractUQ)
    prim0 = begin
        if uq.method == "galerkin"
            zeros(axes(KS.ib.fw(KS.ps.x0, KS.ps.y0, KS.ib.p), 1), uq.nr + 1)
        else
            zeros(axes(KS.ib.fw(KS.ps.x0, KS.ps.y0, KS.ib.p), 1), uq.op.quad.Nquad)
        end
    end

    w = [deepcopy(prim0) for i in axes(KS.ps.x, 1), j in axes(KS.ps.x, 2)]
    prim = [deepcopy(prim0) for i in axes(KS.ps.x, 1), j in axes(KS.ps.x, 2)]
    for j in axes(w, 2), i in axes(w, 1)
        if uq.method == "galerkin"
            w[i, j][:, 1] .= KS.ib.fw(KS.ps.x[i, j], KS.ps.y[i, j], KS.ib.p)
        else
            for k = 1:uq.op.quad.Nquad
                w[i, j][:, k] .= KS.ib.fw(KS.ps.x[i, j], KS.ps.y[i, j], KS.ib.p)
            end
        end
        prim[i, j] .= uq_conserve_prim(w[i, j], KS.gas.γ, uq)
    end

    n1 = [[1.0, 0.0] for i = 1:KS.ps.nx+1, j = 1:KS.ps.ny]
    n2 = [[0.0, 1.0] for i = 1:KS.ps.nx, j = 1:KS.ps.ny+1]
    for j = 1:ps.ny
        for i = 1:ps.nx
            _n = unit_normal(ps.vertices[i, j, 1, :], ps.vertices[i, j, 4, :])
            n1[i, j] .= ifelse(
                dot(_n, [ps.x[i, j], ps.y[i, j]] .- ps.vertices[i, j, 1, :]) >= 0,
                _n,
                -_n,
            )
        end
        _n = unit_normal(ps.vertices[ps.nx, j, 2, :], ps.vertices[ps.nx, j, 3, :])
        n1[KS.ps.nx+1, j] .= ifelse(
            dot(_n, ps.vertices[ps.nx, j, 2, :] .- [ps.x[ps.nx, j], ps.y[ps.nx, j]]) >= 0,
            _n,
            -_n,
        )
    end
    for i = 1:ps.nx
        for j = 1:ps.ny
            _n = unit_normal(ps.vertices[i, j, 1, :], ps.vertices[i, j, 2, :])
            n2[i, j] .= ifelse(
                dot(_n, [ps.x[i, j], ps.y[i, j]] .- ps.vertices[i, j, 1, :]) >= 0,
                _n,
                -_n,
            )
        end
        _n = unit_normal(ps.vertices[i, ps.ny, 3, :], ps.vertices[i, ps.ny, 4, :])
        n2[i, ps.ny+1] .= ifelse(
            dot(_n, ps.vertices[i, ps.ny, 3, :] .- [ps.x[i, ps.ny], ps.y[i, ps.ny]]) >= 0,
            _n,
            -_n,
        )
    end

    facew1 = [zeros(axes(prim0)) for i = 1:KS.ps.nx+1, j = 1:KS.ps.ny]
    facefw1 = [zeros(axes(prim0)) for i = 1:KS.ps.nx+1, j = 1:KS.ps.ny]
    facew2 = [zeros(axes(prim0)) for i = 1:KS.ps.nx, j = 1:KS.ps.ny+1]
    facefw2 = [zeros(axes(prim0)) for i = 1:KS.ps.nx, j = 1:KS.ps.ny+1]

    n = (n1, n2)
    facew = (facew1, facew2)
    facefw = (facefw1, facefw2)

    if KS.set.space[3:4] == "1f"
        f = [
            uq_maxwellian(KS.vs.u, KS.vs.v, prim[i, j], uq) for i in axes(KS.ps.x, 1),
            j in axes(KS.ps.x, 2)
        ]

        f0 = uq_maxwellian(KS.vs.u, KS.vs.v, prim0, uq)
        faceff1 = [zero(f0) for i = 1:KS.ps.nx+1, j = 1:KS.ps.ny]
        faceff2 = [zero(f0) for i = 1:KS.ps.nx, j = 1:KS.ps.ny+1]
        faceff = (faceff1, faceff2)

        sol = Solution2D(w, prim, f)
        flux = Flux2F{typeof(n),typeof(facew),typeof(faceff),2}(n, facew, facefw, faceff)

    elseif KS.set.space[3:4] == "2f"
        h = [
            uq_maxwellian(KS.vs.u, KS.vs.v, prim[i, j], uq, KS.gas.K)[1] for
            i in axes(KS.ps.x, 1), j in axes(KS.ps.x, 2)
        ]
        b = [
            uq_maxwellian(KS.vs.u, KS.vs.v, prim[i, j], uq, KS.gas.K)[2] for
            i in axes(KS.ps.x, 1), j in axes(KS.ps.x, 2)
        ]

        h0, b0 = uq_maxwellian(KS.vs.u, KS.vs.v, prim0, uq, KS.gas.K)
        facefh1 = [zeros(axes(h0)) for i = 1:KS.ps.nx+1, j = 1:KS.ps.ny]
        facefb1 = [zeros(axes(b0)) for i = 1:KS.ps.nx+1, j = 1:KS.ps.ny]
        facefh2 = [zeros(axes(h0)) for i = 1:KS.ps.nx, j = 1:KS.ps.ny+1]
        facefb2 = [zeros(axes(b0)) for i = 1:KS.ps.nx, j = 1:KS.ps.ny+1]
        facefh = (facefh1, facefh2)
        facefb = (facefb1, facefb2)

        sol = Solution2D(w, prim, h, b)
        flux = Flux2F{typeof(n),typeof(facew),typeof(facefh),2}(
            n,
            facew,
            facefw,
            facefh,
            facefb,
        )

    end

    return sol, flux
end


"""
$(SIGNATURES)

Initialize finite volume structs
"""
function init_fvm(KS, uq::AbstractUQ)
    w0 = KS.ib.fw(KS.ps.x[1], KS.ib.p)

    if ndims(w0) == 1
        ctr, face = pure_fvm(KS, uq)
    elseif ndims(w0) == 2
        if KS.set.space[3:4] in ("3f", "4f")
            ctr, face = plasma_fvm(KS, KS.ps, uq)
        else
            ctr, face = mixture_fvm(KS, KS.ps, uq)
        end
    end

    return ctr, face
end


pure_fvm(KS, uq::AbstractUQ) = pure_fvm(KS, KS.ps, uq)

function pure_fvm(KS, pSpace::PSpace1D, uq::AbstractUQ)
    idx0 = (eachindex(pSpace.x)|>collect)[1]
    idx1 = (eachindex(pSpace.x)|>collect)[end]

    if KS.set.space[3:4] == "1f"
        ctr = OffsetArray{ControlVolume1F}(undef, idx0:idx1) # with ghost cells
        face = Array{Interface1F}(undef, KS.ps.nx + 1)
    elseif KS.set.space[3:4] == "2f"
        ctr = OffsetArray{ControlVolume2F}(undef, idx0:idx1)
        face = Array{Interface2F}(undef, KS.ps.nx + 1)
    elseif KS.set.space[3:4] == "4f"
        ctr = OffsetArray{ControlVolume1D4F}(undef, idx0:idx1)
    end

    if uq.method == "galerkin"
        prim = zeros(axes(KS.ib.bc(KS.ps.x0, KS.ib.p), 1), uq.nr + 1)

        for i in eachindex(ctr)
            prim[:, 1] .= KS.ib.bc(KS.ps.x[i], KS.ib.p)
            w = uq_prim_conserve(prim, KS.gas.γ, uq)

            ctr[i] = begin
                if KS.set.space[3:4] == "1f"
                    f = uq_maxwellian(KS.vs.u, prim, uq)
                    ControlVolume(w, prim, f, 1)
                elseif KS.set.space[3:4] == "2f"
                elseif KS.set.space[3:4] == "4f"
                end
            end
        end

        prim[:, 1] .= KS.ib.bc(KS.ps.x0, KS.ib.p)
        w = uq_prim_conserve(prim, KS.gas.γ, uq)
        f = uq_maxwellian(KS.vs.u, prim, uq)
        for i in eachindex(face)
            face[i] = begin
                if KS.set.space[3:4] == "1f"
                    Interface(w, f, 1)
                elseif KS.set.space[3:4] == "2f"
                    b = uq_energy_distribution(f, prim, KS.gas.K, uq)
                    ControlVolume(w, prim, f, b, 1)
                end
            end
        end
    elseif uq.method == "collocation"
        prim = zeros(axes(KS.ib.bc(KS.ps.x0, KS.ib.p), 1), uq.op.quad.Nquad)
        for i in eachindex(ctr)
            for j in axes(prim, 2)
                prim[:, j] .= KS.ib.bc(KS.ps.x[i], KS.ib.p)
            end
            w = uq_prim_conserve(prim, KS.gas.γ, uq)
            f = uq_maxwellian(KS.vs.u, prim, uq)

            ctr[i] = begin
                if KS.set.space[3:4] == "1f"
                    ControlVolume(w, prim, f, 1)
                elseif KS.set.space[3:4] == "2f"
                    b = uq_energy_distribution(f, prim, KS.gas.K, uq)
                    ControlVolume(w, prim, f, b, 1)
                end
            end
        end

        w = uq_prim_conserve(prim, KS.gas.γ, uq)
        f = uq_maxwellian(KS.vs.u, prim, uq)
        for i in eachindex(face)
            face[i] = begin
                if KS.set.space[3:4] == "1f"
                    Interface(w, f, 1)
                elseif KS.set.space[3:4] == "2f"
                    b = uq_energy_distribution(f, prim, KS.gas.K, uq)
                    Interface(w, f, b, 1)
                end
            end
        end
    end

    return ctr, face

end


function mixture_fvm(KS, pSpace::PSpace1D, uq::AbstractUQ)

    idx0 = (eachindex(pSpace.x)|>collect)[1]
    idx1 = (eachindex(pSpace.x)|>collect)[end]

    ctr = OffsetArray{ControlVolume1D4F}(undef, idx0:idx1)

    if uq.uqMethod == "galerkin"

        # upstream
        primL = zeros(5, uq.nr + 1, 2)
        primL[:, 1, :] .= ks.ib.bc(ks.ps.x0, ks.ib.p)

        wL = get_conserved(primL, KS.gas.γ, uq)
        h0L, h1L, h2L, h3L = get_maxwell(KS.vs.u, primL, uq)

        EL = zeros(3, uq.nr + 1)
        BL = zeros(3, uq.nr + 1)
        BL[:, 1] .= KS.ib.BL

        # downstream
        primR = zeros(5, uq.nr + 1, 2)
        primR[:, 1, :] .= ks.ib.bc(ks.ps.x1, ks.ib.p)

        wR = get_conserved(primR, KS.gas.γ, uq)
        h0R, h1R, h2R, h3R = get_maxwell(KS.vs.u, primR, uq)

        ER = zeros(3, uq.nr + 1)
        BR = zeros(3, uq.nr + 1)
        BR[:, 1] .= KS.ib.BR

        lorenz = zeros(3, uq.nr + 1, 2)

        for i in eachindex(ctr)
            if i <= KS.ps.nx ÷ 2
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wL,
                    primL,
                    h0L,
                    h1L,
                    h2L,
                    h3L,
                    EL,
                    BL,
                    lorenz,
                )
            else
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wR,
                    primR,
                    h0R,
                    h1R,
                    h2R,
                    h3R,
                    ER,
                    BR,
                    lorenz,
                )
            end
        end

        face = Array{Interface1D4F}(undef, KS.ps.nx + 1)
        for i = 1:KS.ps.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    elseif uq.uqMethod == "collocation"

        # upstream
        primL = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primL[:, j, :] .= ks.ib.bc(ks.ps.x0, ks.ib.p)
        end

        wL = get_conserved(primL, KS.gas.γ)
        h0L, h1L, h2L, h3L = get_maxwell(KS.vs.u, primL)

        EL = zeros(3, uq.op.quad.Nquad)
        BL = zeros(3, uq.op.quad.Nquad)
        for j in axes(BL, 2)
            BL[:, j] .= KS.ib.BL
        end

        # downstream
        primR = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primR[:, j, :] .= ks.ib.bc(ks.ps.x1, ks.ib.p)
        end

        wR = get_conserved(primR, KS.gas.γ)
        h0R, h1R, h2R, h3R = get_maxwell(KS.vs.u, primR)

        ER = zeros(3, uq.op.quad.Nquad)
        BR = zeros(3, uq.op.quad.Nquad)
        for j in axes(BR, 2)
            BR[:, j] .= KS.ib.BR
        end

        lorenz = zeros(3, uq.op.quad.Nquad, 2)

        for i in eachindex(ctr)
            if i <= KS.ps.nx ÷ 2
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wL,
                    primL,
                    h0L,
                    h1L,
                    h2L,
                    h3L,
                    EL,
                    BL,
                    lorenz,
                )
            else
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wR,
                    primR,
                    h0R,
                    h1R,
                    h2R,
                    h3R,
                    ER,
                    BR,
                    lorenz,
                )
            end
        end

        #--- setup of cell interface ---#
        face = Array{Interface1D4F}(undef, KS.ps.nx + 1)
        for i = 1:KS.ps.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    end

    return ctr, face

end


function plasma_fvm(KS, pSpace::PSpace1D, uq::AbstractUQ)

    idx0 = (eachindex(pSpace.x)|>collect)[1]
    idx1 = (eachindex(pSpace.x)|>collect)[end]

    ctr = OffsetArray{ControlVolume1D4F}(undef, idx0:idx1)

    if uq.method == "galerkin"

        # upstream
        primL = zeros(5, uq.nr + 1, 2)
        primL[:, 1, :] .= ks.ib.bc(ks.ps.x0, ks.ib.p)

        wL = uq_prim_conserve(primL, KS.gas.γ, uq)
        h0L, h1L, h2L, h3L = uq_maxwellian(KS.vs.u, primL, uq)

        EL = zeros(3, uq.nr + 1)
        BL = zeros(3, uq.nr + 1)
        BL[:, 1] .= KS.ib.BL

        # downstream
        primR = zeros(5, uq.nr + 1, 2)
        primR[:, 1, :] .= ks.ib.bc(ks.ps.x1, ks.ib.p)

        wR = uq_prim_conserve(primR, KS.gas.γ, uq)
        h0R, h1R, h2R, h3R = uq_maxwellian(KS.vs.u, primR, uq)

        ER = zeros(3, uq.nr + 1)
        BR = zeros(3, uq.nr + 1)
        BR[:, 1] .= KS.ib.BR

        lorenz = zeros(3, uq.nr + 1, 2)

        for i in eachindex(ctr)
            if i <= KS.ps.nx ÷ 2
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wL,
                    primL,
                    h0L,
                    h1L,
                    h2L,
                    h3L,
                    EL,
                    BL,
                    lorenz,
                )
            else
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wR,
                    primR,
                    h0R,
                    h1R,
                    h2R,
                    h3R,
                    ER,
                    BR,
                    lorenz,
                )
            end
        end

        face = Array{Interface1D4F}(undef, KS.ps.nx + 1)
        for i = 1:KS.ps.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    elseif uq.method == "collocation"

        # upstream
        primL = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primL[:, j, :] .= ks.ib.bc(ks.ps.x0, ks.ib.p)
        end

        wL = uq_prim_conserve(primL, KS.gas.γ, uq)
        h0L, h1L, h2L, h3L = uq_maxwellian(KS.vs.u, primL, uq)

        EL = zeros(3, uq.op.quad.Nquad)
        BL = zeros(3, uq.op.quad.Nquad)
        for j in axes(BL, 2)
            BL[:, j] .= KS.ib.BL
        end

        # downstream
        primR = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primR[:, j, :] .= ks.ib.bc(ks.ps.x1, ks.ib.p)
        end

        wR = uq_prim_conserve(primR, KS.gas.γ, uq)
        h0R, h1R, h2R, h3R = uq_maxwellian(KS.vs.u, primR, uq)

        ER = zeros(3, uq.op.quad.Nquad)
        BR = zeros(3, uq.op.quad.Nquad)
        for j in axes(BR, 2)
            BR[:, j] .= KS.ib.BR
        end

        lorenz = zeros(3, uq.op.quad.Nquad, 2)

        for i in eachindex(ctr)
            if i <= KS.ps.nx ÷ 2
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wL,
                    primL,
                    h0L,
                    h1L,
                    h2L,
                    h3L,
                    EL,
                    BL,
                    lorenz,
                )
            else
                ctr[i] = ControlVolume1D4F(
                    KS.ps.x[i],
                    KS.ps.dx[i],
                    wR,
                    primR,
                    h0R,
                    h1R,
                    h2R,
                    h3R,
                    ER,
                    BR,
                    lorenz,
                )
            end
        end

        #--- setup of cell interface ---#
        face = Array{Interface1D4F}(undef, KS.ps.nx + 1)
        for i = 1:KS.ps.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    end

    return ctr, face

end
