# ============================================================
# Initialize Simulation
# ============================================================

"""
Initialize solver

"""
function initialize(configfilename::AbstractString, structure = "sol"::AbstractString)

    println("==============================================================")
    println("SKS.jl: A Software Package for Stochastic Kinetic Simulation")
    println("==============================================================")
    println("")
    println("reading configurations from $configfilename")
    println("")
    println("initializeing solver:")

    allowed = ["uqMethod", "nr", "nRec", "opType", "parameter1", "parameter2"]
    D = read_dict(configfilename, allowed)

    nr = D["nr"]
    nRec = D["nRec"]
    opType = D["opType"]
    parameter1 = D["parameter1"]
    parameter2 = D["parameter2"]
    uqMethod = D["uqMethod"]

    ks = SolverSet(configfilename)
    uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)

    if structure == "ctr"
        ctr, face = init_fvm(ks, uq)
        return ks, ctr, face, uq, 0.0
    elseif structure == "sol"
        sol, flux = init_sol(ks, uq)
        return ks, sol, flux, uq, 0.0
    else
        throw(ArgumentError("no data structure available for $structure"))
    end

end


"""
Initialize solution structs

"""
function init_sol(KS::AbstractSolverSet, uq::AbstractUQ)

    if uq.method == "galerkin"
        # upstream
        primL = zeros(axes(ks.ib.primL, 1), uq.nr + 1)
        primL[:, 1] .= KS.ib.primL

        # downstream
        primR = zeros(axes(ks.ib.primR, 1), uq.nr + 1)
        primR[:, 1] .= KS.ib.primR
    elseif uq.method == "collocation"
        # upstream
        primL = zeros(axes(KS.ib.primL, 1), uq.op.quad.Nquad)
        for j in axes(primL, 2)
            primL[:, j] .= KS.ib.primL
        end

        # downstream
        primR = zeros(axes(KS.ib.primR, 1), uq.op.quad.Nquad)
        for j in axes(primR, 2)
            primR[:, j] .= KS.ib.primR
        end
    end

    wL = uq_prim_conserve(primL, KS.gas.γ, uq)
    wR = uq_prim_conserve(primR, KS.gas.γ, uq)
    facew = [zeros(axes(wL)) for i = 1:KS.pSpace.nx+1]
    facefw = [zeros(axes(wL)) for i = 1:KS.pSpace.nx+1]

    if KS.set.space[1:2] == "1d"

        w = [deepcopy(wL) for i in axes(KS.pSpace.x, 1)]
        prim = [deepcopy(primL) for i in axes(KS.pSpace.x, 1)]
        for i in axes(w, 1)
            if i > KS.pSpace.nx ÷ 2
                w[i] .= deepcopy(wR)
                prim[i] .= deepcopy(primR)
            end
        end

        if KS.set.space[3:4] == "1f"

            if KS.set.space[5:6] == "1v"
                fL = uq_maxwellian(KS.vSpace.u, primL, uq)
                fR = uq_maxwellian(KS.vSpace.u, primR, uq)
            elseif KS.set.space[5:6] == "3v"
                fL = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, KS.vSpace.w, primL, uq)
                fR = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, KS.vSpace.w, primR, uq)
            end

            f = [deepcopy(fL) for i in axes(KS.pSpace.x, 1)]
            for i in axes(w, 1)
                if i > KS.pSpace.nx ÷ 2
                    f[i] .= deepcopy(fR)
                end
            end
            faceff = [zeros(axes(fL)) for i = 1:KS.pSpace.nx+1]

            sol = Solution1D1F(w, prim, f)
            flux = Flux1D1F(facew, facefw, faceff)

            return sol, flux

        elseif KS.set.space[3:4] == "2f"

            if KS.set.space[5:6] == "1v"
                hL = uq_maxwellian(KS.vSpace.u, primL, uq)
                hR = uq_maxwellian(KS.vSpace.u, primR, uq)
                bL = deepcopy(hL)
                bR = deepcopy(hR)
                for j in axes(bL, 2)
                    bL[:, j] .= hL[:, j] .* KS.gas.K ./ (2.0 * primL[end, j])
                    bR[:, j] .= hR[:, j] .* KS.gas.K ./ (2.0 * primR[end, j])
                end
            elseif KS.set.space[5:6] == "2v"
                hL = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primL, uq)
                hR = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primR, uq)
                bL = deepcopy(hL)
                bR = deepcopy(hR)
                for j in axes(bL, 3)
                    bL[:, :, j] .= hL[:, :, j] .* KS.gas.K ./ (2.0 * primL[end, j])
                    bR[:, :, j] .= hR[:, :, j] .* KS.gas.K ./ (2.0 * primR[end, j])
                end
            end

            h = [deepcopy(hL) for i in axes(KS.pSpace.x, 1)]
            b = [deepcopy(bL) for i in axes(KS.pSpace.x, 1)]
            for i in axes(w, 1)
                if i > KS.pSpace.nx ÷ 2
                    h[i] .= deepcopy(hR)
                    b[i] .= deepcopy(bR)
                end
            end
            facefh = [zeros(axes(hL)) for i = 1:KS.pSpace.nx+1]
            facefb = [zeros(axes(bL)) for i = 1:KS.pSpace.nx+1]

            sol = Solution1D2F(w, prim, h, b)
            flux = Flux1D2F(facew, facefw, facefh, facefb)

            return sol, flux

        end

    elseif KS.set.space[1:2] == "2d"

        #--- cell ---#
        w = [deepcopy(wL) for i in axes(KS.pSpace.x, 1), j in axes(KS.pSpace.x, 2)]
        prim = [deepcopy(primL) for i in axes(KS.pSpace.x, 1), j in axes(KS.pSpace.x, 2)]
        for j in axes(w, 2), i in axes(w, 1)
            if j > KS.pSpace.ny ÷ 2
                w[i, j] .= deepcopy(wR)
                prim[i, j] .= deepcopy(primR)
            end
        end

        #--- interface ---#
        n1 = [[1.0, 0.0] for i = 1:KS.pSpace.nx+1, j = 1:KS.pSpace.ny]
        n2 = [[0.0, 1.0] for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny+1]
        facew1 = [zeros(axes(wL)) for i = 1:KS.pSpace.nx+1, j = 1:KS.pSpace.ny]
        facefw1 = [zeros(axes(wL)) for i = 1:KS.pSpace.nx+1, j = 1:KS.pSpace.ny]
        facew2 = [zeros(axes(wL)) for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny+1]
        facefw2 = [zeros(axes(wL)) for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny+1]

        if KS.set.space[3:4] == "1f"

            #--- cell ---#
            fL = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primL, uq)
            fR = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primR, uq)

            f = [deepcopy(fL) for i in axes(KS.pSpace.x, 1), j in axes(KS.pSpace.x, 2)]
            for j in axes(w, 2), i in axes(w, 1)
                if j > KS.pSpace.ny ÷ 2
                    f[i, j] .= deepcopy(fR)
                end
            end

            #--- interface ---#
            faceff1 = [zeros(axes(fL)) for i = 1:KS.pSpace.nx+1, j = 1:KS.pSpace.ny]
            faceff2 = [zeros(axes(fL)) for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny+1]

            sol = Solution2D1F(w, prim, f)
            flux = Flux2D1F(
                n1,
                facew1,
                facefw1,
                faceff1,
                n2,
                facew2,
                facefw2,
                faceff2,
            )

        elseif KS.set.space[3:4] == "2f"

            #--- cell ---#
            hL, bL = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primL, uq, KS.gas.K)
            hR, bR = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primR, uq, KS.gas.K)

            h = [deepcopy(hL) for i in axes(KS.pSpace.x, 1), j in axes(KS.pSpace.x, 2)]
            b = [deepcopy(bL) for i in axes(KS.pSpace.x, 1), j in axes(KS.pSpace.x, 2)]

            for j in axes(w, 2), i in axes(w, 1)
                if j > KS.pSpace.ny ÷ 2
                    w[i, j] .= deepcopy(wR)
                    prim[i, j] .= deepcopy(primR)
                    h[i, j] .= deepcopy(hR)
                    b[i, j] .= deepcopy(bR)
                end
            end

            #--- interface ---#
            facefh1 = [zeros(axes(hL)) for i = 1:KS.pSpace.nx+1, j = 1:KS.pSpace.ny]
            facefb1 = [zeros(axes(bL)) for i = 1:KS.pSpace.nx+1, j = 1:KS.pSpace.ny]
            facefh2 = [zeros(axes(hL)) for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny+1]
            facefb2 = [zeros(axes(bL)) for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny+1]

            sol = Solution2D2F(w, prim, h, b)
            flux = Flux2D2F(
                n1,
                facew1,
                facefw1,
                facefh1,
                facefb1,
                n2,
                facew2,
                facefw2,
                facefh2,
                facefb2,
            )

        end

        return sol, flux

    end

end


"""
Initialize finite volume structs

"""
function init_fvm(KS::AbstractSolverSet, uq::AbstractUQ)

    if ndims(KS.ib.wL) == 1

        ctr, face = pure_fvm(KS, KS.pSpace, uq)

    elseif ndims(KS.ib.wL) == 2

        if KS.set.space[3:4] in ("3f", "4f")
            ctr, face = plasma_fvm(KS, KS.pSpace, uq)
        else
            ctr, face = mixture_fvm(KS, KS.pSpace, uq)
        end

    end

    return ctr, face

end


function pure_fvm(KS::AbstractSolverSet, pSpace::PSpace1D, uq::AbstractUQ)

    #--- setup of control volume ---#
    idx0 = (eachindex(pSpace.x)|>collect)[1]
    idx1 = (eachindex(pSpace.x)|>collect)[end]

    if KS.set.space[3:4] == "1f"
        ctr = OffsetArray{ControlVolume1D1F}(undef, idx0:idx1) # with ghost cells
    elseif KS.set.space[3:4] == "2f"
        ctr = OffsetArray{ControlVolume1D2F}(undef, idx0:idx1)
    elseif KS.set.space[3:4] == "4f"
        ctr = OffsetArray{ControlVolume1D4F}(undef, idx0:idx1)
    end

    if uq.method == "galerkin"

        # upstream
        primL = zeros(axes(KS.ib.primL, 1), uq.nr + 1)
        primL[:, 1] .= KS.ib.primL
        wL = uq_prim_conserve(primL, KS.gas.γ, uq)
        fL = uq_maxwellian(KS.vSpace.u, primL, uq)

        # downstream
        primR = zeros(axes(KS.ib.primR, 1), uq.nr + 1)
        primR[:, 1] .= KS.ib.primR
        wR = uq_prim_conserve(primR, KS.gas.γ, uq)
        fR = uq_maxwellian(KS.vSpace.u, primR, uq)

        for i in eachindex(ctr)
            if i <= KS.pSpace.nx ÷ 2
                ctr[i] = ControlVolume1D1F(KS.pSpace.x[i], KS.pSpace.dx[i], wL, primL, fL)
            else
                ctr[i] = ControlVolume1D1F(KS.pSpace.x[i], KS.pSpace.dx[i], wR, primR, fR)
            end
        end

        # --- setup of cell interface ---#
        face = Array{Interface1D1F}(undef, KS.pSpace.nx + 1)
        for i in eachindex(face)
            face[i] = Interface1D1F(wL, fL)
        end

    elseif uq.method == "collocation"

        # upstream
        primL = zeros(axes(KS.ib.primL, 1), uq.op.quad.Nquad)
        for j in axes(primL, 2)
            primL[:, j] .= KS.ib.primL
        end
        wL = uq_prim_conserve(primL, KS.gas.γ, uq)
        fL = uq_maxwellian(KS.vSpace.u, primL, uq)

        # downstream
        primR = zeros(axes(KS.ib.primL, 1), uq.op.quad.Nquad)
        for j in axes(primL, 2)
            primR[:, j] .= KS.ib.primR
        end
        wR = uq_prim_conserve(primR, KS.gas.γ, uq)
        fR = uq_maxwellian(KS.vSpace.u, primR, uq)

        for i in eachindex(ctr)
            if i <= KS.pSpace.nx ÷ 2
                ctr[i] = ControlVolume1D1F(KS.pSpace.x[i], KS.pSpace.dx[i], wL, primL, fL)
            else
                ctr[i] = ControlVolume1D1F(KS.pSpace.x[i], KS.pSpace.dx[i], wR, primR, fR)
            end
        end

        # --- setup of cell interface ---#
        face = Array{Interface1D1F}(undef, KS.pSpace.nx + 1)
        for i in eachindex(face)
            face[i] = Interface1D1F(wL, fL)
        end

    end

    return ctr, face

end


function mixture_fvm(KS::AbstractSolverSet, pSpace::PSpace1D, uq::AbstractUQ)

    idx0 = (eachindex(pSpace.x) |> collect)[1]
    idx1 = (eachindex(pSpace.x) |> collect)[end]

    ctr = OffsetArray{ControlVolume1D4F}(undef, idx0:idx1)

    if uq.uqMethod == "galerkin"

        # upstream
        primL = zeros(5, uq.nr+1, 2)
        primL[:,1,:] .= KS.ib.primL

        wL = get_conserved(primL, KS.gas.γ, uq)
        h0L, h1L, h2L, h3L = get_maxwell(KS.vSpace.u, primL, uq)

        EL = zeros(3, uq.nr+1)
        BL = zeros(3, uq.nr+1); BL[:,1] .= KS.ib.BL

        # downstream
        primR = zeros(5, uq.nr+1, 2)
        primR[:,1,:] .= KS.ib.primR

        wR = get_conserved(primR, KS.gas.γ, uq)
        h0R, h1R, h2R, h3R = get_maxwell(KS.vSpace.u, primR, uq)

        ER = zeros(3, uq.nr+1)
        BR = zeros(3, uq.nr+1); BR[:,1] .= KS.ib.BR

        lorenz = zeros(3, uq.nr+1, 2)

        for i in eachindex(ctr)
            if i <= KS.pSpace.nx ÷ 2
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wL, primL,
                                            h0L, h1L, h2L, h3L, EL, BL, lorenz )
            else
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wR, primR,
                                            h0R, h1R, h2R, h3R, ER, BR, lorenz)
            end
        end

        face = Array{Interface1D4F}(undef, KS.pSpace.nx+1)
        for i=1:KS.pSpace.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    elseif uq.uqMethod == "collocation"

        # upstream
        primL = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primL[:,j,:] .= KS.ib.primL
        end

        wL = get_conserved(primL, KS.gas.γ)
        h0L, h1L, h2L, h3L = get_maxwell(KS.vSpace.u, primL)

        EL = zeros(3, uq.op.quad.Nquad)
        BL = zeros(3, uq.op.quad.Nquad)
        for j in axes(BL, 2)
            BL[:,j] .= KS.ib.BL
        end

        # downstream
        primR = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primR[:,j,:] .= KS.ib.primR
        end

        wR = get_conserved(primR, KS.gas.γ)
        h0R, h1R, h2R, h3R = get_maxwell(KS.vSpace.u, primR)

        ER = zeros(3, uq.op.quad.Nquad)
        BR = zeros(3, uq.op.quad.Nquad)
        for j in axes(BR, 2)
            BR[:,j] .= KS.ib.BR
        end

        lorenz = zeros(3, uq.op.quad.Nquad, 2)

        for i in eachindex(ctr)
            if i <= KS.pSpace.nx ÷ 2
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wL, primL,
                                            h0L, h1L, h2L, h3L, EL, BL, lorenz )
            else
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wR, primR,
                                            h0R, h1R, h2R, h3R, ER, BR, lorenz)
            end
        end

        #--- setup of cell interface ---#
        face = Array{Interface1D4F}(undef, KS.pSpace.nx+1)
        for i=1:KS.pSpace.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    end

    return ctr, face

end


function plasma_fvm(KS::AbstractSolverSet, pSpace::PSpace1D, uq::AbstractUQ)

    idx0 = (eachindex(pSpace.x) |> collect)[1]
    idx1 = (eachindex(pSpace.x) |> collect)[end]

    ctr = OffsetArray{ControlVolume1D4F}(undef, idx0:idx1)

    if uq.method == "galerkin"

        # upstream
        primL = zeros(5, uq.nr+1, 2)
        primL[:,1,:] .= KS.ib.primL

        wL = uq_prim_conserve(primL, KS.gas.γ, uq)
        h0L, h1L, h2L, h3L = uq_maxwellian(KS.vSpace.u, primL, uq)

        EL = zeros(3, uq.nr+1)
        BL = zeros(3, uq.nr+1); BL[:,1] .= KS.ib.BL

        # downstream
        primR = zeros(5, uq.nr+1, 2)
        primR[:,1,:] .= KS.ib.primR

        wR = uq_prim_conserve(primR, KS.gas.γ, uq)
        h0R, h1R, h2R, h3R = uq_maxwellian(KS.vSpace.u, primR, uq)

        ER = zeros(3, uq.nr+1)
        BR = zeros(3, uq.nr+1); BR[:,1] .= KS.ib.BR

        lorenz = zeros(3, uq.nr+1, 2)

        for i in eachindex(ctr)
            if i <= KS.pSpace.nx ÷ 2
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wL, primL,
                                            h0L, h1L, h2L, h3L, EL, BL, lorenz )
            else
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wR, primR,
                                            h0R, h1R, h2R, h3R, ER, BR, lorenz)
            end
        end

        face = Array{Interface1D4F}(undef, KS.pSpace.nx+1)
        for i=1:KS.pSpace.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    elseif uq.method == "collocation"

        # upstream
        primL = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primL[:,j,:] .= KS.ib.primL
        end

        wL = uq_prim_conserve(primL, KS.gas.γ, uq)
        h0L, h1L, h2L, h3L = uq_maxwellian(KS.vSpace.u, primL, uq)

        EL = zeros(3, uq.op.quad.Nquad)
        BL = zeros(3, uq.op.quad.Nquad)
        for j in axes(BL, 2)
            BL[:,j] .= KS.ib.BL
        end

        # downstream
        primR = zeros(5, uq.op.quad.Nquad, 2)
        for j in axes(primL, 2)
            primR[:,j,:] .= KS.ib.primR
        end

        wR = uq_prim_conserve(primR, KS.gas.γ, uq)
        h0R, h1R, h2R, h3R = uq_maxwellian(KS.vSpace.u, primR, uq)

        ER = zeros(3, uq.op.quad.Nquad)
        BR = zeros(3, uq.op.quad.Nquad)
        for j in axes(BR, 2)
            BR[:,j] .= KS.ib.BR
        end

        lorenz = zeros(3, uq.op.quad.Nquad, 2)

        for i in eachindex(ctr)
            if i <= KS.pSpace.nx ÷ 2
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wL, primL,
                                            h0L, h1L, h2L, h3L, EL, BL, lorenz )
            else
                ctr[i] = ControlVolume1D4F( KS.pSpace.x[i], KS.pSpace.dx[i], wR, primR,
                                            h0R, h1R, h2R, h3R, ER, BR, lorenz)
            end
        end

        #--- setup of cell interface ---#
        face = Array{Interface1D4F}(undef, KS.pSpace.nx+1)
        for i=1:KS.pSpace.nx+1
            face[i] = Interface1D4F(wL, h0L, EL)
        end

    end

    return ctr, face

end
