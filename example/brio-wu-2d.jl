using Revise, ProgressMeter, OffsetArrays, Kinetic
#include("/home/vavrines/Coding/SKS.jl/src/SKS.jl")
#using .SKS
begin
    using Reexport
    using OffsetArrays
    using Plots
    using JLD2
    using PolyChaos

    include("/home/vavrines/Coding/Langevin.jl/src/uq.jl")
    include("/home/vavrines/Coding/Langevin.jl/src/kinetic.jl")
    include("/home/vavrines/Coding/Langevin.jl/src/initialize.jl")
    include("/home/vavrines/Coding/Langevin.jl/src/flux.jl")
    include("/home/vavrines/Coding/Langevin.jl/src/out.jl")
    include("/home/vavrines/Coding/Langevin.jl/src/solver.jl")
    include("/home/vavrines/Coding/Langevin.jl/src/step.jl")
end

begin
    case = "weibel"
    space = "1d3f2v"
    nSpecies = 2
    interpOrder = 2
    limiter = "vanleer"
    cfl = 0.3
    maxTime = 0.1

    # physical space
    x0 = 0
    x1 = 1
    nx = 100
    pMeshType = "uniform"
    nxg = 2

    # velocity space
    vMeshType = "rectangle"
    umin = -5
    umax = 5
    nu = 28
    nug = 0

    vmin = -5
    vmax = 5
    nv = 28
    nvg = 0

    # random space
    uqMethod = "collocation"
    nr = 1
    nRec = 3
    opType = "uniform"
    parameter1 = 0.95
    parameter2 = 1.05

    # gas
    knudsen = 0.000001
    mach = 0.0
    prandtl = 1
    inK = 0

    mi = 1
    ni = 0.5
    #me = 0.001
    me = 0.0005446623
    ne = 0.5
    lD = 0.01
    rL = 0.003

    # electromagnetic field
    sol = 100
    echi = 1
    bnu = 1
end

begin
    γ = heat_capacity_ratio(inK, 3)
    set = Setup(case, space, nSpecies, interpOrder, limiter, cfl, maxTime)
    pSpace = PSpace1D(x0, x1, nx, pMeshType, nxg)

    ue0 = umin * sqrt(mi / me)
    ue1 = umax * sqrt(mi / me)
    ve0 = vmin * sqrt(mi / me)
    ve1 = vmax * sqrt(mi / me)
    kne = knudsen * (me / mi)

    vSpace = MVSpace2D(umin, umax, ue0, ue1, nu, vmin, vmax, ve0, ve1, nv, vMeshType, nug, nvg)
    plasma = Plasma2D([knudsen,kne], mach, prandtl, inK, γ, mi, ni, me, ne, lD, rL, sol, echi, bnu)

    begin
        # upstream
        primL = zeros(5, 2)
        primL[1, 1] = 1.0 * mi
        primL[2, 1] = 0.0
        primL[3, 1] = 0.0
        primL[4, 1] = 0.0
        primL[5, 1] = mi / 1.0
        primL[1, 2] = 1.0 * me
        primL[2, 2] = 0.0
        primL[3, 2] = 0.0
        primL[4, 2] = 0.0
        primL[5, 2] = me / 1.0

        wL = mixture_prim_conserve(primL, γ)
        h0L = mixture_maxwellian(vSpace.u, vSpace.v, primL)

        h1L = similar(h0L)
        h2L = similar(h0L)
        for j in axes(h0L, 3)
            h1L[:, :, j] .= primL[4, j] .* h0L[:, :, j]
            h2L[:, :, j] .= (primL[4, j]^2 + 1.0 / (2.0 * primL[end, j])) .* h0L[:, :, j]
        end

        EL = zeros(3)
        BL = zeros(3)
        BL[1] = 0.75
        BL[2] = 1.0

        # downstream
        primR = zeros(5, 2)
        primR[1, 1] = 0.125 * mi
        primR[2, 1] = 0.0
        primR[3, 1] = 0.0
        primR[4, 1] = 0.0
        primR[5, 1] = mi * 1.25
        primR[1, 2] = 0.125 * me
        primR[2, 2] = 0.0
        primR[3, 2] = 0.0
        primR[4, 2] = 0.0
        primR[5, 2] = me * 1.25

        wR = mixture_prim_conserve(primR, γ)
        h0R = mixture_maxwellian(vSpace.u, vSpace.v, primR)

        h1R = similar(h0R)
        h2R = similar(h0R)
        for j in axes(h0R, 3)
            h1R[:, :, j] .= primR[4, j] .* h0R[:, :, j]
            h2R[:, :, j] .= (primR[4, j]^2 + 1.0 / (2.0 * primR[end, j])) .* h0R[:, :, j]
        end

        ER = zeros(3)
        BR = zeros(3)
        BR[1] = 0.75
        BR[2] = -1.0

        lorenzL = zeros(3, 2)
        lorenzR = zeros(3, 2)
        bcL = zeros(5, 2)
        bcR = zeros(5, 2)
    end

    ib = IB3F(
            wL,
            primL,
            h0L,
            h1L,
            h2L,
            bcL,
            EL,
            BL,
            lorenzL,
            wR,
            primR,
            h0R,
            h1R,
            h2R,
            bcR,
            ER,
            BR,
            lorenzR,
        )

    outputFolder = pwd()

    ks = SolverSet(set, pSpace, vSpace, plasma, ib, outputFolder)
    KS = ks

    ctr = OffsetArray{ControlVolume1D3F}(undef, axes(KS.pSpace.x, 1))
    face = Array{Interface1D3F}(undef, KS.pSpace.nx + 1)

    uq = UQ1D(nr, nRec, parameter1, parameter2, opType, uqMethod)
end

begin
    idx0 = (eachindex(pSpace.x) |> collect)[1]
    idx1 = (eachindex(pSpace.x) |> collect)[end]

    # upstream
    primL = zeros(5, uq.op.quad.Nquad, 2)
    for j in axes(primL, 2)
        primL[:,j,:] .= KS.ib.primL
    end

    wL = uq_prim_conserve(primL, KS.gas.γ, uq)
    h0L, h1L, h2L = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primL, uq)

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
    h0R, h1R, h2R= uq_maxwellian(KS.vSpace.u, KS.vSpace.v, primR, uq)

    ER = zeros(3, uq.op.quad.Nquad)
    BR = zeros(3, uq.op.quad.Nquad)
    for j in axes(BR, 2)
        BR[:,j] .= KS.ib.BR
    end

    lorenz = zeros(3, uq.op.quad.Nquad, 2)

    for i in eachindex(ctr)
        if i <= KS.pSpace.nx ÷ 2                
            ctr[i] = ControlVolume1D3F( KS.pSpace.x[i], KS.pSpace.dx[i], wL, primL, 
                                        h0L, h1L, h2L, EL, BL, lorenz )
        else
            ctr[i] = ControlVolume1D3F( KS.pSpace.x[i], KS.pSpace.dx[i], wR, primR, 
                                        h0R, h1R, h2R, ER, BR, lorenz)
        end
    end

    face = Array{Interface1D3F}(undef, KS.pSpace.nx+1)
    for i=1:KS.pSpace.nx+1
        face[i] = Interface1D3F(wL, h0L, EL)
    end
end

iter = 0
res = zeros(5, 2)
simTime = 0.0
dt = timestep(KS, ctr, simTime, uq)
#dt = SKS.timestep(KS, ctr, simTime, uq)
nt = Int(floor(ks.set.maxTime / dt))

res = zeros(5, 2)
@showprogress for iter in 1:nt
    #dt = timestep(KS, ctr, simTime, uq)
#    Kinetic.reconstruct!(KS, ctr)
    
    #evolve!(KS, uq, ctr, face, dt)

    @inbounds Threads.@threads for i in eachindex(face)
        uqflux_flow!(KS, ctr[i-1], face[i], ctr[i], dt, mode=:kfvs)
        
        for j = 1:uq.op.quad.Nquad
            femL = @view face[i].femL[:, j]
            femR = @view face[i].femR[:, j]
    
            flux_em!(
                femL,
                femR,
                ctr[i-2].E[:, j],
                ctr[i-2].B[:, j],
                ctr[i-1].E[:, j],
                ctr[i-1].B[:, j],
                ctr[i].E[:, j],
                ctr[i].B[:, j],
                ctr[i+1].E[:, j],
                ctr[i+1].B[:, j],
                ctr[i-1].ϕ[j],
                ctr[i].ϕ[j],
                ctr[i-1].ψ[j],
                ctr[i].ψ[j],
                ctr[i-1].dx,
                ctr[i].dx,
                KS.gas.A1p,
                KS.gas.A1n,
                KS.gas.D1,
                KS.gas.sol,
                KS.gas.χ,
                KS.gas.ν,
                dt,
            )
        end
        
    end

    update!(KS, uq, ctr, face, dt, res)

    #iter += 1
    #t += dt

    #if iter%1000 == 0
    #    println("iter: $(iter), time: $(simTime), dt: $(dt), res: $(res[1:5,1:2])")
    #end

    #if simTime > KS.set.maxTime || maximum(res) < 1.e-7
    #    break
    #end
end

sol = zeros(ks.pSpace.nx, 10, 2)
for i in 1:ks.pSpace.nx
    sol[i, 1, 1] = ctr[i].prim[1,1,1]
    sol[i, 1, 2] = ctr[i].prim[1,1,2] / ks.gas.me
    sol[i, 2:4, 1] .= ctr[i].prim[2:4,1,1]
    sol[i, 2:4, 2] .= ctr[i].prim[2:4,1,2]
    sol[i, 5, 1] = 1. / ctr[i].prim[5,1,1]
    sol[i, 5, 2] = ks.gas.me / ctr[i].prim[5,1,2]

    sol[i, 6, 1] = ctr[i].B[2,1]
    sol[i, 6, 2] = ctr[i].E[1,1]
end
using Plots
plot(ks.pSpace.x[1:ks.pSpace.nx], sol[:,1,1])
plot!(ks.pSpace.x[1:ks.pSpace.nx], sol[:,1,2])

plot(ks.pSpace.x[1:ks.pSpace.nx], sol[:,6,1])
plot!(ks.pSpace.x[1:ks.pSpace.nx], sol[:,6,2])

@save "sol.jld2" ks uq ctr
