using CairoMakie, Langevin
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
ks, ctr, face, uq, t = initialize("config/sod.txt", :ctr)

function init_uq!(KS, ctr, uq)
    for i in eachindex(ctr)
        if i <= KS.pSpace.nx ÷ 2
            if uq.method == "collocation"
                for j in 1:uq.op.quad.Nquad
                    ctr[i].prim[1, j] *= uq.pceSample[j]
                    ctr[i].w .= Langevin.uq_prim_conserve(ctr[i].prim, KS.gas.γ, uq)
                    ctr[i].f .= Langevin.uq_maxwellian(KS.vSpace.u, ctr[i].prim, uq)
                end
            elseif uq.method == "galerkin"
                for j in 1:uq.nr+1
                    ctr[i].prim[1, :] .= uq.pce
                    ctr[i].w .= Langevin.uq_prim_conserve(ctr[i].prim, KS.gas.γ, uq)
                    ctr[i].f .= Langevin.uq_maxwellian(KS.vSpace.u, ctr[i].prim, uq)
                end
            end
        end
    end
end

init_uq!(ks, ctr, uq)

t = 0.0
dt = timestep(ks, uq, ctr, t)
nt = ks.set.maxTime ÷ dt |> Int
res = zeros(3)

@showprogress for iter in 1:nt
    KitBase.reconstruct!(ks, ctr)
    evolve!(ks, uq, ctr, face, dt)
    update!(ks, uq, ctr, face, dt, res)
end

begin
    solmat = zeros(ks.pSpace.nx, 10, size(ctr[1].prim, 2))
    for i in 1:ks.pSpace.nx
        solmat[i, 1:2, :] .= ctr[i].prim[1:2, :]
        solmat[i, 3, :] .= Langevin.lambda_tchaos(ctr[i].prim[3, :], 1, uq)
    end

    sol = zeros(ks.pSpace.nx, 10, 2)
    for i in 1:ks.pSpace.nx, j in 1:3
        sol[i, j, 1] = mean(solmat[i, j, :], uq.op)
        sol[i, j, 2] = std(solmat[i, j, :], uq.op)
    end
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="n", title="")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 1, 1]; label="mean")
    lines!(ks.ps.x[1:ks.ps.nx], sol[:, 1, 2]; label="std")
    axislegend()
    fig
end
