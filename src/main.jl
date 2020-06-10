# ============================================================
# Main Program
# ============================================================

using ProgressMeter

cd(@__DIR__)
include("SKS.jl")
using .SKS

#--- setup ---#
ks, sol, flux, uq, simTime = SKS.initialize("./config/cavity.txt");

dt = SKS.timestep(
    ks,
    uq,
    sol,
    simTime
)



#--- main loop ---#
residual = zeros(axes(ks.ib.primL))

nstep = floor(ks.set.maxTime / dt) |> Int

function solve!(ks, uq, sol, flux, simTime, dt, nstep)

    @showprogress for iter = 1:nstep
#=
        SKS.evolve!(
            ks,
            sol,
            flux,
            dt
        )
=#
        SKS.calc_flux_kfvs!(
            ks,
            sol,
            flux,
            dt
        )

        SKS.calc_flux_boundary_maxwell!(
            ks,
            sol,
            flux,
            dt
        )

        SKS.update!(
            ks,
            uq,
            sol,
            flux,
            dt,
            residual
        )

        simTime += dt

        if minimum(residual) < 1.e-6
            break
        end
    end

    return simTime

end

simTime = solve!(ks, uq, sol, flux, simTime, dt, 10)



#--- visualization ---#
using Plots
x = ks.pSpace.x[1:ks.pSpace.nx,1]
y = ks.pSpace.y[1,1:ks.pSpace.ny]

plty = zeros(ks.pSpace.nx, ks.pSpace.ny, 6)
for j in axes(plty, 2), i in axes(plty, 1)
    plty[i,j,1] = sol.prim[i,j][1,1]
    plty[i,j,2] = sol.prim[i,j][2,1]
    plty[i,j,3] = sol.prim[i,j][3,1]
    plty[i,j,4] = 1. / sol.prim[i,j][4,1]
end
contour(x,y,plty[:,:,4]',fill=true)
