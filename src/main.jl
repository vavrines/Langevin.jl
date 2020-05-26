# ============================================================
# Main Program
# ============================================================

cd(@__DIR__)

cd(@__FILE__)

include("SKS.jl")
using .SKS


ks, sol, flux, uq, simTime = SKS.initialize("config.txt")

dt = SKS.timestep(
    ks,
    uq,
    sol,
    simTime
)

SKS.calc_flux_kfvs!(
    ks,
    sol,
    flux,
    dt
)


residual = zeros(3)
SKS.update!(ks,uq,sol,flux,dt,residual)






using Plots
plot(ks.pSpace.x[1:100], sol.prim[1:100][1,1])
