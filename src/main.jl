# ============================================================
# Main Program
# ============================================================

cd(@__DIR__)

include("SKS.jl")
using .SKS

#--- setup ---#
ks, sol, flux, uq, simTime = SKS.initialize("config.txt");

dt = SKS.timestep(
    ks,
    uq,
    sol,
    simTime
)

#--- main loop ---#
residual = zeros(3)
for iter = 1:1000

    SKS.calc_flux_kfvs!(
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

end

#--- visualization ---#
using Plots
pltx = deepcopy(ks.pSpace.x[1:100])
plty = zeros(100, 6)
for i in axes(plty, 1)
    plty[i,1] = sol.prim[i][1,5]
    plty[i,2] = sol.prim[i][2,5]
    plty[i,3] = 1. / sol.prim[i][3,5]
end
plot(pltx, plty[:,1:3])
