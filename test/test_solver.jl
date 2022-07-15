# ============================================================
# Full Solver
# ============================================================

cd(@__DIR__)

###
# ctr-face
###

# 1d1f1v
ks, ctr, face, uq, simTime = Langevin.initialize("../example/config/sod.txt", "ctr")
dt = timestep(ks, uq, ctr, simTime)
res = zeros(3)
evolve!(ks, uq, ctr, face, dt)
update!(ks, uq, ctr, face, dt, res)

# 1d2f1v
ks, ctr, face, uq, simTime = Langevin.initialize("../example/config/shock.txt", "ctr")
evolve!(ks, uq, ctr, face, dt)
update!(ks, uq, ctr, face, dt, res)

###
# sol-flux
###

# 2d2f2v
ks, sol, flux, uq, simTime = Langevin.initialize("../example/config/cavity.txt")
KS = ks

dt = timestep(ks, uq, sol, simTime)
simTime = 0.0
residual = zeros(4)

for j = 1:KS.pSpace.ny
    for i = 1:KS.pSpace.nx+1
        un = KS.vSpace.u .* flux.n[1][i, j][1] .+ KS.vSpace.v .* flux.n[1][i, j][2]
        ut = KS.vSpace.v .* flux.n[1][i, j][1] .- KS.vSpace.u .* flux.n[1][i, j][2]

        for k in axes(sol.w[1, 1], 2)
            fw = @view flux.fw[1][i, j][:, k]
            fh = @view flux.fh[1][i, j][:, :, k]
            fb = @view flux.fb[1][i, j][:, :, k]

            flux_kfvs!(
                fw,
                fh,
                fb,
                sol.h[i-1, j][:, :, k] .+
                0.5 .* KS.pSpace.dx[i-1, j] .* sol.∇h[i-1, j][:, :, k, 1],
                sol.b[i-1, j][:, :, k] .+
                0.5 .* KS.pSpace.dx[i-1, j] .* sol.∇b[i-1, j][:, :, k, 1],
                sol.h[i, j][:, :, k] .-
                0.5 .* KS.pSpace.dx[i, j] .* sol.∇h[i, j][:, :, k, 1],
                sol.b[i, j][:, :, k] .-
                0.5 .* KS.pSpace.dx[i, j] .* sol.∇b[i, j][:, :, k, 1],
                un,
                ut,
                KS.vSpace.weights,
                dt,
                0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                sol.∇h[i-1, j][:, :, k, 1],
                sol.∇b[i-1, j][:, :, k, 1],
                sol.∇h[i, j][:, :, k, 1],
                sol.∇b[i, j][:, :, k, 1],
            )
        end
    end
end
