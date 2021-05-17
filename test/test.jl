# ============================================================
# Main Program
# ============================================================

using ProgressMeter
using BenchmarkTools
using Langevin

ks, sol, flux, uq, simTime = Langevin.initialize("./config/cavity.txt");

dt = SKS.timestep(ks, uq, sol, simTime)

simTime = 0.0
residual = zeros(axes(ks.ib.primL))

KS = ks

@time for j = 1:KS.pSpace.ny
    for i = 1:KS.pSpace.nx+1
        un = KS.vSpace.u .* flux.n1[i, j][1] .+ KS.vSpace.v .* flux.n1[i, j][2]
        ut = KS.vSpace.v .* flux.n1[i, j][1] .- KS.vSpace.u .* flux.n1[i, j][2]

        for k in axes(sol.w[1, 1], 2)
            flux.fw1[i, j][:, k], flux.fh1[i, j][:, :, k], flux.fb1[i, j][:, :, k] =
                flux_kfvs(
                    sol.h[i-1, j][:, :, k] .+
                    0.5 .* KS.pSpace.dx[i-1, j] .* sol.sh[i-1, j][:, :, k, 1],
                    sol.b[i-1, j][:, :, k] .+
                    0.5 .* KS.pSpace.dx[i-1, j] .* sol.sb[i-1, j][:, :, k, 1],
                    sol.h[i, j][:, :, k] .-
                    0.5 .* KS.pSpace.dx[i, j] .* sol.sh[i, j][:, :, k, 1],
                    sol.b[i, j][:, :, k] .-
                    0.5 .* KS.pSpace.dx[i, j] .* sol.sb[i, j][:, :, k, 1],
                    un,
                    ut,
                    KS.vSpace.weights,
                    dt,
                    0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                    sol.sh[i-1, j][:, :, k, 1],
                    sol.sb[i-1, j][:, :, k, 1],
                    sol.sh[i, j][:, :, k, 1],
                    sol.sb[i, j][:, :, k, 1],
                )
        end
    end
end

@time for j = 1:KS.pSpace.ny
    for i = 1:KS.pSpace.nx+1
        un = KS.vSpace.u .* flux.n1[i, j][1] .+ KS.vSpace.v .* flux.n1[i, j][2]
        ut = KS.vSpace.v .* flux.n1[i, j][1] .- KS.vSpace.u .* flux.n1[i, j][2]

        for k in axes(sol.w[1, 1], 2)
            flux_kfvs!(
                flux.fw1[i, j][:, k],
                flux.fh1[i, j][:, :, k],
                flux.fb1[i, j][:, :, k],
                sol.h[i-1, j][:, :, k] .+
                0.5 .* KS.pSpace.dx[i-1, j] .* sol.sh[i-1, j][:, :, k, 1],
                sol.b[i-1, j][:, :, k] .+
                0.5 .* KS.pSpace.dx[i-1, j] .* sol.sb[i-1, j][:, :, k, 1],
                sol.h[i, j][:, :, k] .-
                0.5 .* KS.pSpace.dx[i, j] .* sol.sh[i, j][:, :, k, 1],
                sol.b[i, j][:, :, k] .-
                0.5 .* KS.pSpace.dx[i, j] .* sol.sb[i, j][:, :, k, 1],
                un,
                ut,
                KS.vSpace.weights,
                dt,
                0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                sol.sh[i-1, j][:, :, k, 1],
                sol.sb[i-1, j][:, :, k, 1],
                sol.sh[i, j][:, :, k, 1],
                sol.sb[i, j][:, :, k, 1],
            )
        end
    end
end
