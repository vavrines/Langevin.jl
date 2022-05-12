using Langevin
using KitBase.ProgressMeter: @showprogress

function ev!(KS, sol, flux, dt)
    @inbounds for i in eachindex(flux.fw)
        for j in axes(sol.w[1], 2)
            fw = @view flux.fw[i][:, j]
            fh = @view flux.fh[i][:, :, j]
            fb = @view flux.fb[i][:, :, j]

            flux_kfvs!(
                fw,
                fh,
                fb,
                sol.h[i-1][:, :, j] .+ 0.5 .* KS.pSpace.dx[i-1] .* sol.∇h[i-1][:, :, j],
                sol.b[i-1][:, :, j] .+ 0.5 .* KS.pSpace.dx[i-1] .* sol.∇b[i-1][:, :, j],
                sol.h[i][:, :, j] .- 0.5 .* KS.pSpace.dx[i] .* sol.∇h[i][:, :, j],
                sol.b[i][:, :, j] .- 0.5 .* KS.pSpace.dx[i] .* sol.∇b[i][:, :, j],
                KS.vSpace.u,
                KS.vSpace.v,
                KS.vSpace.weights,
                dt,
                1.0, # interface length
                sol.∇h[i-1][:, :, j],
                sol.∇b[i-1][:, :, j],
                sol.∇h[i][:, :, j],
                sol.∇b[i][:, :, j],
            )
        end
    end
end

function up!(KS, uq, sol, flux, dt, residual)
    w_old = deepcopy(sol.w)

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        @. sol.w[i] += (flux.fw[i] - flux.fw[i+1]) / KS.pSpace.dx[i]
        sol.prim[i] .= uq_conserve_prim(sol.w[i], KS.gas.γ, uq)
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    H = [
        uq_maxwellian(KS.vSpace.u, KS.vSpace.v, sol.prim[i], uq) for
        i in eachindex(sol.prim)
    ]
    B = deepcopy(H)
    for i = 1:KS.pSpace.nx
        for j in axes(B[1], 3)
            B[i][:, :, j] .= H[i][:, :, j] .* KS.gas.K ./ (2.0 .* sol.prim[i][end, j])
        end
    end

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        for j in axes(sol.w[1], 2)
            @. sol.h[i][:, :, j] =
                (
                    sol.h[i][:, :, j] +
                    (flux.fh[i][:, :, j] - flux.fh[i+1][:, :, j]) / KS.pSpace.dx[i] +
                    dt / τ[i][j] * H[i][:, :, j]
                ) / (1.0 + dt / τ[i][j])
            @. sol.b[i][:, :, j] =
                (
                    sol.b[i][:, :, j] +
                    (flux.fb[i][:, :, j] - flux.fb[i+1][:, :, j]) / KS.pSpace.dx[i] +
                    dt / τ[i][j] * B[i][:, :, j]
                ) / (1.0 + dt / τ[i][j])
        end
    end

    # record residuals
    sumRes = zeros(axes(sol.w[1], 1))
    sumAvg = zeros(axes(sol.w[1], 1))
    for j in axes(sumRes, 1)
        for i = 1:KS.pSpace.nx
            sumRes[j] += sum((sol.w[i][j, :] .- w_old[i][j, :]) .^ 2)
            sumAvg[j] += sum(abs.(sol.w[i][j, :]))
        end
    end
    @. residual = sqrt(sumRes * KS.pSpace.nx) / (sumAvg + 1.e-7)

    return nothing
end

set = Setup(case = "layer", space = "1d2f2v", maxTime = 0.5539)
ps = PSpace1D(-1, 1, 1000, 1)
vs = VSpace2D(-4.5, 4.5, 32, -4.5, 4.5, 64)
gas = Gas(Kn = 0.005, K = 1)

ib = begin
    primL = [1.0, 0.0, 1.0, 1.0]
    primR = [1.0, 0.0, -1.0, 2.0]
    wL = prim_conserve(primL, gas.γ)
    wR = prim_conserve(primR, gas.γ)

    p = (
        x0 = ps.x0,
        x1 = ps.x1,
        u = vs.u,
        γ = gas.γ,
        K = gas.K,
        wL = wL,
        wR = wR,
        primL = primL,
        primR = primR,
    )

    fw = function (x, p)
        if x <= (p.x0 + p.x1) / 2
            return p.wL
        else
            return p.wR
        end
    end

    bc = function (x, p)
        if x <= (p.x0 + p.x1) / 2
            return p.primL
        else
            return p.primR
        end
    end

    ff = function (x, p)
        w = ifelse(x <= (p.x0 + p.x1) / 2, p.wL, p.wR)
        prim = conserve_prim(w, p.γ)
        h = maxwellian(p.u, p.v, prim)
        b = @. h * p.K / 2 / prim[end]
        return h, b
    end

    IB2F{typeof(bc)}(fw, ff, bc, p)
end

ks = SolverSet(set, ps, vs, gas, ib)
uq = UQ1D(5, 10, 0.9, 1.1, "uniform", "collocation")
sol, flux = Langevin.init_collo_sol(ks, uq)

# add uncertainty
for i in eachindex(ks.pSpace.x)
    if i <= ks.pSpace.nx ÷ 2
        for j in axes(sol.w[1], 2)
            sol.prim[i][3, j] *= uq.pceSample[j]
            sol.w[i] .= uq_prim_conserve(sol.prim[i], ks.gas.γ, uq)
            sol.h[i], sol.b[i] =
                uq_maxwellian(ks.vSpace.u, ks.vSpace.v, sol.prim[i], uq, ks.gas.K)
        end
    end
end

simTime = 0.0
iter = 0

dt = 0.2 * ks.pSpace.dx[1]
nt = floor(ks.set.maxTime / dt + 1) |> Int

#simTime, iter = solve!(ks, uq, sol, flux, simTime, dt, nt)

reconstruct!(ks, sol)
ev!(ks, sol, flux, dt)
up!(ks, uq, sol, flux, dt, zeros(4, uq.op.quad.Nquad))
