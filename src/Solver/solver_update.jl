"""
$(SIGNATURES)

Update solution
"""
update!(
    KS::AbstractSolverSet,
    uq::AbstractUQ,
    sol::AbstractSolution,
    flux::AbstractFlux,
    dt,
    residual,
) = update!(KS, KS.ps, uq, sol, flux, dt, residual)

function update!(
    KS::AbstractSolverSet,
    ps::AbstractPhysicalSpace1D,
    uq::AbstractUQ,
    sol::Solution1F{T1,T2,T3,T4,1},
    flux::Flux1F,
    dt,
    residual,
) where {T1,T2,T3,T4}

    w_old = deepcopy(sol.w)

    @inbounds @threads for i = 1:KS.ps.nx
        @. sol.w[i] += (flux.fw[i] - flux.fw[i+1]) / KS.ps.dx[i]
        sol.prim[i] .= uq_conserve_prim(sol.w[i], KS.gas.γ, uq)
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    M = [uq_maxwellian(KS.vs.u, sol.prim[i], uq) for i in eachindex(sol.prim)]

    @inbounds @threads for i = 1:KS.ps.nx
        for j in axes(sol.w[1], 2)
            @. sol.f[i][:, j] =
                (
                    sol.f[i][:, j] +
                    (flux.ff[i][:, j] - flux.ff[i+1][:, j]) / KS.ps.dx[i] +
                    dt / τ[i][j] * M[i][:, j]
                ) / (1.0 + dt / τ[i][j])
        end
    end

    # record residuals
    sumRes = zeros(axes(w_old[1], 1))
    sumAvg = zeros(axes(w_old[1], 1))
    for j in axes(sumRes, 1)
        for i = 1:KS.ps.nx
            sumRes[j] += sum((sol.w[i][j, :] .- w_old[i][j, :]) .^ 2)
            sumAvg[j] += sum(abs.(sol.w[i][j, :]))
        end
    end
    @. residual = sqrt(sumRes * KS.ps.nx) / (sumAvg + 1.e-7)

    return nothing

end

function update!(
    KS::AbstractSolverSet,
    ps::AbstractPhysicalSpace1D,
    uq::AbstractUQ,
    sol::Solution2F{T1,T2,T3,T4,1},
    flux::Flux2F,
    dt,
    residual,
) where {T1,T2,T3,T4}

    w_old = deepcopy(sol.w)

    @inbounds @threads for i = 1:KS.ps.nx
        @. sol.w[i] += (flux.fw[i] - flux.fw[i+1]) / KS.ps.dx[i]
        sol.prim[i] .= uq_conserve_prim(sol.w[i], KS.gas.γ, uq)
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    H = [uq_maxwellian(KS.vs.u, sol.prim[i], uq) for i in eachindex(sol.prim)]
    B = [
        uq_energy_distribution(H[i], sol.prim[i], KS.gas.K, uq) for i in eachindex(sol.prim)
    ]

    @inbounds @threads for i = 1:KS.ps.nx
        for j in axes(sol.w[1], 2)
            @. sol.h[i][:, j] =
                (
                    sol.h[i][:, j] +
                    (flux.fh[i][:, j] - flux.fh[i+1][:, j]) / KS.ps.dx[i] +
                    dt / τ[i][j] * H[i][:, j]
                ) / (1.0 + dt / τ[i][j])
            @. sol.b[i][:, j] =
                (
                    sol.h[i][:, j] +
                    (flux.fb[i][:, j] - flux.fb[i+1][:, j]) / KS.ps.dx[i] +
                    dt / τ[i][j] * B[i][:, j]
                ) / (1.0 + dt / τ[i][j])
        end
    end

    # record residuals
    sumRes = zeros(axes(w_old[1], 1))
    sumAvg = zeros(axes(w_old[1], 1))
    for j in axes(sumRes, 1)
        for i = 1:KS.ps.nx
            sumRes[j] += sum((sol.w[i][j, :] .- w_old[i][j, :]) .^ 2)
            sumAvg[j] += sum(abs.(sol.w[i][j, :]))
        end
    end
    @. residual = sqrt(sumRes * KS.ps.nx) / (sumAvg + 1.e-7)

    return nothing

end

function update!(
    KS::SolverSet,
    ps::AbstractPhysicalSpace2D,
    uq::AbstractUQ,
    sol::Solution2F{T1,T2,T3,T4,2},
    flux::Flux2F,
    dt,
    residual,
) where {T1,T2,T3,T4}

    w_old = deepcopy(sol.w)

    @inbounds @threads for j = 1:KS.ps.ny
        for i = 1:KS.ps.nx
            @. sol.w[i, j] +=
                (
                    flux.fw[1][i, j] - flux.fw[1][i+1, j] + flux.fw[2][i, j] -
                    flux.fw[2][i, j+1]
                ) / (KS.ps.dx[i, j] * KS.ps.dy[i, j])
            sol.prim[i, j] .= uq_conserve_prim(sol.w[i, j], KS.gas.γ, uq)
        end
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    H = [
        uq_maxwellian(KS.vs.u, KS.vs.v, sol.prim[i, j], uq) for i in axes(sol.prim, 1),
        j in axes(sol.prim, 2)
    ]
    B = deepcopy(H)
    for i in axes(sol.prim, 1), j in axes(sol.prim, 2)
        for k in axes(B[1, 1], 3)
            B[i, j][:, :, k] .=
                B[i, j][:, :, k] .* KS.gas.K ./ (2.0 * sol.prim[i, j][end, k])
        end
    end

    @inbounds @threads for i = 1:KS.ps.nx
        for j = 1:KS.ps.ny
            for k in axes(sol.w[1, 1], 2)
                @. sol.h[i, j][:, :, k] =
                    (
                        sol.h[i, j][:, :, k] +
                        (
                            flux.fh[1][i, j][:, :, k] - flux.fh[1][i+1, j][:, :, k] +
                            flux.fh[2][i, j][:, :, k] - flux.fh[2][i, j+1][:, :, k]
                        ) / (KS.ps.dx[i, j] * KS.ps.dy[i, j]) +
                        dt / τ[i, j][k] * H[i, j][:, :, k]
                    ) / (1.0 + dt / τ[i, j][k])
                @. sol.b[i, j][:, :, k] =
                    (
                        sol.b[i, j][:, :, k] +
                        (
                            flux.fb[1][i, j][:, :, k] - flux.fb[1][i+1, j][:, :, k] +
                            flux.fb[2][i, j][:, :, k] - flux.fb[2][i, j+1][:, :, k]
                        ) / (KS.ps.dx[i, j] * KS.ps.dy[i, j]) +
                        dt / τ[i, j][k] * B[i, j][:, :, k]
                    ) / (1.0 + dt / τ[i, j][k])
            end
        end
    end

    # record residuals
    sumRes = zeros(axes(w_old[1], 1))
    sumAvg = zeros(axes(w_old[1], 1))
    @inbounds for k in axes(sumRes, 1)
        for j = 1:KS.ps.ny
            for i = 1:KS.ps.nx
                sumRes[k] += sum((sol.w[i, j][k, :] .- w_old[i, j][k, :]) .^ 2)
                sumAvg[k] += sum(abs.(sol.w[i, j][k, :]))
            end
        end
    end

    @. residual = sqrt(sumRes * KS.ps.nx * KS.ps.ny) / (sumAvg + 1.e-7)

end

function update!(
    KS::AbstractSolverSet,
    uq::AbstractUQ,
    ctr::AV{T},
    face,
    dt,
    residual;
    coll = :bgk,
    bc = :fix,
    fn = step!,
) where {T<:AbstractControlVolume}

    sumRes = zeros(3)
    sumAvg = zeros(3)

    @inbounds @threads for i = 2:KS.ps.nx-1
        fn(KS, uq, face[i], ctr[i], face[i+1], (dt, KS.ps.dx[i], sumRes, sumAvg), coll)
    end

    for i in axes(residual, 1)
        residual[i] = sqrt(sumRes[i] * KS.ps.nx) / (sumAvg[i] + 1.e-7)
    end

    update_boundary!(
        KS,
        uq,
        ctr,
        face,
        dt,
        residual;
        coll = coll,
        bc = bc,
        isMHD = false,
        fn = fn,
    )

end

function update!(
    KS::AbstractSolverSet,
    uq::AbstractUQ,
    ctr::AV{ControlVolume1D3F},
    face,
    dt,
    residual::AM;
    coll = :bgk,
    bc = :extra,
    isMHD = true,
    fn = step!,
)

    sumRes = zeros(5, 2)
    sumAvg = zeros(5, 2)

    @inbounds @threads for i = 2:KS.ps.nx-1
        fn(
            KS,
            uq,
            face[i],
            ctr[i],
            face[i+1],
            (dt, KS.ps.dx[i], sumRes, sumAvg),
            coll,
            isMHD,
        )
    end

    for i in axes(residual, 1)
        @. residual[i, :] = sqrt(sumRes[i, :] * KS.ps.nx) / (sumAvg[i, :] + 1.e-7)
    end

    update_boundary!(
        KS,
        uq,
        ctr,
        face,
        dt,
        residual;
        coll = coll,
        bc = bc,
        isMHD = isMHD,
        fn = fn,
    )

end

"""
$(SIGNATURES)

1D4F1V
"""
function update!(
    KS::AbstractSolverSet,
    uq::AbstractUQ,
    ctr::AV{ControlVolume1D4F},
    face,
    dt,
    residual::AM;
    coll = :bgk,
    bc = :extra,
    isMHD = true,
    fn = step!,
)

    sumRes = zeros(5, 2)
    sumAvg = zeros(5, 2)

    @inbounds @threads for i = 2:KS.ps.nx-1
        fn(KS, uq, face[i], ctr[i], face[i+1], (dt, KS.ps.dx[i], sumRes, sumAvg))
    end

    for i in axes(residual, 1)
        @. residual[i, :] = sqrt(sumRes[i, :] * KS.ps.nx) / (sumAvg[i, :] + 1.e-7)
    end

    update_boundary!(
        KS,
        uq,
        ctr,
        face,
        dt,
        residual;
        coll = coll,
        bc = bc,
        isMHD = isMHD,
        fn = fn,
    )

end


"""
$(SIGNATURES)

Update boundary solution
"""
function update_boundary!(
    KS::AbstractSolverSet,
    uq::AbstractUQ,
    ctr::AV,
    face,
    dt,
    residual;
    coll = :bgk,
    bc = :extra,
    isMHD = true,
    fn = step!,
)

    if bc != :fix
        resL = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)
        avgL = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)
        resR = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)
        avgR = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)

        i = 1
        j = KS.ps.nx
        if KS.set.space[3:4] == "0f"
            fn(KS, uq, face[i], ctr[i], face[i+1], (dt, KS.ps.dx[i], resL, avgL))
            fn(KS, uq, face[j], ctr[j], face[j+1], (dt, KS.ps.dx[j], resR, avgR))
        elseif KS.set.space[3:4] in ["3f", "4f"]
            fn(
                KS,
                uq,
                face[i],
                ctr[i],
                face[i+1],
                (dt, KS.ps.dx[i], resL, avgL),
                coll,
                isMHD,
            )
            fn(
                KS,
                uq,
                face[j],
                ctr[j],
                face[j+1],
                (dt, KS.ps.dx[i], resR, avgR),
                coll,
                isMHD,
            )
        else
            fn(KS, uq, face[i], ctr[i], face[i+1], dt, KS.ps.dx[i], resL, avgL, coll)
            fn(KS, uq, face[j], ctr[j], face[j+1], dt, KS.ps.dx[j], resR, avgR, coll)
        end

        for i in eachindex(residual)
            residual[i] += sqrt((resL[i] + resR[i]) * 2) / (avgL[i] + avgR[i] + 1.e-7)
        end
    end

    ng = 1 - first(eachindex(KS.ps.x))
    if bc == :extra

        for i = 1:ng
            ctr[1-i].w .= ctr[1].w
            ctr[1-i].prim .= ctr[1].prim
            ctr[KS.ps.nx+i].w .= ctr[KS.ps.nx].w
            ctr[KS.ps.nx+i].prim .= ctr[KS.ps.nx].prim

            if KS.set.space[3:4] == "1f"
                ctr[1-i].f .= ctr[1].f
                ctr[KS.ps.nx+i].f .= ctr[KS.ps.nx].f
            elseif KS.set.space[3:4] == "2f"
                ctr[1-i].h .= ctr[1].h
                ctr[1-i].b .= ctr[1].b
                ctr[KS.ps.nx+i].h .= ctr[KS.ps.nx].h
                ctr[KS.ps.nx+i].b .= ctr[KS.ps.nx].b
            elseif KS.set.space[3:4] == "3f"
                ctr[1-i].h0 .= ctr[1].h0
                ctr[1-i].h1 .= ctr[1].h1
                ctr[1-i].h2 .= ctr[1].h2
                ctr[1-i].E .= ctr[1].E
                ctr[1-i].B .= ctr[1].B
                ctr[1-i].ϕ = deepcopy(ctr[1].ϕ)
                ctr[1-i].ψ = deepcopy(ctr[1].ψ)
                ctr[1-i].lorenz .= ctr[1].lorenz

                ctr[KS.ps.nx+i].h0 .= ctr[KS.ps.nx].h0
                ctr[KS.ps.nx+i].h1 .= ctr[KS.ps.nx].h1
                ctr[KS.ps.nx+i].h2 .= ctr[KS.ps.nx].h2
                ctr[KS.ps.nx+i].E .= ctr[KS.ps.nx].E
                ctr[KS.ps.nx+i].B .= ctr[KS.ps.nx].B
                ctr[KS.ps.nx+i].ϕ = deepcopy(ctr[KS.ps.nx].ϕ)
                ctr[KS.ps.nx+i].ψ = deepcopy(ctr[KS.ps.nx].ψ)
                ctr[KS.ps.nx+i].lorenz .= ctr[KS.ps.nx].lorenz
            elseif KS.set.space[3:4] == "4f"
                ctr[1-i].h0 .= ctr[1].h0
                ctr[1-i].h1 .= ctr[1].h1
                ctr[1-i].h2 .= ctr[1].h2
                ctr[1-i].h3 .= ctr[1].h3
                ctr[1-i].E .= ctr[1].E
                ctr[1-i].B .= ctr[1].B
                ctr[1-i].ϕ = deepcopy(ctr[1].ϕ)
                ctr[1-i].ψ = deepcopy(ctr[1].ψ)
                ctr[1-i].lorenz .= ctr[1].lorenz

                ctr[KS.ps.nx+i].h0 .= ctr[KS.ps.nx].h0
                ctr[KS.ps.nx+i].h1 .= ctr[KS.ps.nx].h1
                ctr[KS.ps.nx+i].h2 .= ctr[KS.ps.nx].h2
                ctr[KS.ps.nx+i].h3 .= ctr[KS.ps.nx].h3
                ctr[KS.ps.nx+i].E .= ctr[KS.ps.nx].E
                ctr[KS.ps.nx+i].B .= ctr[KS.ps.nx].B
                ctr[KS.ps.nx+i].ϕ = deepcopy(ctr[KS.ps.nx].ϕ)
                ctr[KS.ps.nx+i].ψ = deepcopy(ctr[KS.ps.nx].ψ)
                ctr[KS.ps.nx+i].lorenz .= ctr[KS.ps.nx].lorenz
            else
                throw("incorrect amount of distribution functions")
            end
        end

    elseif bc == :period

        for i = 1:ng
            ctr[1-i].w .= ctr[KS.ps.nx+1-i].w
            ctr[1-i].prim .= ctr[KS.ps.nx+1-i].prim
            ctr[KS.ps.nx+i].w .= ctr[i].w
            ctr[KS.ps.nx+i].prim .= ctr[i].prim

            ctr[1-i].h0 .= ctr[KS.ps.nx+1-i].h0
            ctr[1-i].h1 .= ctr[KS.ps.nx+1-i].h1
            ctr[1-i].h2 .= ctr[KS.ps.nx+1-i].h2
            ctr[1-i].E .= ctr[KS.ps.nx+1-i].E
            ctr[1-i].B .= ctr[KS.ps.nx+1-i].B
            ctr[1-i].ϕ = deepcopy(ctr[KS.ps.nx+1-i].ϕ)
            ctr[1-i].ψ = deepcopy(ctr[KS.ps.nx+1-i].ψ)
            ctr[1-i].lorenz .= ctr[KS.ps.nx+1-i].lorenz

            ctr[KS.ps.nx+i].h0 .= ctr[i].h0
            ctr[KS.ps.nx+i].h1 .= ctr[i].h1
            ctr[KS.ps.nx+i].h2 .= ctr[i].h2
            ctr[KS.ps.nx+i].E .= ctr[i].E
            ctr[KS.ps.nx+i].B .= ctr[i].B
            ctr[KS.ps.nx+i].ϕ = deepcopy(ctr[i].ϕ)
            ctr[KS.ps.nx+i].ψ = deepcopy(ctr[i].ψ)
            ctr[KS.ps.nx+i].lorenz .= ctr[i].lorenz

            if KS.set.space[3:4] == "1f"
                ctr[1-i].f .= ctr[KS.ps.nx+1-i].f
                ctr[KS.ps.nx+i].f .= ctr[i].f
            elseif KS.set.space[3:4] == "2f"
                ctr[1-i].h .= ctr[KS.ps.nx+1-i].h
                ctr[1-i].b .= ctr[KS.ps.nx+1-i].b
                ctr[KS.ps.nx+i].h .= ctr[i].h
                ctr[KS.ps.nx+i].b .= ctr[i].b
            elseif KS.set.space[3:4] == "3f"
                ctr[1-i].h0 .= ctr[KS.ps.nx+1-i].h0
                ctr[1-i].h1 .= ctr[KS.ps.nx+1-i].h1
                ctr[1-i].h2 .= ctr[KS.ps.nx+1-i].h2
                ctr[1-i].E .= ctr[KS.ps.nx+1-i].E
                ctr[1-i].B .= ctr[KS.ps.nx+1-i].B
                ctr[1-i].ϕ = deepcopy(ctr[KS.ps.nx+1-i].ϕ)
                ctr[1-i].ψ = deepcopy(ctr[KS.ps.nx+1-i].ψ)
                ctr[1-i].lorenz .= ctr[KS.ps.nx+1-i].lorenz

                ctr[KS.ps.nx+i].h0 .= ctr[i].h0
                ctr[KS.ps.nx+i].h1 .= ctr[i].h1
                ctr[KS.ps.nx+i].h2 .= ctr[i].h2
                ctr[KS.ps.nx+i].E .= ctr[i].E
                ctr[KS.ps.nx+i].B .= ctr[i].B
                ctr[KS.ps.nx+i].ϕ = deepcopy(ctr[i].ϕ)
                ctr[KS.ps.nx+i].ψ = deepcopy(ctr[i].ψ)
                ctr[KS.ps.nx+i].lorenz .= ctr[i].lorenz
            elseif KS.set.space[3:4] == "4f"
                ctr[1-i].h0 .= ctr[KS.ps.nx+1-i].h0
                ctr[1-i].h1 .= ctr[KS.ps.nx+1-i].h1
                ctr[1-i].h2 .= ctr[KS.ps.nx+1-i].h2
                ctr[1-i].h3 .= ctr[KS.ps.nx+1-i].h3
                ctr[1-i].E .= ctr[KS.ps.nx+1-i].E
                ctr[1-i].B .= ctr[KS.ps.nx+1-i].B
                ctr[1-i].ϕ = deepcopy(ctr[KS.ps.nx+1-i].ϕ)
                ctr[1-i].ψ = deepcopy(ctr[KS.ps.nx+1-i].ψ)
                ctr[1-i].lorenz .= ctr[KS.ps.nx+1-i].lorenz

                ctr[KS.ps.nx+i].h0 .= ctr[i].h0
                ctr[KS.ps.nx+i].h1 .= ctr[i].h1
                ctr[KS.ps.nx+i].h2 .= ctr[i].h2
                ctr[KS.ps.nx+i].h3 .= ctr[i].h3
                ctr[KS.ps.nx+i].E .= ctr[i].E
                ctr[KS.ps.nx+i].B .= ctr[i].B
                ctr[KS.ps.nx+i].ϕ = deepcopy(ctr[i].ϕ)
                ctr[KS.ps.nx+i].ψ = deepcopy(ctr[i].ψ)
                ctr[KS.ps.nx+i].lorenz .= ctr[i].lorenz
            else
                throw("incorrect amount of distribution functions")
            end
        end

    elseif bc == :balance

        @. ctr[0].w = 0.5 * (ctr[-1].w + ctr[1].w)
        @. ctr[0].prim = 0.5 * (ctr[-1].prim + ctr[1].prim)
        @. ctr[0].h0 = 0.5 * (ctr[-1].h0 + ctr[1].h0)
        @. ctr[0].h1 = 0.5 * (ctr[-1].h1 + ctr[1].h1)
        @. ctr[0].h2 = 0.5 * (ctr[-1].h2 + ctr[1].h2)
        @. ctr[0].E = 0.5 * (ctr[-1].E + ctr[1].E)
        @. ctr[0].B = 0.5 * (ctr[-1].B + ctr[1].B)
        ctr[0].ϕ = 0.5 * (ctr[-1].ϕ + ctr[1].ϕ)
        ctr[0].ψ = 0.5 * (ctr[-1].ψ + ctr[1].ψ)
        @. ctr[0].lorenz = 0.5 * (ctr[-1].lorenz + ctr[1].lorenz)

        @. ctr[KS.ps.nx+1].w = 0.5 * (ctr[KS.ps.nx].w + ctr[KS.ps.nx+2].w)
        @. ctr[KS.ps.nx+1].prim = 0.5 * (ctr[KS.ps.nx].prim + ctr[KS.ps.nx+2].prim)
        @. ctr[KS.ps.nx+1].h0 = 0.5 * (ctr[KS.ps.nx].h0 + ctr[KS.ps.nx+2].h0)
        @. ctr[KS.ps.nx+1].h1 = 0.5 * (ctr[KS.ps.nx].h1 + ctr[KS.ps.nx+2].h1)
        @. ctr[KS.ps.nx+1].h2 = 0.5 * (ctr[KS.ps.nx].h2 + ctr[KS.ps.nx+2].h2)
        @. ctr[KS.ps.nx+1].E = 0.5 * (ctr[KS.ps.nx].E + ctr[KS.ps.nx+2].E)
        @. ctr[KS.ps.nx+1].B = 0.5 * (ctr[KS.ps.nx].B + ctr[KS.ps.nx+2].B)
        ctr[KS.ps.nx+1].ϕ = 0.5 * (ctr[KS.ps.nx].ϕ + ctr[KS.ps.nx+2].ϕ)
        ctr[KS.ps.nx+1].ψ = 0.5 * (ctr[KS.ps.nx].ψ + ctr[KS.ps.nx+2].ψ)
        @. ctr[KS.ps.nx+1].lorenz = 0.5 * (ctr[KS.ps.nx].lorenz + ctr[KS.ps.nx+2].lorenz)

        if KS.set.space[3:4] == "1f"
            @. ctr[0].f = 0.5 * (ctr[-1].f + ctr[1].f)
            @. ctr[KS.ps.nx+1].f = 0.5 * (ctr[KS.ps.nx].f + ctr[KS.ps.nx+2].f)
        elseif KS.set.space[3:4] == "2f"
            @. ctr[0].h = 0.5 * (ctr[-1].h + ctr[1].h)
            @. ctr[0].b = 0.5 * (ctr[-1].b + ctr[1].b)
            @. ctr[KS.ps.nx+1].h = 0.5 * (ctr[KS.ps.nx].h + ctr[KS.ps.nx+2].h)
            @. ctr[KS.ps.nx+1].b = 0.5 * (ctr[KS.ps.nx].b + ctr[KS.ps.nx+2].b)
        elseif KS.set.space[3:4] == "3f"
            @. ctr[0].h0 = 0.5 * (ctr[-1].h0 + ctr[1].h0)
            @. ctr[0].h1 = 0.5 * (ctr[-1].h1 + ctr[1].h1)
            @. ctr[0].h2 = 0.5 * (ctr[-1].h2 + ctr[1].h2)
            @. ctr[0].E = 0.5 * (ctr[-1].E + ctr[1].E)
            @. ctr[0].B = 0.5 * (ctr[-1].B + ctr[1].B)
            ctr[0].ϕ = 0.5 * (ctr[-1].ϕ + ctr[1].ϕ)
            ctr[0].ψ = 0.5 * (ctr[-1].ψ + ctr[1].ψ)
            @. ctr[0].lorenz = 0.5 * (ctr[-1].lorenz + ctr[1].lorenz)

            @. ctr[KS.ps.nx+1].h0 = 0.5 * (ctr[KS.ps.nx].h0 + ctr[KS.ps.nx+2].h0)
            @. ctr[KS.ps.nx+1].h1 = 0.5 * (ctr[KS.ps.nx].h1 + ctr[KS.ps.nx+2].h1)
            @. ctr[KS.ps.nx+1].h2 = 0.5 * (ctr[KS.ps.nx].h2 + ctr[KS.ps.nx+2].h2)
            @. ctr[KS.ps.nx+1].E = 0.5 * (ctr[KS.ps.nx].E + ctr[KS.ps.nx+2].E)
            @. ctr[KS.ps.nx+1].B = 0.5 * (ctr[KS.ps.nx].B + ctr[KS.ps.nx+2].B)
            ctr[KS.ps.nx+1].ϕ = 0.5 * (ctr[KS.ps.nx].ϕ + ctr[KS.ps.nx+2].ϕ)
            ctr[KS.ps.nx+1].ψ = 0.5 * (ctr[KS.ps.nx].ψ + ctr[KS.ps.nx+2].ψ)
            @. ctr[KS.ps.nx+1].lorenz =
                0.5 * (ctr[KS.ps.nx].lorenz + ctr[KS.ps.nx+2].lorenz)
        elseif KS.set.space[3:4] == "4f"
            @. ctr[0].h0 = 0.5 * (ctr[-1].h0 + ctr[1].h0)
            @. ctr[0].h1 = 0.5 * (ctr[-1].h1 + ctr[1].h1)
            @. ctr[0].h2 = 0.5 * (ctr[-1].h2 + ctr[1].h2)
            @. ctr[0].h3 = 0.5 * (ctr[-1].h3 + ctr[1].h3)
            @. ctr[0].E = 0.5 * (ctr[-1].E + ctr[1].E)
            @. ctr[0].B = 0.5 * (ctr[-1].B + ctr[1].B)
            ctr[0].ϕ = 0.5 * (ctr[-1].ϕ + ctr[1].ϕ)
            ctr[0].ψ = 0.5 * (ctr[-1].ψ + ctr[1].ψ)
            @. ctr[0].lorenz = 0.5 * (ctr[-1].lorenz + ctr[1].lorenz)

            @. ctr[KS.ps.nx+1].h0 = 0.5 * (ctr[KS.ps.nx].h0 + ctr[KS.ps.nx+2].h0)
            @. ctr[KS.ps.nx+1].h1 = 0.5 * (ctr[KS.ps.nx].h1 + ctr[KS.ps.nx+2].h1)
            @. ctr[KS.ps.nx+1].h2 = 0.5 * (ctr[KS.ps.nx].h2 + ctr[KS.ps.nx+2].h2)
            @. ctr[KS.ps.nx+1].h3 = 0.5 * (ctr[KS.ps.nx].h3 + ctr[KS.ps.nx+2].h3)
            @. ctr[KS.ps.nx+1].E = 0.5 * (ctr[KS.ps.nx].E + ctr[KS.ps.nx+2].E)
            @. ctr[KS.ps.nx+1].B = 0.5 * (ctr[KS.ps.nx].B + ctr[KS.ps.nx+2].B)
            ctr[KS.ps.nx+1].ϕ = 0.5 * (ctr[KS.ps.nx].ϕ + ctr[KS.ps.nx+2].ϕ)
            ctr[KS.ps.nx+1].ψ = 0.5 * (ctr[KS.ps.nx].ψ + ctr[KS.ps.nx+2].ψ)
            @. ctr[KS.ps.nx+1].lorenz =
                0.5 * (ctr[KS.ps.nx].lorenz + ctr[KS.ps.nx+2].lorenz)
        else
            throw("incorrect amount of distribution functions")
        end

    end

end
