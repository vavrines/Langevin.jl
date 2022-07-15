# ============================================================
# Solution Algorithm
# ============================================================

"""
Calculate time step

"""
function timestep(KS, uq::AbstractUQ, sol::AbstractSolution, simTime)
    tmax = 0.0

    if KS.set.nSpecies == 1

        if KS.set.space[1:2] == "1d"
            @inbounds Threads.@threads for i = 1:KS.ps.nx
                sos = uq_sound_speed(sol.prim[i], KS.gas.γ, uq)
                vmax = KS.vs.u1 + maximum(sos)
                tmax = max(tmax, vmax / KS.ps.dx[i])
            end
        elseif KS.set.space[1:2] == "2d"
            @inbounds Threads.@threads for j = 1:KS.ps.ny
                for i = 1:KS.ps.nx
                    sos = uq_sound_speed(sol.prim[i, j], KS.gas.γ, uq)
                    vmax = max(KS.vs.u1, KS.vs.v1) + maximum(sos)
                    tmax = max(tmax, vmax / KS.ps.dx[i, j] + vmax / KS.ps.dy[i, j])
                end
            end
        end

    elseif KS.set.nSpecies == 2

        @inbounds Threads.@threads for i = 1:KS.ps.nx
            prim = sol.prim[i]
            sos = uq_sound_speed(prim, KS.gas.γ, uq)
            vmax = max(maximum(KS.vs.u1), maximum(abs.(prim[2, :, :]))) + maximum(sos)
            tmax = max(tmax, vmax / KS.ps.dx[i])
        end

    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt
end

function timestep(
    KS,
    uq::AbstractUQ,
    ctr::AbstractVector{T},
    simTime,
) where {T<:Union{ControlVolume1F,ControlVolume2F,ControlVolume1D1F,ControlVolume1D2F}}

    tmax = 0.0
    Threads.@threads for i = 1:KS.ps.nx
        @inbounds prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vs.u1), maximum(abs.(prim[2, :]))) + maximum(sos)
        tmax = max(tmax, vmax / KS.ps.dx[i])
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt

end

function timestep(
    KS::SolverSet,
    ctr::AbstractArray{ControlVolume1D4F,1},
    simTime::Real,
    uq::AbstractUQ,
)
    tmax = 0.0

    Threads.@threads for i = 1:KS.ps.nx
        @inbounds prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vs.u1), maximum(abs.(prim[2, :, :]))) + maximum(sos)
        tmax = max(tmax, vmax / KS.ps.dx[i], KS.gas.sol / KS.ps.dx[i])
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt
end

function timestep(
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractArray{ControlVolume1D3F,1},
    simTime::Real,
)
    tmax = 0.0

    @inbounds Threads.@threads for i = 1:KS.ps.nx
        prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vs.u1), maximum(abs.(prim[2, :, :]))) + maximum(sos)
        smax = maximum(abs.(ctr[i].lorenz))
        tmax = max(tmax, vmax / KS.ps.dx[i], KS.gas.sol / KS.ps.dx[i], smax / KS.vs.du[1])
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt
end


"""
Reconstruct solution

"""
function reconstruct!(KS::SolverSet, sol::Solution1F{T1,T2,T3,T4,1}) where {T1,T2,T3,T4}
    @inbounds Threads.@threads for i = 1:KS.ps.nx
        KitBase.reconstruct3!(
            sol.∇w[i],
            sol.w[i-1],
            sol.w[i],
            sol.w[i+1],
            0.5 * (KS.ps.dx[i-1] + KS.ps.dx[i]),
            0.5 * (KS.ps.dx[i] + KS.ps.dx[i+1]),
        )

        KitBase.reconstruct3!(
            sol.∇f[i],
            sol.f[i-1],
            sol.f[i],
            sol.f[i+1],
            0.5 * (KS.ps.dx[i-1] + KS.ps.dx[i]),
            0.5 * (KS.ps.dx[i] + KS.ps.dx[i+1]),
        )
    end
end

function reconstruct!(KS::SolverSet, sol::Solution2F{T1,T2,T3,T4,1}) where {T1,T2,T3,T4}
    @inbounds Threads.@threads for i = 1:KS.ps.nx
        KitBase.reconstruct3!(
            sol.∇w[i],
            sol.w[i-1],
            sol.w[i],
            sol.w[i+1],
            0.5 * (KS.ps.dx[i-1] + KS.ps.dx[i]),
            0.5 * (KS.ps.dx[i] + KS.ps.dx[i+1]),
        )

        KitBase.reconstruct3!(
            sol.∇h[i],
            sol.h[i-1],
            sol.h[i],
            sol.h[i+1],
            0.5 * (KS.ps.dx[i-1] + KS.ps.dx[i]),
            0.5 * (KS.ps.dx[i] + KS.ps.dx[i+1]),
        )

        KitBase.reconstruct3!(
            sol.∇b[i],
            sol.b[i-1],
            sol.b[i],
            sol.b[i+1],
            0.5 * (KS.ps.dx[i-1] + KS.ps.dx[i]),
            0.5 * (KS.ps.dx[i] + KS.ps.dx[i+1]),
        )
    end
end

function reconstruct!(KS::SolverSet, sol::Solution2F{T1,T2,T3,T4,2}) where {T1,T2,T3,T4}

    #--- x direction ---#
    @inbounds Threads.@threads for j = 1:KS.ps.ny
        KitBase.reconstruct2!(
            sol.sw[1, j][:, :, 1],
            sol.w[1, j],
            sol.w[2, j],
            0.5 * (KS.ps.dx[1, j] + KS.ps.dx[2, j]),
        )
        KitBase.reconstruct2!(
            sol.sh[1, j][:, :, :, 1],
            sol.h[1, j],
            sol.h[2, j],
            0.5 * (KS.ps.dx[1, j] + KS.ps.dx[2, j]),
        )
        KitBase.reconstruct2!(
            sol.sb[1, j][:, :, :, 1],
            sol.b[1, j],
            sol.b[2, j],
            0.5 * (KS.ps.dx[1, j] + KS.ps.dx[2, j]),
        )

        KitBase.reconstruct2!(
            sol.sw[KS.ps.nx, j][:, :, 1],
            sol.w[KS.ps.nx-1, j],
            sol.w[KS.ps.nx, j],
            0.5 * (KS.ps.dx[KS.ps.nx-1, j] + KS.ps.dx[KS.ps.nx, j]),
        )
        KitBase.reconstruct2!(
            sol.sh[KS.ps.nx, j][:, :, :, 1],
            sol.h[KS.ps.nx-1, j],
            sol.h[KS.ps.nx, j],
            0.5 * (KS.ps.dx[KS.ps.nx-1, j] + KS.ps.dx[KS.ps.nx, j]),
        )
        KitBase.reconstruct2!(
            sol.sb[KS.ps.nx, j][:, :, :, 1],
            sol.b[KS.ps.nx-1, j],
            sol.b[KS.ps.nx, j],
            0.5 * (KS.ps.dx[KS.ps.nx-1, j] + KS.ps.dx[KS.ps.nx, j]),
        )
    end

    @inbounds Threads.@threads for j = 1:KS.ps.ny
        for i = 2:KS.ps.nx-1
            KitBase.reconstruct3!(
                sol.sw[i, j][:, :, 1],
                sol.w[i-1, j],
                sol.w[i, j],
                sol.w[i+1, j],
                0.5 * (KS.ps.dx[i-1, j] + KS.ps.dx[i, j]),
                0.5 * (KS.ps.dx[i, j] + KS.ps.dx[i+1, j]),
            )

            KitBase.reconstruct3!(
                sol.sh[i, j][:, :, :, 1],
                sol.h[i-1, j],
                sol.h[i, j],
                sol.h[i+1, j],
                0.5 * (KS.ps.dx[i-1, j] + KS.ps.dx[i, j]),
                0.5 * (KS.ps.dx[i, j] + KS.ps.dx[i+1, j]),
            )
            KitBase.reconstruct3!(
                sol.sb[i, j][:, :, :, 1],
                sol.b[i-1, j],
                sol.b[i, j],
                sol.b[i+1, j],
                0.5 * (KS.ps.dx[i-1, j] + KS.ps.dx[i, j]),
                0.5 * (KS.ps.dx[i, j] + KS.ps.dx[i+1, j]),
            )
        end
    end

    #--- y direction ---#
    @inbounds Threads.@threads for i = 1:KS.ps.nx
        KitBase.reconstruct2!(
            sol.sw[i, 1][:, :, 2],
            sol.w[i, 1],
            sol.w[i, 2],
            0.5 * (KS.ps.dy[i, 1] + KS.ps.dy[i, 2]),
        )
        KitBase.reconstruct2!(
            sol.sh[i, 1][:, :, :, 2],
            sol.h[i, 1],
            sol.h[i, 2],
            0.5 * (KS.ps.dy[i, 1] + KS.ps.dy[i, 2]),
        )
        KitBase.reconstruct2!(
            sol.sb[i, 1][:, :, :, 2],
            sol.b[i, 1],
            sol.b[i, 2],
            0.5 * (KS.ps.dy[i, 1] + KS.ps.dy[i, 2]),
        )

        KitBase.reconstruct2!(
            sol.sw[i, KS.ps.ny][:, :, 2],
            sol.w[i, KS.ps.ny-1],
            sol.w[i, KS.ps.ny],
            0.5 * (KS.ps.dy[i, KS.ps.ny-1] + KS.ps.dy[i, KS.ps.ny]),
        )
        KitBase.reconstruct2!(
            sol.sh[i, KS.ps.ny][:, :, :, 2],
            sol.h[i, KS.ps.ny-1],
            sol.h[i, KS.ps.ny],
            0.5 * (KS.ps.dy[i, KS.ps.ny-1] + KS.ps.dy[i, KS.ps.ny]),
        )
        KitBase.reconstruct2!(
            sol.sb[i, KS.ps.ny][:, :, :, 2],
            sol.b[i, KS.ps.ny-1],
            sol.b[i, KS.ps.ny],
            0.5 * (KS.ps.dy[i, KS.ps.ny-1] + KS.ps.dy[i, KS.ps.ny]),
        )
    end

    @inbounds Threads.@threads for j = 2:KS.ps.ny-1
        for i = 1:KS.ps.nx
            KitBase.reconstruct3!(
                sol.sw[i, j][:, :, 2],
                sol.w[i, j-1],
                sol.w[i, j],
                sol.w[i, j+1],
                0.5 * (KS.ps.dy[i, j-1] + KS.ps.dy[i, j]),
                0.5 * (KS.ps.dy[i, j] + KS.ps.dy[i, j+1]),
            )

            KitBase.reconstruct3!(
                sol.sh[i, j][:, :, :, 2],
                sol.h[i, j-1],
                sol.h[i, j],
                sol.h[i, j+1],
                0.5 * (KS.ps.dy[i, j-1] + KS.ps.dy[i, j]),
                0.5 * (KS.ps.dy[i, j] + KS.ps.dy[i, j+1]),
            )
            KitBase.reconstruct3!(
                sol.sb[i, j][:, :, :, 2],
                sol.b[i, j-1],
                sol.b[i, j],
                sol.b[i, j+1],
                0.5 * (KS.ps.dy[i, j-1] + KS.ps.dy[i, j]),
                0.5 * (KS.ps.dy[i, j] + KS.ps.dy[i, j+1]),
            )
        end
    end

end


"""
Update solution

"""
function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution1F{T1,T2,T3,T4,1},
    flux::Flux1F,
    dt::Float64,
    residual::Array{Float64,1},
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
    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))
    for j in axes(sumRes, 1)
        for i = 1:KS.ps.nx
            sumRes[j] += sum((sol.w[i][j, :] .- w_old[i][j, :]) .^ 2)
            sumAvg[j] += sum(abs.(sol.w[i][j, :]))
        end
    end
    @. residual = sqrt(sumRes * KS.ps.nx) / (sumAvg + 1.e-7)

end


function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution2F{T1,T2,T3,T4,2},
    flux::Flux2F,
    dt::Float64,
    residual::Array{Float64,1},
) where {T1,T2,T3,T4}

    w_old = deepcopy(sol.w)

    @inbounds Threads.@threads for j = 1:KS.ps.ny
        for i = 1:KS.ps.nx
            @. sol.w[i, j] +=
                (flux.fw1[i, j] - flux.fw1[i+1, j] + flux.fw2[i, j] - flux.fw2[i, j+1]) /
                (KS.ps.dx[i, j] * KS.ps.dy[i, j])
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

    @inbounds Threads.@threads for i = 1:KS.ps.nx
        for j = 1:KS.ps.ny
            for k in axes(sol.w[1, 1], 2)
                @. sol.h[i, j][:, :, k] =
                    (
                        sol.h[i, j][:, :, k] +
                        (
                            flux.fh1[i, j][:, :, k] - flux.fh1[i+1, j][:, :, k] +
                            flux.fh2[i, j][:, :, k] - flux.fh2[i, j+1][:, :, k]
                        ) / (KS.ps.dx[i, j] * KS.ps.dy[i, j]) +
                        dt / τ[i, j][k] * H[i, j][:, :, k]
                    ) / (1.0 + dt / τ[i, j][k])
                @. sol.b[i, j][:, :, k] =
                    (
                        sol.b[i, j][:, :, k] +
                        (
                            flux.fb1[i, j][:, :, k] - flux.fb1[i+1, j][:, :, k] +
                            flux.fb2[i, j][:, :, k] - flux.fb2[i, j+1][:, :, k]
                        ) / (KS.ps.dx[i, j] * KS.ps.dy[i, j]) +
                        dt / τ[i, j][k] * B[i, j][:, :, k]
                    ) / (1.0 + dt / τ[i, j][k])
            end
        end
    end

    # record residuals
    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))
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
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractVector{T},
    face,
    dt,
    residual;
    coll = :bgk,
    bc = :fix,
) where {T<:AbstractControlVolume}

    sumRes = zeros(3)
    sumAvg = zeros(3)

    @inbounds Threads.@threads for i = 2:KS.ps.nx-1
        step!(KS, uq, face[i], ctr[i], face[i+1], dt, KS.ps.dx[i], sumRes, sumAvg, coll)
    end

    for i in axes(residual, 1)
        residual[i] = sqrt(sumRes[i] * KS.ps.nx) / (sumAvg[i] + 1.e-7)
    end

    update_boundary!(KS, uq, ctr, face, dt, residual; coll = coll, bc = bc, isMHD = false)

end

function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractVector{ControlVolume1D3F},
    face::AbstractVector{Interface1D3F},
    dt::Real,
    residual::Array{<:Real,2};
    coll = :bgk::Symbol,
    bc = :extra::Symbol,
    isMHD = true::Bool,
)

    sumRes = zeros(5, 2)
    sumAvg = zeros(5, 2)

    @inbounds Threads.@threads for i = 2:KS.ps.nx-1
        step!(KS, uq, face[i], ctr[i], face[i+1], dt, sumRes, sumAvg, coll, isMHD)
    end

    for i in axes(residual, 1)
        @. residual[i, :] = sqrt(sumRes[i, :] * KS.ps.nx) / (sumAvg[i, :] + 1.e-7)
    end

    update_boundary!(KS, uq, ctr, face, dt, residual; coll = coll, bc = bc, isMHD = isMHD)

end

function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractVector{ControlVolume1D4F},
    face::AbstractVector{Interface1D4F},
    dt::Real,
    residual::Array{<:Real,2};
    coll = :bgk::Symbol,
    bc = :extra::Symbol,
    isMHD = true::Bool,
)

    sumRes = zeros(5, 2)
    sumAvg = zeros(5, 2)

    @inbounds Threads.@threads for i = 2:KS.ps.nx-1
        step!(KS, uq, face[i], ctr[i], face[i+1], dt, sumRes, sumAvg)
    end

    for i in axes(residual, 1)
        @. residual[i, :] = sqrt(sumRes[i, :] * KS.ps.nx) / (sumAvg[i, :] + 1.e-7)
    end

    update_boundary!(KS, uq, ctr, face, dt, residual; coll = coll, bc = bc, isMHD = isMHD)

end


function update_boundary!(
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractVector,
    face::AbstractVector,
    dt,
    residual;
    coll = :bgk,
    bc = :extra,
    isMHD = true,
)

    if bc != :fix
        resL = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)
        avgL = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)
        resR = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)
        avgR = zeros(size(KS.ib.wL, 1), KS.set.nSpecies)

        i = 1
        j = KS.ps.nx
        if KS.set.space[3:4] == "0f"
            step!(KS, uq, face[i], ctr[i], face[i+1], dt, KS.ps.dx[i], resL, avgL)
            step!(KS, uq, face[j], ctr[j], face[j+1], dt, KS.ps.dx[j], resR, avgR)
        elseif KS.set.space[3:4] in ["3f", "4f"]
            step!(KS, uq, face[i], ctr[i], face[i+1], dt, resL, avgL, coll, isMHD)
            step!(KS, uq, face[j], ctr[j], face[j+1], dt, resR, avgR, coll, isMHD)
        else
            step!(KS, uq, face[i], ctr[i], face[i+1], dt, KS.ps.dx[i], resL, avgL, coll)
            step!(KS, uq, face[j], ctr[j], face[j+1], dt, KS.ps.dx[j], resR, avgR, coll)
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

    else

    end

end
