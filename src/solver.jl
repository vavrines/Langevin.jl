# ============================================================
# Module of Solver
# ============================================================


"""
Calculate time step

"""

function timestep(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::AbstractSolution,
    simTime::Real,
)

    tmax = 0.0

    if KS.set.nSpecies == 1

        if KS.set.space[1:2] == "1d"
            Threads.@threads for i = 1:KS.pSpace.nx
                sos = uq_sound_speed(sol.prim[i], KS.gas.γ, uq)
                vmax = KS.vSpace.u1 + maximum(sos)
                @inbounds tmax = max(tmax, vmax / KS.pSpace.dx[i])
            end
        elseif KS.set.space[1:2] == "2d"
            for j = 1:KS.pSpace.ny
                for i = 1:KS.pSpace.nx
                    sos = uq_sound_speed(sol.prim[i, j], KS.gas.γ, uq)
                    vmax = max(KS.vSpace.u1, KS.vSpace.v1) + maximum(sos)
                    @inbounds tmax = max(
                        tmax,
                        vmax / KS.pSpace.dx[i, j] + vmax / KS.pSpace.dy[i, j],
                    )
                end
            end
        end

    elseif KS.set.nSpecies == 2

        Threads.@threads for i = 1:KS.pSpace.nx
            @inbounds prim = sol.prim[i]
            sos = uq_sound_speed(prim, KS.gas.γ, uq)
            vmax =
                max(maximum(KS.vSpace.u1), maximum(abs.(prim[2, :, :]))) +
                maximum(sos)
            tmax = max(tmax, vmax / KS.pSpace.dx[i])
        end

    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt

end


function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution1D1F,
    flux::Flux1D1F,
    dt::Float64,
    residual::Array{Float64,1},
)

    w_old = deepcopy(sol.w)
    #=
        @. sol.w[2:KS.pSpace.nx-1] +=
            (flux.fw[2:end - 2] - flux.fw[3:end-1]) / KS.pSpace.dx[2:KS.pSpace.nx-1]
        uq_conserve_prim!(sol, KS.gas.γ, uq)
    =#
    for i = 1:KS.pSpace.nx
        @. sol.w[i] += (flux.fw[i] - flux.fw[i+1]) / KS.pSpace.dx[i]
        sol.prim[i] .= uq_conserve_prim(sol.w[i], KS.gas.γ, uq)
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    M = [
        uq_maxwellian(KS.vSpace.u, sol.prim[i], uq)
        for i in eachindex(sol.prim)
    ]

    for i = 1:KS.pSpace.nx
        for j in axes(sol.w[1], 2)
            @. sol.f[i][:, j] =
                (
                    sol.f[i][:, j] +
                    (flux.ff[i][:, j] - flux.ff[i+1][:, j]) / KS.pSpace.dx[i] +
                    dt / τ[i][j] * M[i][:, j]
                ) / (1.0 + dt / τ[i][j])
        end
    end

    # record residuals
    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))
    for j in axes(sumRes, 1)
        for i = 1:KS.pSpace.nx
            sumRes[j] += sum((sol.w[i][j, :] .- w_old[i][j, :]) .^ 2)
            sumAvg[j] += sum(abs.(sol.w[i][j, :]))
        end
    end
    @. residual = sqrt(sumRes * KS.pSpace.nx) / (sumAvg + 1.e-7)

end




function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution2D2F,
    flux::Flux2D2F,
    dt::Float64,
    residual::Array{Float64,1},
)

    w_old = deepcopy(sol.w)

    for j = 1:KS.pSpace.ny
        for i = 1:KS.pSpace.nx
            @. sol.w[i, j] +=
                (
                    flux.fw1[i, j] - flux.fw1[i+1, j] + flux.fw2[i, j] -
                    flux.fw2[i, j+1]
                ) / (KS.pSpace.dx[i, j] * KS.pSpace.dy[i, j])
            sol.prim[i, j] .= uq_conserve_prim(sol.w[i, j], KS.gas.γ, uq)
        end
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    H = [
        uq_maxwellian(KS.vSpace.u, KS.vSpace.v, sol.prim[i,j], uq)
        for i in axes(sol.prim, 1), j in axes(sol.prim, 2)
    ]
    B = deepcopy(H)
    for i in axes(sol.prim, 1), j in axes(sol.prim, 2)
        for k in axes(B[1, 1], 3)
            B[i, j][:, :, k] .=
                B[i, j][:, :, k] .* KS.gas.K ./ (2.0 * sol.prim[i, j][end, k])
        end
    end

    for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny
        for k in axes(sol.w[1, 1], 2)
            @. sol.h[i, j][:, :, k] =
                (
                    sol.h[i, j][:, :, k] +
                    (
                        flux.fh1[i, j][:, :, k] - flux.fh1[i+1, j][:, :, k] +
                        flux.fh2[i, j][:, :, k] - flux.fh2[i, j+1][:, :, k]
                    ) / (KS.pSpace.dx[i, j] * KS.pSpace.dy[i, j]) +
                    dt / τ[i, j][k] * H[i, j][:, :, k]
                ) / (1.0 + dt / τ[i, j][k])
            @. sol.b[i, j][:, :, k] =
                (
                    sol.b[i, j][:, :, k] +
                    (
                        flux.fb1[i, j][:, :, k] - flux.fb1[i+1, j][:, :, k] +
                        flux.fb2[i, j][:, :, k] - flux.fb2[i, j+1][:, :, k]
                    ) / (KS.pSpace.dx[i, j] * KS.pSpace.dy[i, j]) +
                    dt / τ[i, j][k] * B[i, j][:, :, k]
                ) / (1.0 + dt / τ[i, j][k])
        end
    end

    # record residuals
    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))
    for k in axes(sumRes, 1)
        for i = 1:KS.pSpace.nx, j = 1:KS.pSpace.ny
            sumRes[k] += sum((sol.w[i, j][k, :] .- w_old[i, j][k, :]) .^ 2)
            sumAvg[k] += sum(abs.(sol.w[i, j][k, :]))
        end
    end
    @. residual = sqrt(sumRes * KS.pSpace.nx * KS.pSpace.ny) / (sumAvg + 1.e-7)

end



function step!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution2D2F,
    flux::Flux2D2F,
    dt::Float64,
    residual::Array{Float64,1},
)

    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))

    Threads.@threads for j = 1:KS.pSpace.ny
        for i = 1:KS.pSpace.nx
            step!(
                KS,
                uq,
                sol.w[i, j],
                sol.prim[i, j],
                sol.h[i, j],
                sol.b[i, j],
                flux.fw1[i, j],
                flux.fh1[i, j],
                flux.fb1[i, j],
                flux.fw1[i+1, j],
                flux.fh1[i+1, j],
                flux.fb1[i+1, j],
                flux.fw2[i, j],
                flux.fh2[i, j],
                flux.fb2[i, j],
                flux.fw2[i, j+1],
                flux.fh2[i, j+1],
                flux.fb2[i, j+1],
                dt,
                KS.pSpace.dx[i, j] * KS.pSpace.dy[i, j],
                sumRes,
                sumAvg,
            )
        end
    end

    @. residual = sqrt(sumRes * KS.pSpace.nx * KS.pSpace.ny) / (sumAvg + 1.e-7)

end


function step!(
    KS::SolverSet,
    uq::AbstractUQ,
    w::Array{Float64,2},
    prim::Array{Float64,2},
    h::AbstractArray{Float64,3},
    b::AbstractArray{Float64,3},
    fwL::Array{Float64,2},
    fhL::AbstractArray{Float64,3},
    fbL::AbstractArray{Float64,3},
    fwR::Array{Float64,2},
    fhR::AbstractArray{Float64,3},
    fbR::AbstractArray{Float64,3},
    fwU::Array{Float64,2},
    fhU::AbstractArray{Float64,3},
    fbU::AbstractArray{Float64,3},
    fwD::Array{Float64,2},
    fhD::AbstractArray{Float64,3},
    fbD::AbstractArray{Float64,3},
    dt::Float64,
    area::Float64,
    sumRes::Array{Float64,1},
    sumAvg::Array{Float64,1},
)

    w_old = deepcopy(w)

    @. w += (fwL - fwR + fwD - fwU) / area
    prim .= uq_conserve_prim(w, KS.gas.γ, uq)

    τ = uq_vhs_collision_time(prim, KS.gas.μᵣ, KS.gas.ω, uq)
    H = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, prim, uq)
    B = similar(H)
    for k in axes(H, 3)
        B[:, :, k] .= H[:, :, k] .* KS.gas.K ./ (2.0 * prim[end, k])
    end

    for k in axes(h, 3)
        @. h[:, :, k] =
            (
                h[:, :, k] +
                (fhL[:, :, k] - fhR[:, :, k] + fhD[:, :, k] - fhU[:, :, k]) /
                area +
                dt / τ[k] * H[:, :, k]
            ) / (1.0 + dt / τ[k])
        @. b[:, :, k] =
            (
                b[:, :, k] +
                (fbL[:, :, k] - fbR[:, :, k] + fbD[:, :, k] - fbU[:, :, k]) /
                area +
                dt / τ[k] * B[:, :, k]
            ) / (1.0 + dt / τ[k])
    end

    for i = 1:4
        sumRes[i] += sum((w[i, :] .- w_old[i, :]) .^ 2)
        sumAvg[i] += sum(abs.(w[i, :]))
    end

end
