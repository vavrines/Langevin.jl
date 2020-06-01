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

        Threads.@threads for i = 1:KS.pSpace.nx
            @inbounds prim = sol.prim[i]
            sos = uq_sound_speed(prim, KS.gas.γ, uq)
            vmax = max(KS.vSpace.u1, maximum(abs.(prim[2, :]))) + maximum(sos)
            @inbounds tmax = max(tmax, vmax / KS.pSpace.dx[i])
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
    for i in 1:KS.pSpace.nx
        @. sol.w[i] += (flux.fw[i] - flux.fw[i+1]) / KS.pSpace.dx[i]
        for j in axes(sol.prim[1], 2)
            sol.prim[i] .= uq_conserve_prim(sol.w[i], KS.gas.γ, uq)
        end
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    M = [
        uq_maxwellian(KS.vSpace.u, sol.prim[i], uq)
        for i in eachindex(sol.prim)
    ]

    for i in 1:KS.pSpace.nx
        for j in axes(sol.w[1], 2)
            @. sol.f[i][:, j] = (
                sol.f[i][:, j] +
                (flux.ff[i][:, j] - flux.ff[i + 1][:, j]) /
                KS.pSpace.dx[i] +
                dt / τ[i][j] * M[i][:, j]
                ) / (1.0 + dt / τ[i][j])
        end
    end

    # record residuals
    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))
    for j in axes(sumRes, 1)
        for i in 1:KS.pSpace.nx
            sumRes[j] += sum((sol.w[i][j,:] .- w_old[i][j,:]).^2)
            sumAvg[j] += sum(abs.(sol.w[i][j,:]))
        end
    end
    @. residual = sqrt(sumRes * KS.pSpace.nx) / (sumAvg + 1.e-7)

end
