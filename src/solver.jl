# ============================================================
# Module of Solver
# ============================================================


"""
Calculate time step

"""
function timestep(KS::SolverSet, uq::AbstractUQ, sol::AbstractSolution, simTime::Real)

    tmax = 0.0

    if KS.set.nSpecies == 1

        if KS.set.space[1:2] == "1d"
            @inbounds Threads.@threads for i = 1:KS.pSpace.nx
                sos = uq_sound_speed(sol.prim[i], KS.gas.γ, uq)
                vmax = KS.vSpace.u1 + maximum(sos)
                tmax = max(tmax, vmax / KS.pSpace.dx[i])
            end
        elseif KS.set.space[1:2] == "2d"
            @inbounds Threads.@threads for j = 1:KS.pSpace.ny
                for i = 1:KS.pSpace.nx
                    sos = uq_sound_speed(sol.prim[i, j], KS.gas.γ, uq)
                    vmax = max(KS.vSpace.u1, KS.vSpace.v1) + maximum(sos)
                    tmax = max(tmax, vmax / KS.pSpace.dx[i, j] + vmax / KS.pSpace.dy[i, j])
                end
            end
        end

    elseif KS.set.nSpecies == 2

        @inbounds Threads.@threads for i = 1:KS.pSpace.nx
            prim = sol.prim[i]
            sos = uq_sound_speed(prim, KS.gas.γ, uq)
            vmax = max(maximum(KS.vSpace.u1), maximum(abs.(prim[2, :, :]))) + maximum(sos)
            tmax = max(tmax, vmax / KS.pSpace.dx[i])
        end

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

    Threads.@threads for i = 1:KS.pSpace.nx
        @inbounds prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vSpace.u1), maximum(abs.(prim[2, :, :]))) + maximum(sos)
        tmax = max(tmax, vmax / ctr[i].dx, KS.gas.sol / ctr[i].dx)
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt
end

function timestep(
    KS::SolverSet,
    ctr::AbstractArray{ControlVolume1D3F,1},
    simTime::Real,
    uq::AbstractUQ,
)
    tmax = 0.0

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vSpace.u1), maximum(abs.(prim[2, :, :]))) + maximum(sos)
        smax = maximum(abs.(ctr[i].lorenz))
        tmax = max(tmax, vmax / ctr[i].dx, KS.gas.sol / ctr[i].dx, smax / KS.vSpace.du[1])
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt
end


"""
Reconstruct solution

"""
function reconstruct!(KS::SolverSet, sol::Solution1D1F)

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        Kinetic.reconstruct3!(
            sol.sw[i],
            sol.w[i-1],
            sol.w[i],
            sol.w[i+1],
            0.5 * (KS.pSpace.dx[i-1] + KS.pSpace.dx[i]),
            0.5 * (KS.pSpace.dx[i] + KS.pSpace.dx[i+1]),
        )

        Kinetic.reconstruct3!(
            sol.sf[i],
            sol.f[i-1],
            sol.f[i],
            sol.f[i+1],
            0.5 * (KS.pSpace.dx[i-1] + KS.pSpace.dx[i]),
            0.5 * (KS.pSpace.dx[i] + KS.pSpace.dx[i+1]),
        )
    end

end

function reconstruct!(KS::SolverSet, sol::Solution2D2F)

    #--- x direction ---#
    @inbounds Threads.@threads for j = 1:KS.pSpace.ny
        Kinetic.reconstruct2!(
            sol.sw[1, j][:, :, 1],
            sol.w[1, j],
            sol.w[2, j],
            0.5 * (KS.pSpace.dx[1, j] + KS.pSpace.dx[2, j]),
        )
        Kinetic.reconstruct2!(
            sol.sh[1, j][:, :, :, 1],
            sol.h[1, j],
            sol.h[2, j],
            0.5 * (KS.pSpace.dx[1, j] + KS.pSpace.dx[2, j]),
        )
        Kinetic.reconstruct2!(
            sol.sb[1, j][:, :, :, 1],
            sol.b[1, j],
            sol.b[2, j],
            0.5 * (KS.pSpace.dx[1, j] + KS.pSpace.dx[2, j]),
        )

        Kinetic.reconstruct2!(
            sol.sw[KS.pSpace.nx, j][:, :, 1],
            sol.w[KS.pSpace.nx-1, j],
            sol.w[KS.pSpace.nx, j],
            0.5 * (KS.pSpace.dx[KS.pSpace.nx-1, j] + KS.pSpace.dx[KS.pSpace.nx, j]),
        )
        Kinetic.reconstruct2!(
            sol.sh[KS.pSpace.nx, j][:, :, :, 1],
            sol.h[KS.pSpace.nx-1, j],
            sol.h[KS.pSpace.nx, j],
            0.5 * (KS.pSpace.dx[KS.pSpace.nx-1, j] + KS.pSpace.dx[KS.pSpace.nx, j]),
        )
        Kinetic.reconstruct2!(
            sol.sb[KS.pSpace.nx, j][:, :, :, 1],
            sol.b[KS.pSpace.nx-1, j],
            sol.b[KS.pSpace.nx, j],
            0.5 * (KS.pSpace.dx[KS.pSpace.nx-1, j] + KS.pSpace.dx[KS.pSpace.nx, j]),
        )
    end

    @inbounds Threads.@threads for j = 1:KS.pSpace.ny
        for i = 2:KS.pSpace.nx-1
            Kinetic.reconstruct3!(
                sol.sw[i, j][:, :, 1],
                sol.w[i-1, j],
                sol.w[i, j],
                sol.w[i+1, j],
                0.5 * (KS.pSpace.dx[i-1, j] + KS.pSpace.dx[i, j]),
                0.5 * (KS.pSpace.dx[i, j] + KS.pSpace.dx[i+1, j]),
            )

            Kinetic.reconstruct3!(
                sol.sh[i, j][:, :, :, 1],
                sol.h[i-1, j],
                sol.h[i, j],
                sol.h[i+1, j],
                0.5 * (KS.pSpace.dx[i-1, j] + KS.pSpace.dx[i, j]),
                0.5 * (KS.pSpace.dx[i, j] + KS.pSpace.dx[i+1, j]),
            )
            Kinetic.reconstruct3!(
                sol.sb[i, j][:, :, :, 1],
                sol.b[i-1, j],
                sol.b[i, j],
                sol.b[i+1, j],
                0.5 * (KS.pSpace.dx[i-1, j] + KS.pSpace.dx[i, j]),
                0.5 * (KS.pSpace.dx[i, j] + KS.pSpace.dx[i+1, j]),
            )
        end
    end

    #--- y direction ---#
    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        Kinetic.reconstruct2!(
            sol.sw[i, 1][:, :, 2],
            sol.w[i, 1],
            sol.w[i, 2],
            0.5 * (KS.pSpace.dy[i, 1] + KS.pSpace.dy[i, 2]),
        )
        Kinetic.reconstruct2!(
            sol.sh[i, 1][:, :, :, 2],
            sol.h[i, 1],
            sol.h[i, 2],
            0.5 * (KS.pSpace.dy[i, 1] + KS.pSpace.dy[i, 2]),
        )
        Kinetic.reconstruct2!(
            sol.sb[i, 1][:, :, :, 2],
            sol.b[i, 1],
            sol.b[i, 2],
            0.5 * (KS.pSpace.dy[i, 1] + KS.pSpace.dy[i, 2]),
        )

        Kinetic.reconstruct2!(
            sol.sw[i, KS.pSpace.ny][:, :, 2],
            sol.w[i, KS.pSpace.ny-1],
            sol.w[i, KS.pSpace.ny],
            0.5 * (KS.pSpace.dy[i, KS.pSpace.ny-1] + KS.pSpace.dy[i, KS.pSpace.ny]),
        )
        Kinetic.reconstruct2!(
            sol.sh[i, KS.pSpace.ny][:, :, :, 2],
            sol.h[i, KS.pSpace.ny-1],
            sol.h[i, KS.pSpace.ny],
            0.5 * (KS.pSpace.dy[i, KS.pSpace.ny-1] + KS.pSpace.dy[i, KS.pSpace.ny]),
        )
        Kinetic.reconstruct2!(
            sol.sb[i, KS.pSpace.ny][:, :, :, 2],
            sol.b[i, KS.pSpace.ny-1],
            sol.b[i, KS.pSpace.ny],
            0.5 * (KS.pSpace.dy[i, KS.pSpace.ny-1] + KS.pSpace.dy[i, KS.pSpace.ny]),
        )
    end

    @inbounds Threads.@threads for j = 2:KS.pSpace.ny-1
        for i = 1:KS.pSpace.nx
            Kinetic.reconstruct3!(
                sol.sw[i, j][:, :, 2],
                sol.w[i, j-1],
                sol.w[i, j],
                sol.w[i, j+1],
                0.5 * (KS.pSpace.dy[i, j-1] + KS.pSpace.dy[i, j]),
                0.5 * (KS.pSpace.dy[i, j] + KS.pSpace.dy[i, j+1]),
            )

            Kinetic.reconstruct3!(
                sol.sh[i, j][:, :, :, 2],
                sol.h[i, j-1],
                sol.h[i, j],
                sol.h[i, j+1],
                0.5 * (KS.pSpace.dy[i, j-1] + KS.pSpace.dy[i, j]),
                0.5 * (KS.pSpace.dy[i, j] + KS.pSpace.dy[i, j+1]),
            )
            Kinetic.reconstruct3!(
                sol.sb[i, j][:, :, :, 2],
                sol.b[i, j-1],
                sol.b[i, j],
                sol.b[i, j+1],
                0.5 * (KS.pSpace.dy[i, j-1] + KS.pSpace.dy[i, j]),
                0.5 * (KS.pSpace.dy[i, j] + KS.pSpace.dy[i, j+1]),
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
    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        @. sol.w[i] += (flux.fw[i] - flux.fw[i+1]) / KS.pSpace.dx[i]
        sol.prim[i] .= uq_conserve_prim(sol.w[i], KS.gas.γ, uq)
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    M = [uq_maxwellian(KS.vSpace.u, sol.prim[i], uq) for i in eachindex(sol.prim)]

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
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

    @inbounds Threads.@threads for j = 1:KS.pSpace.ny
        for i = 1:KS.pSpace.nx
            @. sol.w[i, j] +=
                (flux.fw1[i, j] - flux.fw1[i+1, j] + flux.fw2[i, j] - flux.fw2[i, j+1]) /
                (KS.pSpace.dx[i, j] * KS.pSpace.dy[i, j])
            sol.prim[i, j] .= uq_conserve_prim(sol.w[i, j], KS.gas.γ, uq)
        end
    end

    τ = uq_vhs_collision_time(sol, KS.gas.μᵣ, KS.gas.ω, uq)
    H = [
        uq_maxwellian(KS.vSpace.u, KS.vSpace.v, sol.prim[i, j], uq)
        for i in axes(sol.prim, 1), j in axes(sol.prim, 2)
    ]
    B = deepcopy(H)
    for i in axes(sol.prim, 1), j in axes(sol.prim, 2)
        for k in axes(B[1, 1], 3)
            B[i, j][:, :, k] .=
                B[i, j][:, :, k] .* KS.gas.K ./ (2.0 * sol.prim[i, j][end, k])
        end
    end

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        for j = 1:KS.pSpace.ny
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
    end

    # record residuals
    sumRes = zeros(axes(KS.ib.wL, 1))
    sumAvg = zeros(axes(KS.ib.wL, 1))
    @inbounds for k in axes(sumRes, 1)
        for j = 1:KS.pSpace.ny
            for i = 1:KS.pSpace.nx
                sumRes[k] += sum((sol.w[i, j][k, :] .- w_old[i, j][k, :]) .^ 2)
                sumAvg[k] += sum(abs.(sol.w[i, j][k, :]))
            end
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
                (fhL[:, :, k] - fhR[:, :, k] + fhD[:, :, k] - fhU[:, :, k]) / area +
                dt / τ[k] * H[:, :, k]
            ) / (1.0 + dt / τ[k])
        @. b[:, :, k] =
            (
                b[:, :, k] +
                (fbL[:, :, k] - fbR[:, :, k] + fbD[:, :, k] - fbU[:, :, k]) / area +
                dt / τ[k] * B[:, :, k]
            ) / (1.0 + dt / τ[k])
    end

    for i = 1:4
        sumRes[i] += sum((w[i, :] .- w_old[i, :]) .^ 2)
        sumAvg[i] += sum(abs.(w[i, :]))
    end

end

function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractArray{ControlVolume1D3F,1},
    face::AbstractArray{Interface1D3F,1},
    dt::Real,
    residual::Array{<:Real,2};
    bc = :extra::Symbol,
)

    sumRes = zeros(5, 2)
    sumAvg = zeros(5, 2)

    Threads.@threads for i = 1:KS.pSpace.nx
        @inbounds step!(KS, uq, face[i], ctr[i], face[i+1], dt, sumRes, sumAvg)
    end

    for i in axes(residual, 1)
        @. residual[i, :] = sqrt(sumRes[i, :] * KS.pSpace.nx) / (sumAvg[i, :] + 1.e-7)
    end

    #--- periodic boundary ---#
    #ctr[0].w = deepcopy(ctr[KS.pSpace.nx].w); ctr[0].prim = deepcopy(ctr[KS.pSpace.nx].prim)
    #ctr[0].h0 = deepcopy(ctr[KS.pSpace.nx].h0); 
    #ctr[0].h1 = deepcopy(ctr[KS.pSpace.nx].h1)
    #ctr[0].h2 = deepcopy(ctr[KS.pSpace.nx].h2); 
    #ctr[0].E = deepcopy(ctr[KS.pSpace.nx].E); ctr[0].B = deepcopy(ctr[KS.pSpace.nx].B)
    #ctr[0].ϕ = deepcopy(ctr[KS.pSpace.nx].ϕ); ctr[0].ψ = deepcopy(ctr[KS.pSpace.nx].ψ)
    #ctr[0].lorenz = deepcopy(ctr[KS.pSpace.nx].lorenz)

    #ctr[KS.pSpace.nx+1].w = deepcopy(ctr[1].w); ctr[KS.pSpace.nx+1].prim = deepcopy(ctr[1].prim)
    #ctr[KS.pSpace.nx+1].h0 = deepcopy(ctr[1].h0); 
    #ctr[KS.pSpace.nx+1].h1 = deepcopy(ctr[1].h1)
    #ctr[KS.pSpace.nx+1].h2 = deepcopy(ctr[1].h2); 
    #ctr[KS.pSpace.nx+1].E = deepcopy(ctr[1].E); ctr[KS.pSpace.nx+1].B = deepcopy(ctr[1].B)
    #ctr[KS.pSpace.nx+1].ϕ = deepcopy(ctr[1].ϕ); ctr[KS.pSpace.nx+1].ψ = deepcopy(ctr[1].ψ)
    #ctr[KS.pSpace.nx+1].lorenz = deepcopy(ctr[1].lorenz)

    #--- extrapolation boundary ---#
    for i = 1:2
        ctr[1-i].w .= ctr[1].w
        ctr[1-i].prim .= ctr[1].prim
        ctr[1-i].h0 .= ctr[1].h0
        ctr[1-i].h1 .= ctr[1].h1
        ctr[1-i].h2 .= ctr[1].h2
        ctr[1-i].E .= ctr[1].E
        ctr[1-i].B .= ctr[1].B
        ctr[1-i].ϕ .= ctr[1].ϕ
        ctr[1-i].ψ .= ctr[1].ψ
        ctr[1-i].lorenz .= ctr[1].lorenz

        ctr[KS.pSpace.nx+i].w .= ctr[KS.pSpace.nx].w
        ctr[KS.pSpace.nx+i].prim .= ctr[KS.pSpace.nx].prim
        ctr[KS.pSpace.nx+i].h0 .= ctr[KS.pSpace.nx].h0
        ctr[KS.pSpace.nx+i].h1 .= ctr[KS.pSpace.nx].h1
        ctr[KS.pSpace.nx+i].h2 .= ctr[KS.pSpace.nx].h2
        ctr[KS.pSpace.nx+i].E .= ctr[KS.pSpace.nx].E
        ctr[KS.pSpace.nx+i].B .= ctr[KS.pSpace.nx].B
        ctr[KS.pSpace.nx+i].ϕ .= ctr[KS.pSpace.nx].ϕ
        ctr[KS.pSpace.nx+i].ψ .= ctr[KS.pSpace.nx].ψ
        ctr[KS.pSpace.nx+i].lorenz .= ctr[KS.pSpace.nx].lorenz
    end

    #--- balance boundary ---#
    #=
    ctr[0].w .= 0.5 * (ctr[-1].w .+ ctr[1].w); ctr[0].prim .= 0.5 * (ctr[-1].prim .+ ctr[1].prim)
    ctr[0].h0 .= 0.5 * (ctr[-1].h0 .+ ctr[1].h0); ctr[0].h1 .= 0.5 * (ctr[-1].h1 .+ ctr[1].h1)
    ctr[0].h2 .= 0.5 * (ctr[-1].h2 .+ ctr[1].h2); ctr[0].h3 .= 0.5 * (ctr[-1].h3 .+ ctr[1].h3)
    ctr[0].E .= 0.5 * (ctr[-1].E .+ ctr[1].E); ctr[0].B .= 0.5 * (ctr[-1].B .+ ctr[1].B)
    ctr[0].ϕ .= 0.5 * (ctr[-1].ϕ .+ ctr[1].ϕ); ctr[0].ψ .= 0.5 * (ctr[-1].ψ .+ ctr[1].ψ)
    ctr[0].lorenz .= 0.5 * (ctr[-1].lorenz .+ ctr[1].lorenz)
    =#
end

function update!(
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractArray{ControlVolume1D4F,1},
    face::AbstractArray{Interface1D4F,1},
    dt::Real,
    residual::Array{<:Real,2},
)

    sumRes = zeros(5, 2)
    sumAvg = zeros(5, 2)

    @inbounds Threads.@threads for i = 1:KS.pSpace.nx
        step!(KS, uq, face[i], ctr[i], face[i+1], dt, sumRes, sumAvg)
    end

    for i in axes(residual, 1)
        @. residual[i, :] = sqrt(sumRes[i, :] * KS.pSpace.nx) / (sumAvg[i, :] + 1.e-7)
    end

    #--- periodic boundary ---#
    #ctr[0].w = deepcopy(ctr[KS.pSpace.nx].w); ctr[0].prim = deepcopy(ctr[KS.pSpace.nx].prim)
    #ctr[0].h0 = deepcopy(ctr[KS.pSpace.nx].h0); ctr[0].h1 = deepcopy(ctr[KS.pSpace.nx].h1)
    #ctr[0].h2 = deepcopy(ctr[KS.pSpace.nx].h2); ctr[0].h3 = deepcopy(ctr[KS.pSpace.nx].h3)
    #ctr[0].E = deepcopy(ctr[KS.pSpace.nx].E); ctr[0].B = deepcopy(ctr[KS.pSpace.nx].B)
    #ctr[0].ϕ = deepcopy(ctr[KS.pSpace.nx].ϕ); ctr[0].ψ = deepcopy(ctr[KS.pSpace.nx].ψ)
    #ctr[0].lorenz = deepcopy(ctr[KS.pSpace.nx].lorenz)

    #ctr[KS.pSpace.nx+1].w = deepcopy(ctr[1].w); ctr[KS.pSpace.nx+1].prim = deepcopy(ctr[1].prim)
    #ctr[KS.pSpace.nx+1].h0 = deepcopy(ctr[1].h0); ctr[KS.pSpace.nx+1].h1 = deepcopy(ctr[1].h1)
    #ctr[KS.pSpace.nx+1].h2 = deepcopy(ctr[1].h2); ctr[KS.pSpace.nx+1].h3 = deepcopy(ctr[1].h3)
    #ctr[KS.pSpace.nx+1].E = deepcopy(ctr[1].E); ctr[KS.pSpace.nx+1].B = deepcopy(ctr[1].B)
    #ctr[KS.pSpace.nx+1].ϕ = deepcopy(ctr[1].ϕ); ctr[KS.pSpace.nx+1].ψ = deepcopy(ctr[1].ψ)
    #ctr[KS.pSpace.nx+1].lorenz = deepcopy(ctr[1].lorenz)

    #--- extrapolation boundary ---#
    for i = 1:2
        ctr[1-i].w .= ctr[1].w
        ctr[1-i].prim .= ctr[1].prim
        ctr[1-i].h0 .= ctr[1].h0
        ctr[1-i].h1 .= ctr[1].h1
        ctr[1-i].h2 .= ctr[1].h2
        ctr[1-i].h3 .= ctr[1].h3
        ctr[1-i].E .= ctr[1].E
        ctr[1-i].B .= ctr[1].B
        ctr[1-i].ϕ .= ctr[1].ϕ
        ctr[1-i].ψ .= ctr[1].ψ
        ctr[1-i].lorenz .= ctr[1].lorenz

        ctr[KS.pSpace.nx+i].w .= ctr[KS.pSpace.nx].w
        ctr[KS.pSpace.nx+i].prim .= ctr[KS.pSpace.nx].prim
        ctr[KS.pSpace.nx+i].h0 .= ctr[KS.pSpace.nx].h0
        ctr[KS.pSpace.nx+i].h1 .= ctr[KS.pSpace.nx].h1
        ctr[KS.pSpace.nx+i].h2 .= ctr[KS.pSpace.nx].h2
        ctr[KS.pSpace.nx+i].h3 .= ctr[KS.pSpace.nx].h3
        ctr[KS.pSpace.nx+i].E .= ctr[KS.pSpace.nx].E
        ctr[KS.pSpace.nx+i].B .= ctr[KS.pSpace.nx].B
        ctr[KS.pSpace.nx+i].ϕ .= ctr[KS.pSpace.nx].ϕ
        ctr[KS.pSpace.nx+i].ψ .= ctr[KS.pSpace.nx].ψ
        ctr[KS.pSpace.nx+i].lorenz .= ctr[KS.pSpace.nx].lorenz
    end

    #--- balance boundary ---#
    #=
    ctr[0].w .= 0.5 * (ctr[-1].w .+ ctr[1].w); ctr[0].prim .= 0.5 * (ctr[-1].prim .+ ctr[1].prim)
    ctr[0].h0 .= 0.5 * (ctr[-1].h0 .+ ctr[1].h0); ctr[0].h1 .= 0.5 * (ctr[-1].h1 .+ ctr[1].h1)
    ctr[0].h2 .= 0.5 * (ctr[-1].h2 .+ ctr[1].h2); ctr[0].h3 .= 0.5 * (ctr[-1].h3 .+ ctr[1].h3)
    ctr[0].E .= 0.5 * (ctr[-1].E .+ ctr[1].E); ctr[0].B .= 0.5 * (ctr[-1].B .+ ctr[1].B)
    ctr[0].ϕ .= 0.5 * (ctr[-1].ϕ .+ ctr[1].ϕ); ctr[0].ψ .= 0.5 * (ctr[-1].ψ .+ ctr[1].ψ)
    ctr[0].lorenz .= 0.5 * (ctr[-1].lorenz .+ ctr[1].lorenz)
    =#
end



function step!(
    KS::SolverSet,
    uq::AbstractUQ,
    faceL::Interface1D4F,
    cell::ControlVolume1D4F,
    faceR::Interface1D4F,
    dt::AbstractFloat,
    RES::Array{<:AbstractFloat,2},
    AVG::Array{<:AbstractFloat,2},
)

    if uq.method == "galerkin"

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        #@. cell.w += (faceL.fw - faceR.fw) / cell.dx
        #cell.prim .= get_primitive(cell.w, KS.gas.γ, uq)

        # locate variables on random quadrature points
        #wRan = get_ran_array(cell.w, 2, uq)
        #primRan = get_ran_array(cell.prim, 2, uq)

        wRan =
            chaos_ran(cell.w, 2, uq) .+
            (chaos_ran(faceL.fw, 2, uq) .- chaos_ran(faceR.fw, 2, uq)) ./ cell.dx
        primRan = uq_conserve_prim(wRan, KS.gas.γ, uq)

        # temperature protection
        if min(minimum(primRan[5, :, 1]), minimum(primRan[5, :, 2])) < 0
            println("warning: temperature update is negative")
            wRan = chaos_ran(w_old, 2, uq)
            primRan = chaos_ran(prim_old, 2, uq)
        end

        #=
        # source -> w^{n+1}
        # DifferentialEquations.jl
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        for j in axes(wRan, 2)
        prob = ODEProblem( mixture_source, 
                    vcat(wRan[1:5,j,1], wRan[1:5,j,2]),
                    dt,
                    (tau[1], tau[2], KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], KS.gas.γ) )
        sol = solve(prob, Rosenbrock23())

        wRan[1:5,j,1] .= sol[end][1:5]
        wRan[1:5,j,2] .= sol[end][6:10]
        for k=1:2
        primRan[:,j,k] .= Kinetic.conserve_prim(wRan[:,j,k], KS.gas.γ)
        end
        end
        =#
        #=
        # explicit
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        mprim = get_mixprim(cell.prim, tau, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        mw = get_conserved(mprim, KS.gas.γ, uq)
        for k=1:2
        cell.w[:,:,k] .+= (mw[:,:,k] .- w_old[:,:,k]) .* dt ./ tau[k]
        end
        cell.prim .= get_primitive(cell.w, KS.gas.γ, uq)

        wRan .= get_ran_array(cell.w, 2, uq);
        primRan .= get_ran_array(cell.prim, 2, uq);
        =#

        #--- update electromagnetic variables ---#
        # flux -> E^{n+1} & B^{n+1}
        #@. cell.E[1,:] -= dt * (faceL.femR[1,:] + faceR.femL[1,:]) / cell.dx
        #@. cell.E[2,:] -= dt * (faceL.femR[2,:] + faceR.femL[2,:]) / cell.dx
        #@. cell.E[3,:] -= dt * (faceL.femR[3,:] + faceR.femL[3,:]) / cell.dx
        #@. cell.B[1,:] -= dt * (faceL.femR[4,:] + faceR.femL[4,:]) / cell.dx
        #@. cell.B[2,:] -= dt * (faceL.femR[5,:] + faceR.femL[5,:]) / cell.dx
        #@. cell.B[3,:] -= dt * (faceL.femR[6,:] + faceR.femL[6,:]) / cell.dx
        @. cell.ϕ -= dt * (faceL.femR[7, :] + faceR.femL[7, :]) / cell.dx
        @. cell.ψ -= dt * (faceL.femR[8, :] + faceR.femL[8, :]) / cell.dx

        ERan = chaos_ran(cell.E, 2, uq)
        ERan[1, :] .-=
            dt .* (
                evaluatePCE(faceL.femR[1, :], uq.op.quad.nodes, uq.op) .+
                evaluatePCE(faceR.femL[1, :], uq.op.quad.nodes, uq.op)
            ) ./ cell.dx
        ERan[2, :] .-=
            dt .* (
                evaluatePCE(faceL.femR[2, :], uq.op.quad.nodes, uq.op) .+
                evaluatePCE(faceR.femL[2, :], uq.op.quad.nodes, uq.op)
            ) ./ cell.dx
        ERan[3, :] .-=
            dt .* (
                evaluatePCE(faceL.femR[3, :], uq.op.quad.nodes, uq.op) .+
                evaluatePCE(faceR.femL[3, :], uq.op.quad.nodes, uq.op)
            ) ./ cell.dx
        BRan = chaos_ran(cell.B, 2, uq)
        BRan[1, :] .-=
            dt .* (
                evaluatePCE(faceL.femR[4, :], uq.op.quad.nodes, uq.op) .+
                evaluatePCE(faceR.femL[4, :], uq.op.quad.nodes, uq.op)
            ) ./ cell.dx
        BRan[2, :] .-=
            dt .* (
                evaluatePCE(faceL.femR[5, :], uq.op.quad.nodes, uq.op) .+
                evaluatePCE(faceR.femL[5, :], uq.op.quad.nodes, uq.op)
            ) ./ cell.dx
        BRan[3, :] .-=
            dt .* (
                evaluatePCE(faceL.femR[6, :], uq.op.quad.nodes, uq.op) .+
                evaluatePCE(faceR.femL[6, :], uq.op.quad.nodes, uq.op)
            ) ./ cell.dx

        # source -> ϕ
        #@. cell.ϕ += dt * (cell.w[1,:,1] / KS.gas.mi - cell.w[1,:,2] / KS.gas.me) / (KS.gas.lD^2 * KS.gas.rL)

        # source -> U^{n+1}, E^{n+1} and B^{n+1}
        mr = KS.gas.mi / KS.gas.me

        #ERan = get_ran_array(cell.E, 2, uq)
        #BRan = get_ran_array(cell.B, 2, uq)

        xRan = zeros(9, uq.op.quad.Nquad)
        for j in axes(xRan, 2)
            A, b = em_coefficients(
                primRan[:, j, :],
                ERan[:, j],
                BRan[:, j],
                mr,
                KS.gas.lD,
                KS.gas.rL,
                dt,
            )
            xRan[:, j] .= A \ b
        end

        lorenzRan = zeros(3, uq.op.quad.Nquad, 2)
        for j in axes(lorenzRan, 2)
            lorenzRan[1, j, 1] =
                0.5 * (
                    xRan[1, j] + ERan[1, j] + (primRan[3, j, 1] + xRan[5, j]) * BRan[3, j] -
                    (primRan[4, j, 1] + xRan[6, j]) * BRan[2, j]
                ) / KS.gas.rL
            lorenzRan[2, j, 1] =
                0.5 * (
                    xRan[2, j] + ERan[2, j] + (primRan[4, j, 1] + xRan[6, j]) * BRan[1, j] -
                    (primRan[2, j, 1] + xRan[4, j]) * BRan[3, j]
                ) / KS.gas.rL
            lorenzRan[3, j, 1] =
                0.5 * (
                    xRan[3, j] + ERan[3, j] + (primRan[2, j, 1] + xRan[4, j]) * BRan[2, j] -
                    (primRan[3, j, 1] + xRan[5, j]) * BRan[1, j]
                ) / KS.gas.rL
            lorenzRan[1, j, 2] =
                -0.5 *
                (
                    xRan[1, j] + ERan[1, j] + (primRan[3, j, 2] + xRan[8, j]) * BRan[3, j] -
                    (primRan[4, j, 2] + xRan[9, j]) * BRan[2, j]
                ) *
                mr / KS.gas.rL
            lorenzRan[2, j, 2] =
                -0.5 *
                (
                    xRan[2, j] + ERan[2, j] + (primRan[4, j, 2] + xRan[9, j]) * BRan[1, j] -
                    (primRan[2, j, 2] + xRan[7, j]) * BRan[3, j]
                ) *
                mr / KS.gas.rL
            lorenzRan[3, j, 2] =
                -0.5 *
                (
                    xRan[3, j] + ERan[3, j] + (primRan[2, j, 2] + xRan[7, j]) * BRan[2, j] -
                    (primRan[3, j, 2] + xRan[8, j]) * BRan[1, j]
                ) *
                mr / KS.gas.rL
        end

        ERan[1, :] .= xRan[1, :]
        ERan[2, :] .= xRan[2, :]
        ERan[3, :] .= xRan[3, :]

        #--- update conservative flow variables: step 2 ---#
        primRan[2, :, 1] .= xRan[4, :]
        primRan[3, :, 1] .= xRan[5, :]
        primRan[4, :, 1] .= xRan[6, :]
        primRan[2, :, 2] .= xRan[7, :]
        primRan[3, :, 2] .= xRan[8, :]
        primRan[4, :, 2] .= xRan[9, :]

        for j in axes(wRan, 2)
            wRan[:, j, :] .= Kinetic.mixture_prim_conserve(primRan[:, j, :], KS.gas.γ)
        end

        cell.w .= ran_chaos(wRan, 2, uq)
        cell.prim .= ran_chaos(primRan, 2, uq)
        cell.E .= ran_chaos(ERan, 2, uq)
        cell.lorenz .= ran_chaos(lorenzRan, 2, uq)
        cell.B .= ran_chaos(BRan, 2, uq)

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        #@. cell.h0 += (faceL.fh0 - faceR.fh0) / cell.dx
        #@. cell.h1 += (faceL.fh1 - faceR.fh1) / cell.dx
        #@. cell.h2 += (faceL.fh2 - faceR.fh2) / cell.dx
        #@. cell.h3 += (faceL.fh3 - faceR.fh3) / cell.dx

        #h0Ran = get_ran_array(cell.h0, 2, uq)
        #h1Ran = get_ran_array(cell.h1, 2, uq)
        #h2Ran = get_ran_array(cell.h2, 2, uq)
        #h3Ran = get_ran_array(cell.h3, 2, uq)

        h0Ran =
            chaos_ran(cell.h0, 2, uq) .+
            (chaos_ran(faceL.fh0, 2, uq) .- chaos_ran(faceR.fh0, 2, uq)) ./ cell.dx
        h1Ran =
            chaos_ran(cell.h1, 2, uq) .+
            (chaos_ran(faceL.fh1, 2, uq) .- chaos_ran(faceR.fh1, 2, uq)) ./ cell.dx
        h2Ran =
            chaos_ran(cell.h2, 2, uq) .+
            (chaos_ran(faceL.fh2, 2, uq) .- chaos_ran(faceR.fh2, 2, uq)) ./ cell.dx
        h3Ran =
            chaos_ran(cell.h3, 2, uq) .+
            (chaos_ran(faceL.fh3, 2, uq) .- chaos_ran(faceR.fh3, 2, uq)) ./ cell.dx

        # force -> f^{n+1} : step 1
        for j in axes(h0Ran, 2)
            _h0 = @view h0Ran[:, j, :]
            _h1 = @view h1Ran[:, j, :]
            _h2 = @view h2Ran[:, j, :]
            _h3 = @view h3Ran[:, j, :]

            shift_pdf!(_h0, lorenzRan[1, j, :], KS.vSpace.du[1, :], dt)
            shift_pdf!(_h1, lorenzRan[1, j, :], KS.vSpace.du[1, :], dt)
            shift_pdf!(_h2, lorenzRan[1, j, :], KS.vSpace.du[1, :], dt)
            shift_pdf!(_h3, lorenzRan[1, j, :], KS.vSpace.du[1, :], dt)
        end

        # force -> f^{n+1} : step 2
        for k in axes(h1Ran, 3)
            for j in axes(h1Ran, 2)
                @. h3Ran[:, j, k] +=
                    2.0 * dt * lorenzRan[2, j, k] * h1Ran[:, j, k] +
                    (dt * lorenzRan[2, j, k])^2 * h0Ran[:, j, k] +
                    2.0 * dt * lorenzRan[3, j, k] * h2Ran[:, j, k] +
                    (dt * lorenzRan[3, j, k])^2 * h0Ran[:, j, k]
                @. h2Ran[:, j, k] += dt * lorenzRan[3, j, k] * h0Ran[:, j, k]
                @. h1Ran[:, j, k] += dt * lorenzRan[2, j, k] * h0Ran[:, j, k]
            end
        end

        # source -> f^{n+1}
        #tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], uq)
        tau = get_tau(primRan, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])

        # interspecies interaction
        for j in axes(primRan, 2)
            primRan[:, j, :] .= Kinetic.aap_hs_prim(
                primRan[:, j, :],
                tau,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
            )
        end

        gRan = zeros(KS.vSpace.nu, uq.op.quad.Nquad, 2)
        for k in axes(gRan, 3)
            for j in axes(gRan, 2)
                gRan[:, j, k] .= Kinetic.maxwellian(KS.vSpace.u[:, k], primRan[:, j, k])
            end
        end

        # BGK term
        for j in axes(h0Ran, 2)
            Mu, Mv, Mw, MuL, MuR = Kinetic.mixture_gauss_moments(primRan[:, j, :], KS.gas.K)
            for k in axes(h0Ran, 3)
                @. h0Ran[:, j, k] =
                    (h0Ran[:, j, k] + dt / tau[k] * gRan[:, j, k]) / (1.0 + dt / tau[k])
                @. h1Ran[:, j, k] =
                    (h1Ran[:, j, k] + dt / tau[k] * Mv[1, k] * gRan[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. h2Ran[:, j, k] =
                    (h2Ran[:, j, k] + dt / tau[k] * Mw[1, k] * gRan[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. h3Ran[:, j, k] =
                    (h3Ran[:, j, k] + dt / tau[k] * (Mv[2, k] + Mw[2, k]) * gRan[:, j, k]) /
                    (1.0 + dt / tau[k])
            end
        end

        cell.h0 .= ran_chaos(h0Ran, 2, uq)
        cell.h1 .= ran_chaos(h1Ran, 2, uq)
        cell.h2 .= ran_chaos(h2Ran, 2, uq)
        cell.h3 .= ran_chaos(h3Ran, 2, uq)

        #--- record residuals ---#
        @. RES += (w_old[:, 1, :] - cell.w[:, 1, :])^2
        @. AVG += abs(cell.w[:, 1, :])
        #=
        #-- filter ---#
        λ = 0.00001
        for k in 1:2, i in 1:5
        filter!(cell.w[i,:,k], λ)
        filter!(cell.prim[i,:,k], λ)
        end
        for k in 1:2, i in axes(cell.h0, 1)
        filter!(cell.h0[i,:,k], λ)
        filter!(cell.h1[i,:,k], λ)
        filter!(cell.h2[i,:,k], λ)
        filter!(cell.h3[i,:,k], λ)
        end
        for k in 1:2, i in 1:3
        filter!(cell.lorenz[i,:,k], λ)
        end
        for i in 1:3
        filter!(cell.E[i,:], λ)
        filter!(cell.B[i,:], λ)
        end
        filter!(cell.ϕ, λ)
        filter!(cell.ψ, λ)
        =#

    elseif uq.method == "collocation"

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        @. cell.w += (faceL.fw - faceR.fw) / cell.dx
        for j in axes(cell.prim, 2)
            cell.prim[:,j,:] .= mixture_conserve_prim(cell.w[:,j,:], KS.gas.γ)
        end

        # temperature protection
        if min(minimum(cell.prim[5, :, 1]), minimum(cell.prim[5, :, 2])) < 0
            @warn "negative temperature update"
            cell.w .= w_old
            cell.prim .= prim_old
        end

        #=
        # source -> w^{n+1}
        # DifferentialEquations.jl
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        for j in axes(wRan, 2)
        prob = ODEProblem( mixture_source, 
            vcat(cell.w[1:5,j,1], cell.w[1:5,j,2]),
            dt,
            (tau[1], tau[2], KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], KS.gas.γ) )
        sol = solve(prob, Rosenbrock23())

        cell.w[1:5,j,1] .= sol[end][1:5]
        cell.w[1:5,j,2] .= sol[end][6:10]
        for k=1:2
        cell.prim[:,j,k] .= Kinetic.conserve_prim(cell.w[:,j,k], KS.gas.γ)
        end
        end
        =#
        #=
        # explicit
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        mprim = get_mixprim(cell.prim, tau, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        mw = get_conserved(mprim, KS.gas.γ)
        for k=1:2
        cell.w[:,:,k] .+= (mw[:,:,k] .- w_old[:,:,k]) .* dt ./ tau[k]
        end
        cell.prim .= get_primitive(cell.w, KS.gas.γ)
        =#
#=
        #--- update electromagnetic variables ---#
        # flux -> E^{n+1} & B^{n+1}
        @. cell.E[1, :] -= dt * (faceL.femR[1, :] + faceR.femL[1, :]) / cell.dx
        @. cell.E[2, :] -= dt * (faceL.femR[2, :] + faceR.femL[2, :]) / cell.dx
        @. cell.E[3, :] -= dt * (faceL.femR[3, :] + faceR.femL[3, :]) / cell.dx
        @. cell.B[1, :] -= dt * (faceL.femR[4, :] + faceR.femL[4, :]) / cell.dx
        @. cell.B[2, :] -= dt * (faceL.femR[5, :] + faceR.femL[5, :]) / cell.dx
        @. cell.B[3, :] -= dt * (faceL.femR[6, :] + faceR.femL[6, :]) / cell.dx
        @. cell.ϕ -= dt * (faceL.femR[7, :] + faceR.femL[7, :]) / cell.dx
        @. cell.ψ -= dt * (faceL.femR[8, :] + faceR.femL[8, :]) / cell.dx

        # source -> ϕ
        #@. cell.ϕ += dt * (cell.w[1,:,1] / KS.gas.mi - cell.w[1,:,2] / KS.gas.me) / (KS.gas.lD^2 * KS.gas.rL)

        # source -> U^{n+1}, E^{n+1} and B^{n+1}
        mr = KS.gas.mi / KS.gas.me

        x = zeros(9, uq.op.quad.Nquad)
        for j in axes(x, 2)
            A, b = em_coefficients(
                cell.prim[:, j, :],
                cell.E[:, j],
                cell.B[:, j],
                mr,
                KS.gas.lD,
                KS.gas.rL,
                dt,
            )
            x[:, j] .= A \ b
        end

        #--- calculate lorenz force ---#
        for j in axes(cell.lorenz, 2)
            cell.lorenz[1, j, 1] =
                0.5 * (
                    x[1, j] + cell.E[1, j] + (cell.prim[3, j, 1] + x[5, j]) * cell.B[3, j] -
                    (cell.prim[4, j, 1] + x[6, j]) * cell.B[2, j]
                ) / KS.gas.rL
            cell.lorenz[2, j, 1] =
                0.5 * (
                    x[2, j] + cell.E[2, j] + (cell.prim[4, j, 1] + x[6, j]) * cell.B[1, j] -
                    (cell.prim[2, j, 1] + x[4, j]) * cell.B[3, j]
                ) / KS.gas.rL
            cell.lorenz[3, j, 1] =
                0.5 * (
                    x[3, j] + cell.E[3, j] + (cell.prim[2, j, 1] + x[4, j]) * cell.B[2, j] -
                    (cell.prim[3, j, 1] + x[5, j]) * cell.B[1, j]
                ) / KS.gas.rL
            cell.lorenz[1, j, 2] =
                -0.5 *
                (
                    x[1, j] + cell.E[1, j] + (cell.prim[3, j, 2] + x[8, j]) * cell.B[3, j] -
                    (cell.prim[4, j, 2] + x[9, j]) * cell.B[2, j]
                ) *
                mr / KS.gas.rL
            cell.lorenz[2, j, 2] =
                -0.5 *
                (
                    x[2, j] + cell.E[2, j] + (cell.prim[4, j, 2] + x[9, j]) * cell.B[1, j] -
                    (cell.prim[2, j, 2] + x[7, j]) * cell.B[3, j]
                ) *
                mr / KS.gas.rL
            cell.lorenz[3, j, 2] =
                -0.5 *
                (
                    x[3, j] + cell.E[3, j] + (cell.prim[2, j, 2] + x[7, j]) * cell.B[2, j] -
                    (cell.prim[3, j, 2] + x[8, j]) * cell.B[1, j]
                ) *
                mr / KS.gas.rL
        end

        cell.E[1, :] .= x[1, :]
        cell.E[2, :] .= x[2, :]
        cell.E[3, :] .= x[3, :]

        #--- update conservative flow variables: step 2 ---#
        cell.prim[2, :, 1] .= x[4, :]
        cell.prim[3, :, 1] .= x[5, :]
        cell.prim[4, :, 1] .= x[6, :]
        cell.prim[2, :, 2] .= x[7, :]
        cell.prim[3, :, 2] .= x[8, :]
        cell.prim[4, :, 2] .= x[9, :]

        for j in axes(cell.w, 2)
            cell.w[:, j, :] .= Kinetic.mixture_prim_conserve(cell.prim[:, j, :], KS.gas.γ)
        end
=#
        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        @. cell.h0 += (faceL.fh0 - faceR.fh0) / cell.dx
        @. cell.h1 += (faceL.fh1 - faceR.fh1) / cell.dx
        @. cell.h2 += (faceL.fh2 - faceR.fh2) / cell.dx
        @. cell.h3 += (faceL.fh3 - faceR.fh3) / cell.dx
        #=
                # force -> f^{n+1} : step 1
                for k in axes(cell.h0, 3)
                    for j in axes(cell.h0, 2)
                        #shift_pdf!(cell.h0[:,j,:], cell.lorenz[1,j,:], KS.vSpace.du[1,:], dt)
                        #shift_pdf!(cell.h1[:,j,:], cell.lorenz[1,j,:], KS.vSpace.du[1,:], dt)
                        #shift_pdf!(cell.h2[:,j,:], cell.lorenz[1,j,:], KS.vSpace.du[1,:], dt)
                        #shift_pdf!(cell.h3[:,j,:], cell.lorenz[1,j,:], KS.vSpace.du[1,:], dt)

                        _h0 = @view cell.h0[:, j, k]
                        _h1 = @view cell.h1[:, j, k]
                        _h2 = @view cell.h2[:, j, k]
                        _h3 = @view cell.h3[:, j, k]

                        shift_pdf!(_h0, cell.lorenz[1, j, k], KS.vSpace.du[1, k], dt)
                        shift_pdf!(_h1, cell.lorenz[1, j, k], KS.vSpace.du[1, k], dt)
                        shift_pdf!(_h2, cell.lorenz[1, j, k], KS.vSpace.du[1, k], dt)
                        shift_pdf!(_h3, cell.lorenz[1, j, k], KS.vSpace.du[1, k], dt)
                    end
                end

                # force -> f^{n+1} : step 2
                for k in axes(cell.h1, 3), j in axes(cell.h1, 2)
                    @. cell.h3[:,j,k] += 2. * dt * cell.lorenz[2,j,k] * cell.h1[:,j,k] + (dt * cell.lorenz[2,j,k])^2 * cell.h0[:,j,k] +
                                         2. * dt * cell.lorenz[3,j,k] * cell.h2[:,j,k] + (dt * cell.lorenz[3,j,k])^2 * cell.h0[:,j,k]
                    @. cell.h2[:,j,k] += dt * cell.lorenz[3,j,k] * cell.h0[:,j,k]
                    @. cell.h1[:,j,k] += dt * cell.lorenz[2,j,k] * cell.h0[:,j,k]
                end
        =#
        # source -> f^{n+1}
        tau = uq_aap_hs_collision_time(
            cell.prim,
            KS.gas.mi,
            KS.gas.ni,
            KS.gas.me,
            KS.gas.ne,
            KS.gas.Kn[1],
            uq,
        )

        # interspecies interaction
        prim = deepcopy(cell.prim)
        #for j in axes(prim, 2)
        #    prim[:,j,:] .= Kinetic.aap_hs_prim(cell.prim[:,j,:], tau, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        #end

        g = zeros(KS.vSpace.nu, uq.op.quad.Nquad, 2)
        for j in axes(g, 2)
            g[:, j, :] .= mixture_maxwellian(KS.vSpace.u, prim[:, j, :])
        end

        # BGK term
        for j in axes(cell.h0, 2)
            Mu, Mv, Mw, MuL, MuR = mixture_gauss_moments(prim[:, j, :], KS.gas.K)
            for k in axes(cell.h0, 3)
                @. cell.h0[:, j, k] =
                    (cell.h0[:, j, k] + dt / tau[k] * g[:, j, k]) / (1.0 + dt / tau[k])
                @. cell.h1[:, j, k] =
                    (cell.h1[:, j, k] + dt / tau[k] * Mv[1, k] * g[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. cell.h2[:, j, k] =
                    (cell.h2[:, j, k] + dt / tau[k] * Mw[1, k] * g[:, j, k]) /
                    (1.0 + dt / tau[k])
                @. cell.h3[:, j, k] =
                    (cell.h3[:, j, k] + dt / tau[k] * (Mv[2, k] + Mw[2, k]) * g[:, j, k]) /
                    (1.0 + dt / tau[k])
            end
        end

        #--- record residuals ---#
        @. RES += (w_old[:, 1, :] - cell.w[:, 1, :])^2
        @. AVG += abs(cell.w[:, 1, :])

    end

end

function step!(
    KS::SolverSet,
    uq::AbstractUQ,
    faceL::Interface1D3F,
    cell::ControlVolume1D3F,
    faceR::Interface1D3F,
    dt::AbstractFloat,
    RES::Array{<:AbstractFloat,2},
    AVG::Array{<:AbstractFloat,2},
)

    if uq.method == "galerkin"

    elseif uq.method == "collocation"

        #--- update conservative flow variables: step 1 ---#
        # w^n
        w_old = deepcopy(cell.w)
        prim_old = deepcopy(cell.prim)

        # flux -> w^{n+1}
        @. cell.w += (faceL.fw - faceR.fw) / cell.dx
        cell.prim .= uq_conserve_prim(cell.w, KS.gas.γ, uq)

        # temperature protection
        if min(minimum(cell.prim[5, :, 1]), minimum(cell.prim[5, :, 2])) < 0
            println("warning: temperature update is negative")
            cell.w .= w_old
            cell.prim .= prim_old
        end

        #=
        # source -> w^{n+1}
        # DifferentialEquations.jl
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        for j in axes(wRan, 2)
        prob = ODEProblem( mixture_source, 
            vcat(cell.w[1:5,j,1], cell.w[1:5,j,2]),
            dt,
            (tau[1], tau[2], KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1], KS.gas.γ) )
        sol = solve(prob, Rosenbrock23())

        cell.w[1:5,j,1] .= sol[end][1:5]
        cell.w[1:5,j,2] .= sol[end][6:10]
        for k=1:2
        cell.prim[:,j,k] .= Kinetic.conserve_prim(cell.w[:,j,k], KS.gas.γ)
        end
        end
        =#
        #=
        # explicit
        tau = get_tau(cell.prim, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        mprim = get_mixprim(cell.prim, tau, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        mw = get_conserved(mprim, KS.gas.γ)
        for k=1:2
        cell.w[:,:,k] .+= (mw[:,:,k] .- w_old[:,:,k]) .* dt ./ tau[k]
        end
        cell.prim .= get_primitive(cell.w, KS.gas.γ)
        =#

        #--- update electromagnetic variables ---#
        # flux -> E^{n+1} & B^{n+1}
        @. cell.E[1, :] -= dt * (faceL.femR[1, :] + faceR.femL[1, :]) / cell.dx
        @. cell.E[2, :] -= dt * (faceL.femR[2, :] + faceR.femL[2, :]) / cell.dx
        @. cell.E[3, :] -= dt * (faceL.femR[3, :] + faceR.femL[3, :]) / cell.dx
        @. cell.B[1, :] -= dt * (faceL.femR[4, :] + faceR.femL[4, :]) / cell.dx
        @. cell.B[2, :] -= dt * (faceL.femR[5, :] + faceR.femL[5, :]) / cell.dx
        @. cell.B[3, :] -= dt * (faceL.femR[6, :] + faceR.femL[6, :]) / cell.dx
        @. cell.ϕ -= dt * (faceL.femR[7, :] + faceR.femL[7, :]) / cell.dx
        @. cell.ψ -= dt * (faceL.femR[8, :] + faceR.femL[8, :]) / cell.dx

        # source -> ϕ
        #@. cell.ϕ += dt * (cell.w[1,:,1] / KS.gas.mi - cell.w[1,:,2] / KS.gas.me) / (KS.gas.lD^2 * KS.gas.rL)

        # source -> U^{n+1}, E^{n+1} and B^{n+1}
        mr = KS.gas.mi / KS.gas.me

        x = zeros(9, uq.op.quad.Nquad)
        for j in axes(x, 2)
            A, b = em_coefficients(
                cell.prim[:, j, :],
                cell.E[:, j],
                cell.B[:, j],
                mr,
                KS.gas.lD,
                KS.gas.rL,
                dt,
            )
            x[:, j] .= A \ b
        end

        #--- calculate lorenz force ---#
        for j in axes(cell.lorenz, 2)
            cell.lorenz[1, j, 1] =
                0.5 * (
                    x[1, j] + cell.E[1, j] + (cell.prim[3, j, 1] + x[5, j]) * cell.B[3, j] -
                    (cell.prim[4, j, 1] + x[6, j]) * cell.B[2, j]
                ) / KS.gas.rL
            cell.lorenz[2, j, 1] =
                0.5 * (
                    x[2, j] + cell.E[2, j] + (cell.prim[4, j, 1] + x[6, j]) * cell.B[1, j] -
                    (cell.prim[2, j, 1] + x[4, j]) * cell.B[3, j]
                ) / KS.gas.rL
            cell.lorenz[3, j, 1] =
                0.5 * (
                    x[3, j] + cell.E[3, j] + (cell.prim[2, j, 1] + x[4, j]) * cell.B[2, j] -
                    (cell.prim[3, j, 1] + x[5, j]) * cell.B[1, j]
                ) / KS.gas.rL
            cell.lorenz[1, j, 2] =
                -0.5 *
                (
                    x[1, j] + cell.E[1, j] + (cell.prim[3, j, 2] + x[8, j]) * cell.B[3, j] -
                    (cell.prim[4, j, 2] + x[9, j]) * cell.B[2, j]
                ) *
                mr / KS.gas.rL
            cell.lorenz[2, j, 2] =
                -0.5 *
                (
                    x[2, j] + cell.E[2, j] + (cell.prim[4, j, 2] + x[9, j]) * cell.B[1, j] -
                    (cell.prim[2, j, 2] + x[7, j]) * cell.B[3, j]
                ) *
                mr / KS.gas.rL
            cell.lorenz[3, j, 2] =
                -0.5 *
                (
                    x[3, j] + cell.E[3, j] + (cell.prim[2, j, 2] + x[7, j]) * cell.B[2, j] -
                    (cell.prim[3, j, 2] + x[8, j]) * cell.B[1, j]
                ) *
                mr / KS.gas.rL
        end

        cell.E[1, :] .= x[1, :]
        cell.E[2, :] .= x[2, :]
        cell.E[3, :] .= x[3, :]

        #--- update conservative flow variables: step 2 ---#
        cell.prim[2, :, 1] .= x[4, :]
        cell.prim[3, :, 1] .= x[5, :]
        cell.prim[4, :, 1] .= x[6, :]
        cell.prim[2, :, 2] .= x[7, :]
        cell.prim[3, :, 2] .= x[8, :]
        cell.prim[4, :, 2] .= x[9, :]

        for j in axes(cell.w, 2)
            cell.w .= uq_prim_conserve(cell.prim, KS.gas.γ, uq)
        end

        #--- update particle distribution function ---#
        # flux -> f^{n+1}
        @. cell.h0 += (faceL.fh0 - faceR.fh0) / cell.dx
        @. cell.h1 += (faceL.fh1 - faceR.fh1) / cell.dx
        @. cell.h2 += (faceL.fh2 - faceR.fh2) / cell.dx

        # force -> f^{n+1} : step 1
        for j in axes(cell.h0, 3)
            for i in axes(cell.h0, 2)
                _h0 = @view cell.h0[:, i, j, :]
                _h1 = @view cell.h1[:, i, j, :]
                _h2 = @view cell.h2[:, i, j, :]

                shift_pdf!(_h0, cell.lorenz[1, j, :], KS.vSpace.du[1, i, :], dt)
                shift_pdf!(_h1, cell.lorenz[1, j, :], KS.vSpace.du[1, i, :], dt)
                shift_pdf!(_h2, cell.lorenz[1, j, :], KS.vSpace.du[1, i, :], dt)
            end
        end

        for j in axes(cell.h0, 3)
            for i in axes(cell.h0, 1)
                _h0 = @view cell.h0[i, :, j, :]
                _h1 = @view cell.h1[i, :, j, :]
                _h2 = @view cell.h2[i, :, j, :]

                shift_pdf!(_h0, cell.lorenz[2, j, :], KS.vSpace.du[i, 1, :], dt)
                shift_pdf!(_h1, cell.lorenz[2, j, :], KS.vSpace.du[i, 1, :], dt)
                shift_pdf!(_h2, cell.lorenz[2, j, :], KS.vSpace.du[i, 1, :], dt)
            end
        end

        # force -> f^{n+1} : step 2
        for k in axes(cell.h1, 4), j in axes(cell.h1, 3)
            @. cell.h2[:, :, j, k] +=
                2.0 * dt * cell.lorenz[3, j, k] * cell.h1[:, :, j, k] +
                (dt * cell.lorenz[3, j, k])^2 * cell.h0[:, :, j, k]
            @. cell.h1[:, :, j, k] += dt * cell.lorenz[3, j, k] * cell.h0[:, :, j, k]
        end

        # source -> f^{n+1}
        tau = uq_aap_hs_collision_time(
            cell.prim,
            KS.gas.mi,
            KS.gas.ni,
            KS.gas.me,
            KS.gas.ne,
            KS.gas.Kn[1],
            uq,
        )

        # interspecies interaction
        prim = deepcopy(cell.prim)
        #for j in axes(prim, 2)
        #    prim[:,j,:] .= Kinetic.aap_hs_prim(cell.prim[:,j,:], tau, KS.gas.mi, KS.gas.ni, KS.gas.me, KS.gas.ne, KS.gas.Kn[1])
        #end

        H0, H1, H2 = uq_maxwellian(KS.vSpace.u, KS.vSpace.v, prim, uq)

        # BGK term
        for j in axes(cell.h0, 3)
            for k in axes(cell.h0, 4)
                @. cell.h0[:, :, j, k] =
                    (cell.h0[:, :, j, k] + dt / tau[k] * H0[:, :, j, k]) /
                    (1.0 + dt / tau[k])
                @. cell.h1[:, :, j, k] =
                    (cell.h1[:, :, j, k] + dt / tau[k] * H1[:, :, j, k]) /
                    (1.0 + dt / tau[k])
                @. cell.h2[:, :, j, k] =
                    (cell.h2[:, :, j, k] + dt / tau[k] * H2[:, :, j, k]) /
                    (1.0 + dt / tau[k])
            end
        end

        #--- record residuals ---#
        @. RES += (w_old[:, 1, :] - cell.w[:, 1, :])^2
        @. AVG += abs(cell.w[:, 1, :])

    end

end
