"""
$(SIGNATURES)

Calculate time step
"""
function KitBase.timestep(KS, uq::AbstractUQ, sol::AbstractSolution, simTime)
    tmax = 0.0

    if KS.set.nSpecies == 1

        if KS.set.space[1:2] == "1d"
            @inbounds @threads for i = 1:KS.ps.nx
                sos = uq_sound_speed(sol.prim[i], KS.gas.γ, uq)
                vmax = KS.vs.u1 + maximum(sos)
                tmax = max(tmax, vmax / KS.ps.dx[i])
            end
        elseif KS.set.space[1:2] == "2d"
            @inbounds @threads for j = 1:KS.ps.ny
                for i = 1:KS.ps.nx
                    sos = uq_sound_speed(sol.prim[i, j], KS.gas.γ, uq)
                    vmax = max(KS.vs.u1, KS.vs.v1) + maximum(sos)
                    tmax = max(tmax, vmax / KS.ps.dx[i, j] + vmax / KS.ps.dy[i, j])
                end
            end
        end

    elseif KS.set.nSpecies == 2

        @inbounds @threads for i = 1:KS.ps.nx
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

"""
$(SIGNATURES)
"""
function KitBase.timestep(
    KS,
    uq::AbstractUQ,
    ctr::AbstractVector{T},
    simTime,
) where {T<:Union{ControlVolume1F,ControlVolume2F,ControlVolume1D1F,ControlVolume1D2F}}

    tmax = 0.0
    @inbounds @threads for i = 1:KS.ps.nx
        prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vs.u1), maximum(abs.(prim[2, :]))) + maximum(sos)
        tmax = max(tmax, vmax / KS.ps.dx[i])
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt

end

"""
$(SIGNATURES)
"""
function KitBase.timestep(
    KS,
    uq::AbstractUQ,
    ctr::AV{ControlVolume1D4F},
    simTime,
)

    tmax = 0.0

    @inbounds @threads for i = 1:KS.ps.nx
        prim = ctr[i].prim
        sos = uq_sound_speed(prim, KS.gas.γ, uq)
        vmax = max(maximum(KS.vs.u1), maximum(abs.(prim[2, :, :]))) + maximum(sos)
        tmax = max(tmax, vmax / KS.ps.dx[i], KS.gas.sol / KS.ps.dx[i])
    end

    dt = KS.set.cfl / tmax
    dt = ifelse(dt < (KS.set.maxTime - simTime), dt, KS.set.maxTime - simTime)

    return dt

end

"""
$(SIGNATURES)
"""
function KitBase.timestep(
    KS,
    uq::AbstractUQ,
    ctr::AV{ControlVolume1D3F},
    simTime,
)

    tmax = 0.0

    @inbounds @threads for i = 1:KS.ps.nx
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
