# ============================================================
# Fluxes for Control Volume Structures
# ============================================================

"""
$(SIGNATURES)

Evolve field solution

* particle evolution: `KFVS`, `KCU`, `UGKS`
* electromagnetic evolution: wave propagation method
"""
function KitBase.evolve!(
    KS::AbstractSolverSet,
    uq::AbstractUQ,
    ctr::AV,
    face::AV,
    dt;
    mode = :kfvs,
    isPlasma = false,
    isMHD = false,
)

    # flow field
    if uq.method == "collocation"
        @inbounds @threads for i in eachindex(face)
            uqflux_flow_collocation!(
                KS,
                ctr[i-1],
                face[i],
                ctr[i],
                dt,
                KS.ps.dx[i-1],
                KS.ps.dx[i];
                mode = mode,
                isMHD = isMHD,
            )
        end
    elseif uq.method == "galerkin"
        @inbounds @threads for i in eachindex(face)
            uqflux_flow_galerkin!(
                KS,
                uq,
                ctr[i-1],
                face[i],
                ctr[i],
                dt,
                KS.ps.dx[i-1],
                KS.ps.dx[i];
                mode = mode,
                isMHD = isMHD,
            )
        end
    else
        throw("UQ method isn't available")
    end

    # electromagnetic field
    if isPlasma
        @inbounds @threads for i in eachindex(face)
            uqflux_em!(
                KS,
                uq,
                ctr[i-2],
                ctr[i-1],
                face[i],
                ctr[i],
                ctr[i+1],
                dt,
                KS.ps.dx[i-1],
                KS.ps.dx[i],
            )
        end
    end

    return nothing

end


"""
$(SIGNATURES)

Calculate flux of particle transport
"""
function uqflux_flow_galerkin!(
    KS,
    uq,
    cellL::T,
    face,
    cellR::T,
    dt,
    dxL,
    dxR;
    mode = :kfvs,
    isMHD = false,
) where {T<:Union{ControlVolume1F,ControlVolume1D1F}}

    if mode == :kfvs
        @inbounds for j in axes(cellL.f, 2)
            fw = @view face.fw[:, j]
            ff = @view face.ff[:, j]

            flux_kfvs!(
                fw,
                ff,
                cellL.f[:, j] .+ 0.5 .* dxL .* cellL.sf[:, j],
                cellR.f[:, j] .- 0.5 .* dxR .* cellR.sf[:, j],
                KS.vs.u,
                KS.vs.weights,
                dt,
                cellL.sf[:, j],
                cellR.sf[:, j],
            )
        end
    elseif mode == :kcu
        fw = chaos_ran(face.fw, 2, uq)
        ff = chaos_ran(face.ff, 2, uq)

        wL = chaos_ran(cellL.w .+ 0.5 .* dxL .* cellL.sw, 2, uq)
        fL = chaos_ran(cellL.f .+ 0.5 .* dxL .* cellL.sf, 2, uq)

        wR = chaos_ran(cellR.w .- 0.5 .* dxR .* cellR.sw, 2, uq)
        fR = chaos_ran(cellR.f .- 0.5 .* dxR .* cellR.sf, 2, uq)

        @inbounds for j in axes(fL, 2)
            _fw = @view fw[:, j]
            _ff = @view ff[:, j]

            flux_kcu!(
                _fw,
                _ff,
                wL[:, j],
                fL[:, j],
                wR[:, j],
                fR[:, j],
                KS.vs.u,
                KS.vs.weights,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.μᵣ,
                KS.gas.ωᵣ,
                KS.gas.Pr,
                dt,
            )
        end

        face.fw .= chaos_ran(fw, 2, uq)
        face.ff .= chaos_ran(ff, 2, uq)
    elseif mode == :ugks
        fw = chaos_ran(face.fw, 2, uq)
        ff = chaos_ran(face.ff, 2, uq)

        wL = chaos_ran(cellL.w .+ 0.5 .* dxL .* cellL.sw, 2, uq)
        fL = chaos_ran(cellL.f .+ 0.5 .* dxL .* cellL.sf, 2, uq)
        sfL = chaos_ran(cellL.sf, 2, uq)

        wR = chaos_ran(cellR.w .- 0.5 .* dxR .* cellR.sw, 2, uq)
        fR = chaos_ran(cellR.f .- 0.5 .* dxR .* cellR.sf, 2, uq)
        sfR = chaos_ran(cellR.sf, 2, uq)

        @inbounds for j in axes(fL, 2)
            _fw = @view face.fw[:, j]
            _ff = @view face.ff[:, j]

            flux_ugks!(
                _fw,
                _ff,
                wL[:, j],
                fL[:, j],
                wR[:, j],
                fR[:, j],
                KS.vs.u,
                KS.vs.weights,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.μᵣ,
                KS.gas.ωᵣ,
                KS.gas.Pr,
                dt,
                0.5 * dxL,
                0.5 * dxR,
                sfL[:, j],
                sfR[:, j],
            )
        end

        face.fw .= chaos_ran(fw, 2, uq)
        face.ff .= chaos_ran(ff, 2, uq)
    else
        throw("flux mode not available")
    end

    return nothing

end

function uqflux_flow_collocation!(
    KS,
    cellL::T,
    face,
    cellR::T,
    dt,
    dxL,
    dxR;
    mode = :kfvs,
    isMHD = false,
) where {T<:Union{ControlVolume1F,ControlVolume1D1F}}

    if mode == :kfvs

        if ndims(cellL.f) == 2 # pure

            @inbounds for j in axes(cellL.f, 2)
                fw = @view face.fw[:, j]
                ff = @view face.ff[:, j]

                flux_kfvs!(
                    fw,
                    ff,
                    cellL.f[:, j] .+ 0.5 .* dxL .* cellL.sf[:, j],
                    cellR.f[:, j] .- 0.5 .* dxR .* cellR.sf[:, j],
                    KS.vs.u,
                    KS.vs.weights,
                    dt,
                    cellL.sf[:, j],
                    cellR.sf[:, j],
                )
            end

        elseif ndims(cellL.f) == 3 # mixture

            @inbounds for k in axes(cellL.f, 3)
                for j in axes(cellL.f, 2)
                    fw = @view face.fw[:, j, k]
                    ff = @view face.ff[:, j, k]

                    flux_kfvs!(
                        fw,
                        ff,
                        cellL.f[:, j, k] .+ 0.5 .* dxL .* cellL.sf[:, j, k],
                        cellR.f[:, j, k] .- 0.5 .* dxR .* cellR.sf[:, j, k],
                        KS.vs.u[:, k],
                        KS.vs.weights[:, k],
                        dt,
                        cellL.sf[:, j, k],
                        cellR.sf[:, j, k],
                    )
                end
            end

        else

            throw("inconsistent distribution function size")

        end

    end

end

function uqflux_flow_collocation!(
    KS,
    cellL::T,
    face,
    cellR::T,
    dt,
    dxL,
    dxR;
    mode = :kfvs,
    isMHD = false,
) where {T<:Union{ControlVolume2F,ControlVolume1D2F}}

    if mode == :kfvs

        if ndims(cellL.h) == 2 # pure
            @inbounds for j in axes(cellL.h, 2)
                fw = @view face.fw[:, j]
                fh = @view face.fh[:, j]
                fb = @view face.fb[:, j]

                flux_kfvs!(
                    fw,
                    fh,
                    fb,
                    cellL.h[:, j] .+ 0.5 .* dxL .* cellL.sh[:, j],
                    cellL.b[:, j] .+ 0.5 .* dxL .* cellL.sb[:, j],
                    cellR.h[:, j] .- 0.5 .* dxR .* cellR.sh[:, j],
                    cellR.b[:, j] .- 0.5 .* dxR .* cellR.sb[:, j],
                    KS.vs.u,
                    KS.vs.weights,
                    dt,
                    cellL.sh[:, j],
                    cellL.sb[:, j],
                    cellR.sh[:, j],
                    cellR.sb[:, j],
                )
            end
        elseif ndims(cellL.h) == 3 # mixture
            @inbounds for k in axes(cellL.h, 3)
                for j in axes(cellL.f, 2)
                    fw = @view face.fw[:, j, k]
                    fh = @view face.fh[:, j, k]
                    fb = @view face.fb[:, j, k]

                    flux_kfvs!(
                        fw,
                        fh,
                        fb,
                        cellL.h[:, j, k] .+ 0.5 .* dxL .* cellL.sh[:, j, k],
                        cellL.b[:, j, k] .+ 0.5 .* dxL .* cellL.sb[:, j, k],
                        cellR.h[:, j, k] .- 0.5 .* dxR .* cellR.sh[:, j, k],
                        cellR.b[:, j, k] .- 0.5 .* dxR .* cellR.sb[:, j, k],
                        KS.vs.u[:, k],
                        KS.vs.weights[:, k],
                        dt,
                        cellL.sh[:, j, k],
                        cellL.sb[:, j, k],
                        cellR.sh[:, j, k],
                        cellR.sb[:, j, k],
                    )
                end
            end
        else
            throw("inconsistent distribution function size")
        end

    end

end

# ------------------------------------------------------------
# 1D4F1V
# ------------------------------------------------------------
function uqflux_flow_collocation!(
    KS,
    cellL::ControlVolume1D4F,
    face,
    cellR::ControlVolume1D4F,
    dt,
    dxL,
    dxR;
    mode = :kfvs,
    isMHD = false,
)

    if mode == :kfvs

        @inbounds for j in axes(cellL.h0, 2)
            fw = @view face.fw[:, j, :]
            fh0 = @view face.fh0[:, j, :]
            fh1 = @view face.fh1[:, j, :]
            fh2 = @view face.fh2[:, j, :]
            fh3 = @view face.fh3[:, j, :]

            flux_kfvs!(
                fw,
                fh0,
                fh1,
                fh2,
                fh3,
                cellL.h0[:, j, :] .+ 0.5 .* dxL .* cellL.sh0[:, j, :],
                cellL.h1[:, j, :] .+ 0.5 .* dxL .* cellL.sh1[:, j, :],
                cellL.h2[:, j, :] .+ 0.5 .* dxL .* cellL.sh2[:, j, :],
                cellL.h3[:, j, :] .+ 0.5 .* dxL .* cellL.sh3[:, j, :],
                cellR.h0[:, j, :] .- 0.5 .* dxR .* cellR.sh0[:, j, :],
                cellR.h1[:, j, :] .- 0.5 .* dxR .* cellR.sh1[:, j, :],
                cellR.h2[:, j, :] .- 0.5 .* dxR .* cellR.sh2[:, j, :],
                cellR.h3[:, j, :] .- 0.5 .* dxR .* cellR.sh3[:, j, :],
                KS.vs.u,
                KS.vs.weights,
                dt,
                cellL.sh0[:, j, :],
                cellL.sh1[:, j, :],
                cellL.sh2[:, j, :],
                cellL.sh3[:, j, :],
                cellR.sh0[:, j, :],
                cellR.sh1[:, j, :],
                cellR.sh2[:, j, :],
                cellR.sh3[:, j, :],
            )
        end

    elseif mode == :kcu

        @inbounds for j in axes(cellL.h0, 2)
            fw = @view face.fw[:, j, :]
            fh0 = @view face.fh0[:, j, :]
            fh1 = @view face.fh1[:, j, :]
            fh2 = @view face.fh2[:, j, :]
            fh3 = @view face.fh3[:, j, :]

            flux_kcu!(
                fw,
                fh0,
                fh1,
                fh2,
                fh3,
                cellL.w[:, j, :] .+ 0.5 .* dxL .* cellL.sw[:, j, :],
                cellL.h0[:, j, :] .+ 0.5 .* dxL .* cellL.sh0[:, j, :],
                cellL.h1[:, j, :] .+ 0.5 .* dxL .* cellL.sh1[:, j, :],
                cellL.h2[:, j, :] .+ 0.5 .* dxL .* cellL.sh2[:, j, :],
                cellL.h3[:, j, :] .+ 0.5 .* dxL .* cellL.sh3[:, j, :],
                cellR.w[:, j, :] .- 0.5 .* dxR .* cellR.sw[:, j, :],
                cellR.h0[:, j, :] .- 0.5 .* dxR .* cellR.sh0[:, j, :],
                cellR.h1[:, j, :] .- 0.5 .* dxR .* cellR.sh1[:, j, :],
                cellR.h2[:, j, :] .- 0.5 .* dxR .* cellR.sh2[:, j, :],
                cellR.h3[:, j, :] .- 0.5 .* dxR .* cellR.sh3[:, j, :],
                KS.vs.u,
                KS.vs.weights,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                dt,
                isMHD,
            )
        end

    else

        throw("flux mode not available")

    end # if

end


function uqflux_flow_galerkin!(
    KS::SolverSet,
    uq::AbstractUQ,
    cellL::ControlVolume1D3F,
    face::Interface1D3F,
    cellR::ControlVolume1D3F,
    dt::AbstractFloat,
    dxL,
    dxR;
    mode = :kfvs::Symbol,
    isMHD = false::Bool,
)

    if mode == :kfvs

        @inbounds for k in axes(cellL.h0, 4)
            for j in axes(cellL.h0, 3)
                fw = @view face.fw[:, j, k]
                fh0 = @view face.fh0[:, :, j, k]
                fh1 = @view face.fh1[:, :, j, k]
                fh2 = @view face.fh2[:, :, j, k]

                flux_kfvs!(
                    fw,
                    fh0,
                    fh1,
                    fh2,
                    cellL.h0[:, :, j, k] .+ 0.5 .* dxL .* cellL.sh0[:, :, j, k],
                    cellL.h1[:, :, j, k] .+ 0.5 .* dxL .* cellL.sh1[:, :, j, k],
                    cellL.h2[:, :, j, k] .+ 0.5 .* dxL .* cellL.sh2[:, :, j, k],
                    cellR.h0[:, :, j, k] .- 0.5 .* dxR .* cellR.sh0[:, :, j, k],
                    cellR.h1[:, :, j, k] .- 0.5 .* dxR .* cellR.sh1[:, :, j, k],
                    cellR.h2[:, :, j, k] .- 0.5 .* dxR .* cellR.sh2[:, :, j, k],
                    KS.vs.u[:, :, k],
                    KS.vs.v[:, :, k],
                    KS.vs.weights[:, :, k],
                    dt,
                    1.0,
                    cellL.sh0[:, :, j, k],
                    cellL.sh1[:, :, j, k],
                    cellL.sh2[:, :, j, k],
                    cellR.sh0[:, :, j, k],
                    cellR.sh1[:, :, j, k],
                    cellR.sh2[:, :, j, k],
                )
            end
        end

    elseif mode == :kcu

        fw = chaos_ran(face.fw, 2, uq)
        fh0 = chaos_ran(face.fh0, 3, uq)
        fh1 = chaos_ran(face.fh1, 3, uq)
        fh2 = chaos_ran(face.fh2, 3, uq)

        wL = chaos_ran(cellL.w .+ 0.5 .* dxL .* cellL.sw, 2, uq)
        h0L = chaos_ran(cellL.h0 .+ 0.5 .* dxL .* cellL.sh0, 3, uq)
        h1L = chaos_ran(cellL.h1 .+ 0.5 .* dxL .* cellL.sh1, 3, uq)
        h2L = chaos_ran(cellL.h2 .+ 0.5 .* dxL .* cellL.sh2, 3, uq)

        wR = chaos_ran(cellR.w .- 0.5 .* dxR .* cellR.sw, 2, uq)
        h0R = chaos_ran(cellR.h0 .- 0.5 .* dxR .* cellR.sh0, 3, uq)
        h1R = chaos_ran(cellR.h1 .- 0.5 .* dxR .* cellR.sh1, 3, uq)
        h2R = chaos_ran(cellR.h2 .- 0.5 .* dxR .* cellR.sh2, 3, uq)

        @inbounds for j in axes(h0L, 3)
            _fw = @view fw[:, j, :]
            _fh0 = @view fh0[:, :, j, :]
            _fh1 = @view fh1[:, :, j, :]
            _fh2 = @view fh2[:, :, j, :]

            flux_kcu!(
                _fw,
                _fh0,
                _fh1,
                _fh2,
                wL[:, j, :],
                h0L[:, :, j, :],
                h1L[:, :, j, :],
                h2L[:, :, j, :],
                wR[:, j, :],
                h0R[:, :, j, :],
                h1R[:, :, j, :],
                h2R[:, :, j, :],
                KS.vs.u,
                KS.vs.v,
                KS.vs.weights,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                dt,
                1.0,
                isMHD,
            )
        end

        face.fw .= chaos_ran(fw, 2, uq)
        face.fh0 .= chaos_ran(fh0, 3, uq)
        face.fh1 .= chaos_ran(fh1, 3, uq)
        face.fh2 .= chaos_ran(fh2, 3, uq)

    elseif mode == :ugks

        fw = chaos_ran(face.fw, 2, uq)
        fh0 = chaos_ran(face.fh0, 3, uq)
        fh1 = chaos_ran(face.fh1, 3, uq)
        fh2 = chaos_ran(face.fh2, 3, uq)

        wL = chaos_ran(cellL.w .+ 0.5 .* dxL .* cellL.sw, 2, uq)
        h0L = chaos_ran(cellL.h0 .+ 0.5 .* dxL .* cellL.sh0, 3, uq)
        h1L = chaos_ran(cellL.h1 .+ 0.5 .* dxL .* cellL.sh1, 3, uq)
        h2L = chaos_ran(cellL.h2 .+ 0.5 .* dxL .* cellL.sh2, 3, uq)
        sh0L = chaos_ran(cellL.sh0, 3, uq)
        sh1L = chaos_ran(cellL.sh1, 3, uq)
        sh2L = chaos_ran(cellL.sh2, 3, uq)

        wR = chaos_ran(cellR.w .- 0.5 .* dxR .* cellR.sw, 2, uq)
        h0R = chaos_ran(cellR.h0 .- 0.5 .* dxR .* cellR.sh0, 3, uq)
        h1R = chaos_ran(cellR.h1 .- 0.5 .* dxR .* cellR.sh1, 3, uq)
        h2R = chaos_ran(cellR.h2 .- 0.5 .* dxR .* cellR.sh2, 3, uq)
        sh0R = chaos_ran(cellR.sh0, 3, uq)
        sh1R = chaos_ran(cellR.sh1, 3, uq)
        sh2R = chaos_ran(cellR.sh2, 3, uq)

        @inbounds for j in axes(h0L, 3)
            _fw = @view face.fw[:, j, :]
            _fh0 = @view face.fh0[:, :, j, :]
            _fh1 = @view face.fh1[:, :, j, :]
            _fh2 = @view face.fh2[:, :, j, :]

            flux_ugks!(
                _fw,
                _fh0,
                _fh1,
                _fh2,
                wL[:, j, :],
                h0L[:, :, j, :],
                h1L[:, :, j, :],
                h2L[:, :, j, :],
                wR[:, j, :],
                h0R[:, :, j, :],
                h1R[:, :, j, :],
                h2R[:, :, j, :],
                KS.vs.u,
                KS.vs.v,
                KS.vs.weights,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                dt,
                0.5 * dxL,
                0.5 * dxR,
                1.0,
                sh0L[:, :, j, :],
                sh1L[:, :, j, :],
                sh2L[:, :, j, :],
                sh0R[:, :, j, :],
                sh1R[:, :, j, :],
                sh2R[:, :, j, :],
            )
        end

        face.fw .= chaos_ran(fw, 2, uq)
        face.fh0 .= chaos_ran(fh0, 3, uq)
        face.fh1 .= chaos_ran(fh1, 3, uq)
        face.fh2 .= chaos_ran(fh2, 3, uq)

    else

        throw("flux mode not available")

    end # if

end


function uqflux_flow_collocation!(
    KS::SolverSet,
    cellL::ControlVolume1D3F,
    face::Interface1D3F,
    cellR::ControlVolume1D3F,
    dt::AbstractFloat,
    dxL,
    dxR;
    mode = :kfvs::Symbol,
    isMHD = false::Bool,
)

    if mode == :kfvs

        @inbounds for k in axes(cellL.h0, 4)
            for j in axes(cellL.h0, 3)
                fw = @view face.fw[:, j, k]
                fh0 = @view face.fh0[:, :, j, k]
                fh1 = @view face.fh1[:, :, j, k]
                fh2 = @view face.fh2[:, :, j, k]

                flux_kfvs!(
                    fw,
                    fh0,
                    fh1,
                    fh2,
                    cellL.h0[:, :, j, k] .+ 0.5 .* dxL .* cellL.sh0[:, :, j, k],
                    cellL.h1[:, :, j, k] .+ 0.5 .* dxL .* cellL.sh1[:, :, j, k],
                    cellL.h2[:, :, j, k] .+ 0.5 .* dxL .* cellL.sh2[:, :, j, k],
                    cellR.h0[:, :, j, k] .- 0.5 .* dxR .* cellR.sh0[:, :, j, k],
                    cellR.h1[:, :, j, k] .- 0.5 .* dxR .* cellR.sh1[:, :, j, k],
                    cellR.h2[:, :, j, k] .- 0.5 .* dxR .* cellR.sh2[:, :, j, k],
                    KS.vs.u[:, :, k],
                    KS.vs.v[:, :, k],
                    KS.vs.weights[:, :, k],
                    dt,
                    1.0,
                    cellL.sh0[:, :, j, k],
                    cellL.sh1[:, :, j, k],
                    cellL.sh2[:, :, j, k],
                    cellR.sh0[:, :, j, k],
                    cellR.sh1[:, :, j, k],
                    cellR.sh2[:, :, j, k],
                )
            end
        end

    elseif mode == :kcu

        @inbounds for j in axes(cellL.h0, 3)
            fw = @view face.fw[:, j, :]
            fh0 = @view face.fh0[:, :, j, :]
            fh1 = @view face.fh1[:, :, j, :]
            fh2 = @view face.fh2[:, :, j, :]

            flux_kcu!(
                fw,
                fh0,
                fh1,
                fh2,
                cellL.w[:, j, :] .+ 0.5 .* dxL .* cellL.sw[:, j, :],
                cellL.h0[:, :, j, :] .+ 0.5 .* dxL .* cellL.sh0[:, :, j, :],
                cellL.h1[:, :, j, :] .+ 0.5 .* dxL .* cellL.sh1[:, :, j, :],
                cellL.h2[:, :, j, :] .+ 0.5 .* dxL .* cellL.sh2[:, :, j, :],
                cellR.w[:, j, :] .- 0.5 .* dxR .* cellR.sw[:, j, :],
                cellR.h0[:, :, j, :] .- 0.5 .* dxR .* cellR.sh0[:, :, j, :],
                cellR.h1[:, :, j, :] .- 0.5 .* dxR .* cellR.sh1[:, :, j, :],
                cellR.h2[:, :, j, :] .- 0.5 .* dxR .* cellR.sh2[:, :, j, :],
                KS.vs.u[:, :, :],
                KS.vs.v[:, :, :],
                KS.vs.weights[:, :, :],
                KS.gas.K,
                KS.gas.γ,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                dt,
                1.0,
                isMHD,
            )
        end

    elseif mode == :ugks

        @inbounds for j in axes(cellL.h0, 3)
            fw = @view face.fw[:, j, :]
            fh0 = @view face.fh0[:, :, j, :]
            fh1 = @view face.fh1[:, :, j, :]
            fh2 = @view face.fh2[:, :, j, :]

            flux_ugks!(
                fw,
                fh0,
                fh1,
                fh2,
                cellL.w[:, j, :] .+ 0.5 .* dxL .* cellL.sw[:, j, :],
                cellL.h0[:, :, j, :] .+ 0.5 .* dxL .* cellL.sh0[:, :, j, :],
                cellL.h1[:, :, j, :] .+ 0.5 .* dxL .* cellL.sh1[:, :, j, :],
                cellL.h2[:, :, j, :] .+ 0.5 .* dxL .* cellL.sh2[:, :, j, :],
                cellR.w[:, j, :] .- 0.5 .* dxR .* cellR.sw[:, j, :],
                cellR.h0[:, :, j, :] .- 0.5 .* dxR .* cellR.sh0[:, :, j, :],
                cellR.h1[:, :, j, :] .- 0.5 .* dxR .* cellR.sh1[:, :, j, :],
                cellR.h2[:, :, j, :] .- 0.5 .* dxR .* cellR.sh2[:, :, j, :],
                KS.vs.u[:, :, :],
                KS.vs.v[:, :, :],
                KS.vs.weights[:, :, :],
                KS.gas.K,
                KS.gas.γ,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                dt,
                0.5 * dxL,
                0.5 * dxR,
                1.0,
                cellL.sh0[:, :, j, :],
                cellL.sh1[:, :, j, :],
                cellL.sh2[:, :, j, :],
                cellR.sh0[:, :, j, :],
                cellR.sh1[:, :, j, :],
                cellR.sh2[:, :, j, :],
            )
        end

    else

        throw("flux mode not available")

    end # if

end


"""
Calculate flux of electromagnetic propagation

"""
function uqflux_em!(
    KS::SolverSet,
    uq::AbstractUQ,
    cellLL::AbstractControlVolume1D,
    cellL::AbstractControlVolume1D,
    face::AbstractInterface1D,
    cellR::AbstractControlVolume1D,
    cellRR::AbstractControlVolume1D,
    dt::Real,
    dxL,
    dxR,
)

    if uq.method == "collocation"

        for j = 1:uq.op.quad.Nquad
            femL = @view face.femL[:, j]
            femR = @view face.femR[:, j]

            flux_em!(
                femL,
                femR,
                cellLL.E[:, j],
                cellLL.B[:, j],
                cellL.E[:, j],
                cellL.B[:, j],
                cellR.E[:, j],
                cellR.B[:, j],
                cellRR.E[:, j],
                cellRR.B[:, j],
                cellL.ϕ[j],
                cellR.ϕ[j],
                cellL.ψ[j],
                cellR.ψ[j],
                dxL,
                dxR,
                KS.gas.Ap,
                KS.gas.An,
                KS.gas.D,
                KS.gas.sol,
                KS.gas.χ,
                KS.gas.ν,
                dt,
            )
        end

    elseif uq.method == "galerkin"

        ELL = chaos_ran(cellLL.E, 2, uq)
        BLL = chaos_ran(cellLL.B, 2, uq)
        EL = chaos_ran(cellL.E, 2, uq)
        BL = chaos_ran(cellL.B, 2, uq)
        ER = chaos_ran(cellR.E, 2, uq)
        BR = chaos_ran(cellR.B, 2, uq)
        ERR = chaos_ran(cellRR.E, 2, uq)
        BRR = chaos_ran(cellRR.B, 2, uq)
        ϕL = chaos_ran(cellL.ϕ, uq)
        ϕR = chaos_ran(cellR.ϕ, uq)
        ψL = chaos_ran(cellL.ψ, uq)
        ψR = chaos_ran(cellR.ψ, uq)

        femLRan = zeros(8, uq.op.quad.Nquad)
        femRRan = similar(femLRan)
        for j = 1:uq.op.quad.Nquad
            femL = @view femLRan[:, j]
            femR = @view femRRan[:, j]
            flux_em!(
                femL,
                femR,
                ELL[:, j],
                BLL[:, j],
                EL[:, j],
                BL[:, j],
                ER[:, j],
                BR[:, j],
                ERR[:, j],
                BRR[:, j],
                ϕL[j],
                ϕR[j],
                ψL[j],
                ψR[j],
                dxL,
                dxR,
                KS.gas.Ap,
                KS.gas.An,
                KS.gas.D,
                KS.gas.sol,
                KS.gas.χ,
                KS.gas.ν,
                dt,
            )
        end

        face.femL .= ran_chaos(femLRan, 2, uq)
        face.femR .= ran_chaos(femRRan, 2, uq)

    end

end
