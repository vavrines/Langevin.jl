# ============================================================
# Methods of Flux
# ============================================================


"""
Kinetic flux vector splitting (KFVS) method

"""

function calc_flux_kfvs!(
    KS::SolverSet,
    sol::Solution1D1F,
    flux::Flux1D1F,
    dt::AbstractFloat,
    order = 1::Int,
)

    for i in eachindex(flux.fw)
        for j in axes(sol.w[1], 2) # over gPC coefficients or quadrature points
            flux.fw[i][:,j], flux.ff[i][:,j] = Kinetic.flux_kfvs(
                sol.f[i-1][:,j],
                sol.f[i][:,j],
                KS.vSpace.u,
                KS.vSpace.weights,
                dt,
            )
        end
    end

end

function calc_flux_kfvs!(
    KS::SolverSet,
    cellL::ControlVolume1D1F,
    face::Interface1D1F,
    cellR::ControlVolume1D1F,
    dt::AbstractFloat,
    order = 1::Int,
)

    if ndims(cellL.f) == 2

        if order == 1 # first order accuracy
            for j in axes(cellL.f, 2)
                fw, ff = Kinetic.flux_kfvs(
                    cellL.f[:, j],
                    cellR.f[:, j],
                    KS.vSpace.u,
                    KS.vSpace.weights,
                    dt,
                )

                face.fw[:, j] .= fw
                face.ff[:, j] .= ff
            end
        elseif order == 2 # second order accuracy
            for j in axes(cellL.f, 2)
                fw, ff = Kinetic.flux_kfvs(
                    cellL.f[:, j] .+ 0.5 .* cellL.dx .* cellL.sf[:, j],
                    cellR.f[:, j] .- 0.5 .* cellR.dx .* cellR.sf[:, j],
                    KS.vSpace.u,
                    KS.vSpace.weights,
                    dt,
                    cellL.sf[:, j],
                    cellR.sf[:, j],
                )

                face.fw[:, j] .= fw
                face.ff[:, j] .= ff
            end
        end

    elseif ndims(cellL.f) == 3

        if order == 1 # first order accuracy
            for k in axes(cellL.f, 3), j in axes(cellL.f, 2)
                fw, ff = Kinetic.flux_kfvs(
                    cellL.f[:, j, k],
                    cellR.f[:, j, k],
                    KS.vSpace.u[:, k],
                    KS.vSpace.weights[:, k],
                    dt,
                )

                face.fw[:, j, k] .= fw
                face.ff[:, j, k] .= ff
            end
        elseif order == 2 # second order accuracy
            for k in axes(cellL.f, 3), j in axes(cellL.f, 2)
                fw, ff = Kinetic.flux_kfvs(
                    cellL.f[:, j, k] .+ 0.5 .* cellL.dx .* cellL.sf[:, j, k],
                    cellR.f[:, j, k] .- 0.5 .* cellR.dx .* cellR.sf[:, j, k],
                    KS.vSpace.u[:,k],
                    KS.vSpace.weights[:,k],
                    dt,
                    cellL.sf[:, j, k],
                    cellR.sf[:, j, k],
                )

                face.fw[:, j, k] .= fw
                face.ff[:, j, k] .= ff
            end
        end

    else

        throw(DimensionMismatch("distribution function in KFVS flux"))

    end

end
