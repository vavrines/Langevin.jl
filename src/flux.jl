# ============================================================
# Methods of Flux
# ============================================================


"""
Kinetic flux vector splitting (KFVS) method

"""

# ------------------------------------------------------------
# 1D case
# ------------------------------------------------------------
function calc_flux_kfvs!(
    KS::SolverSet,
    sol::Solution1D1F,
    flux::Flux1D1F,
    dt::AbstractFloat,
    order = 1::Int,
)

    Threads.@threads for i in eachindex(flux.fw)
        for j in axes(sol.w[1], 2) # over gPC coefficients or quadrature points
            flux.fw[i][:, j], flux.ff[i][:, j] = Kinetic.flux_kfvs(
                sol.f[i-1][:, j],
                sol.f[i][:, j],
                KS.vSpace.u,
                KS.vSpace.weights,
                dt,
            )
        end
    end

end

# ------------------------------------------------------------
# 2D case
# ------------------------------------------------------------
function calc_flux_kfvs!(
    KS::SolverSet,
    sol::Solution2D2F,
    flux::Flux2D2F,
    dt::AbstractFloat,
    order = 1::Int,
)

    Threads.@threads for j = 1:KS.pSpace.ny
        for i = 1:KS.pSpace.nx+1
            vn =
                KS.vSpace.u .* flux.n1[i, j][1] .+
                KS.vSpace.v .* flux.n1[i, j][2]
            vt =
                KS.vSpace.v .* flux.n1[i, j][1] .-
                KS.vSpace.u .* flux.n1[i, j][2]

            @inbounds for k in axes(sol.w[1, 1], 2)
                fw, flux.fh1[i, j][:, :, k], flux.fb1[i, j][:, :, k] =
                    flux_kfvs(
                        sol.h[i-1, j][:, :, k],
                        sol.b[i-1, j][:, :, k],
                        sol.h[i, j][:, :, k],
                        sol.b[i, j][:, :, k],
                        vn,
                        vt,
                        KS.vSpace.weights,
                        dt,
                        0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                    )
                flux.fw1[i, j][:, k] =
                    global_frame(fw, flux.n1[i, j][1], flux.n1[i, j][2])
            end
        end
    end

    Threads.@threads for j = 1:KS.pSpace.ny+1
        for i = 1:KS.pSpace.nx
            vn =
                KS.vSpace.u .* flux.n2[i, j][1] .+
                KS.vSpace.v .* flux.n2[i, j][2]
            vt =
                KS.vSpace.v .* flux.n2[i, j][1] .-
                KS.vSpace.u .* flux.n2[i, j][2]

            @inbounds for k in axes(sol.w[1, 1], 2)
                fw, flux.fh2[i, j][:, :, k], flux.fb2[i, j][:, :, k] =
                    flux_kfvs(
                        sol.h[i, j-1][:, :, k],
                        sol.b[i, j-1][:, :, k],
                        sol.h[i, j][:, :, k],
                        sol.b[i, j][:, :, k],
                        vn,
                        vt,
                        KS.vSpace.weights,
                        dt,
                        0.5 * (KS.pSpace.dx[i, j-1] + KS.pSpace.dx[i, j]),
                    )
                flux.fw2[i, j][:, k] =
                    global_frame(fw, flux.n2[i, j][1], flux.n2[i, j][2])
            end
        end
    end

end


function evolve!(
    KS::SolverSet,
    sol::Solution2D2F,
    flux::Flux2D2F,
    dt::AbstractFloat,
    order = 1::Int,
)

    Threads.@threads for j = 1:KS.pSpace.ny
        for i = 1:KS.pSpace.nx+1
            @inbounds for k in axes(sol.w[1, 1], 2)
                calc_flux_kfvs!(
                    KS,
                    sol.h[i-1, j],
                    sol.b[i-1, j],
                    sol.h[i, j],
                    sol.b[i, j],
                    flux.fw1[i, j],
                    flux.fh1[i, j],
                    flux.fb1[i, j],
                    KS.vSpace.u,
                    KS.vSpace.v,
                    KS.vSpace.weights,
                    flux.n1[i, j],
                    dt,
                    0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                    1,
                )
            end
        end
    end

    Threads.@threads for j = 1:KS.pSpace.ny+1
        for i = 1:KS.pSpace.nx
            @inbounds for k in axes(sol.w[1, 1], 2)
                calc_flux_kfvs!(
                    KS,
                    sol.h[i, j-1],
                    sol.b[i, j-1],
                    sol.h[i, j],
                    sol.b[i, j],
                    flux.fw2[i, j],
                    flux.fh2[i, j],
                    flux.fb2[i, j],
                    KS.vSpace.u,
                    KS.vSpace.v,
                    KS.vSpace.weights,
                    flux.n2[i, j],
                    dt,
                    0.5 * (KS.pSpace.dx[i, j-1] + KS.pSpace.dx[i, j]),
                    1,
                )
            end
        end
    end

end

function calc_flux_kfvs!(
    KS::SolverSet,
    hL::AbstractArray{Float64,3},
    bL::AbstractArray{Float64,3},
    hR::AbstractArray{Float64,3},
    bR::AbstractArray{Float64,3},
    fw::Array{Float64,2},
    fh::AbstractArray{Float64,3},
    fb::AbstractArray{Float64,3},
    u::AbstractArray{Float64,2},
    v::AbstractArray{Float64,2},
    weights::AbstractArray{Float64,2},
    n::Array{Float64,1},
    dt::Float64,
    dx::Float64,
    order = 1::Int,
)

    vn = u .* n[1] .+ v .* n[2]
    vt = v .* n[1] .- u .* n[2]

    for k in axes(fw, 2)
        fw_local, fh[:, :, k], fb[:, :, k] =
            flux_kfvs(
                hL[:, :, k],
                bL[:, :, k],
                hR[:, :, k],
                bR[:, :, k],
                vn,
                vt,
                weights,
                dt,
                dx,
            )
        fw[:, k] .= global_frame(fw_local, n[1], n[2])
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
            Threads.@threads for j in axes(cellL.f, 2)
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
            Threads.@threads for j in axes(cellL.f, 2)
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
            Threads.@threads for k in axes(cellL.f, 3)
                for j in axes(cellL.f, 2)
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
            end
        elseif order == 2 # second order accuracy
            Threads.@threads for k in axes(cellL.f, 3)
                for j in axes(cellL.f, 2)
                    fw, ff = Kinetic.flux_kfvs(
                        cellL.f[:, j, k] .+
                        0.5 .* cellL.dx .* cellL.sf[:, j, k],
                        cellR.f[:, j, k] .-
                        0.5 .* cellR.dx .* cellR.sf[:, j, k],
                        KS.vSpace.u[:, k],
                        KS.vSpace.weights[:, k],
                        dt,
                        cellL.sf[:, j, k],
                        cellR.sf[:, j, k],
                    )

                    face.fw[:, j, k] .= fw
                    face.ff[:, j, k] .= ff
                end
            end
        end

    else

        throw(DimensionMismatch("distribution function in KFVS flux"))

    end

end


"""
Maxwell's diffusive boundary flux

"""

function calc_flux_boundary_maxwell!(
    bc::Array,
    KS::SolverSet,
    sol::Solution2D2F,
    flux::Flux2D2F,
    dt::AbstractFloat,
)

    Threads.@threads for j = 1:KS.pSpace.ny
        un = KS.vSpace.u .* flux.n1[1, j][1] .+ KS.vSpace.v .* flux.n1[1, j][2]
        ut = KS.vSpace.v .* flux.n1[1, j][1] .- KS.vSpace.u .* flux.n1[1, j][2]

        #bcL = local_frame(KS.ib.bcL, flux.n1[1, j][1], flux.n1[1, j][2])
        #bcR = local_frame(KS.ib.bcR, flux.n1[1, j][1], flux.n1[1, j][2])

        for k in axes(sol.w[1, 1], 2)
            bcL = local_frame(bc[1][:, k], flux.n1[1, j][1], flux.n1[1, j][2])
            bcR = local_frame(bc[2][:, k], flux.n1[1, j][1], flux.n1[1, j][2])

            fw, flux.fh1[1, j][:, :, k], flux.fb1[1, j][:, :, k] =
                flux_boundary_maxwell(
                    bcL, # left
                    sol.h[1, j][:, :, k],
                    sol.b[1, j][:, :, k],
                    un,
                    ut,
                    KS.vSpace.weights,
                    KS.gas.K,
                    dt,
                    KS.pSpace.dy[1, j],
                    1,
                )
            flux.fw1[1, j][:, k] .=
                global_frame(fw, flux.n1[1, j][1], flux.n1[1, j][2])

            fw, flux.fh1[end, j][:, :, k], flux.fb1[end, j][:, :, k] =
                flux_boundary_maxwell(
                    bcR, # right
                    sol.h[KS.pSpace.nx, j][:, :, k],
                    sol.b[KS.pSpace.nx, j][:, :, k],
                    un,
                    ut,
                    KS.vSpace.weights,
                    KS.gas.K,
                    dt,
                    KS.pSpace.dy[KS.pSpace.nx, j],
                    -1,
                )
            flux.fw1[end, j][:, k] .=
                global_frame(fw, flux.n1[end, j][1], flux.n1[end, j][2])
        end
    end

    Threads.@threads for i = 1:KS.pSpace.nx
        vn = KS.vSpace.u .* flux.n2[i, 1][1] .+ KS.vSpace.v .* flux.n2[i, 1][2]
        vt = KS.vSpace.v .* flux.n2[i, 1][1] .- KS.vSpace.u .* flux.n2[i, 1][2]

        #bcU = local_frame(KS.ib.bcU, flux.n2[i, 1][1], flux.n2[i, 1][2])
        #bcD = local_frame(KS.ib.bcD, flux.n2[i, 1][1], flux.n2[i, 1][2])

        for k in axes(sol.w[1, 1], 2)
            bcU = local_frame(bc[3][:, k], flux.n2[i, 1][1], flux.n2[i, 1][2])
            bcD = local_frame(bc[4][:, k], flux.n2[i, 1][1], flux.n2[i, 1][2])

            fw, flux.fh2[i, 1][:, :, k], flux.fb2[i, 1][:, :, k] =
                flux_boundary_maxwell(
                    bcD, # down
                    sol.h[i, 1][:, :, k],
                    sol.b[i, 1][:, :, k],
                    vn,
                    vt,
                    KS.vSpace.weights,
                    KS.gas.K,
                    dt,
                    KS.pSpace.dx[i, 1],
                    1,
                )
            flux.fw2[i, 1][:, k] .=
                global_frame(fw, flux.n2[i, 1][1], flux.n2[i, 1][2])

            fw, flux.fh2[i, end][:, :, k], flux.fb2[i, end][:, :, k] =
                flux_boundary_maxwell(
                    bcU, # down
                    sol.h[i, KS.pSpace.ny][:, :, k],
                    sol.b[i, KS.pSpace.ny][:, :, k],
                    vn,
                    vt,
                    KS.vSpace.weights,
                    KS.gas.K,
                    dt,
                    KS.pSpace.dx[i, KS.pSpace.ny],
                    -1,
                )
            flux.fw2[i, end][:, k] .=
                global_frame(fw, flux.n2[i, end][1], flux.n2[i, end][2])
        end

    end

end
