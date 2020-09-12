# ============================================================
# Flux Functions
# ============================================================


"""
Evolution of particle transport

* kinetic flux vector splitting (KFVS)
* kinetic central-upwind (KCU)
* unified gas-kinetic scheme (UGKS)

"""
function evolve!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution1D1F,
    flux::Flux1D1F,
    dt::AbstractFloat;
    mode = :kfvs::Symbol,
)

    if mode == :kfvs
        @inbounds Threads.@threads for i in eachindex(flux.fw)
            for j in axes(sol.w[1], 2) # over gPC coefficients or quadrature points
                fw = @view flux.fw[i][:, j]
                ff = @view flux.ff[i][:, j]

                Kinetic.flux_kfvs!(
                    fw,
                    ff,
                    sol.f[i-1][:, j] .+ 0.5 .* KS.pSpace.dx[i-1] .* sol.sf[i-1][:, j],
                    sol.f[i][:, j] .- 0.5 .* KS.pSpace.dx[i] .* sol.sf[i][:, j],
                    KS.vSpace.u,
                    KS.vSpace.weights,
                    dt,
                    sol.sf[i-1][:, j],
                    sol.sf[i][:, j],
                )
            end
        end
    else
        throw("flux mode isn't available")
    end

end


function evolve!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution1D2F,
    flux::Flux1D2F,
    dt::AbstractFloat;
    mode = :kfvs::Symbol,
)

    if mode == :kfvs
        @inbounds Threads.@threads for i in eachindex(flux.fw)
            for j in axes(sol.w[1], 2) # over gPC coefficients or quadrature points
                fw = @view flux.fw[i][:, j]
                fh = @view flux.fh[i][:, j]
                fb = @view flux.fb[i][:, j]

                Kinetic.flux_kfvs!(
                    fw,
                    fh,
                    fb,
                    sol.h[i-1][:, j] .+ 0.5 .* KS.pSpace.dx[i-1] .* sol.sh[i-1][:, j],
                    sol.b[i-1][:, j] .+ 0.5 .* KS.pSpace.dx[i-1] .* sol.sb[i-1][:, j],
                    sol.h[i][:, j] .- 0.5 .* KS.pSpace.dx[i] .* sol.sh[i][:, j],
                    sol.b[i][:, j] .- 0.5 .* KS.pSpace.dx[i] .* sol.sb[i][:, j],
                    KS.vSpace.u,
                    KS.vSpace.weights,
                    dt,
                    sol.sh[i-1][:, j],
                    sol.sb[i-1][:, j],
                    sol.sh[i][:, j],
                    sol.sb[i][:, j],
                )
            end
        end
    else
        throw("flux mode isn't available")
    end

end


#--- 2D case ---#
function evolve!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution2D2F,
    flux::Flux2D2F,
    dt::AbstractFloat;
    mode = :kfvs::Symbol,
)

    if mode == :kfvs

        @inbounds for j = 1:KS.pSpace.ny
            for i = 1:KS.pSpace.nx+1
                un = KS.vSpace.u .* flux.n1[i, j][1] .+ KS.vSpace.v .* flux.n1[i, j][2]
                ut = KS.vSpace.v .* flux.n1[i, j][1] .- KS.vSpace.u .* flux.n1[i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw1 = @view flux.fw1[i, j][:, k]
                    fh1 = @view flux.fh1[i, j][:, :, k]
                    fb1 = @view flux.fb1[i, j][:, :, k]

                    flux_kfvs!(
                        fw1,
                        fh1,
                        fb1,
                        sol.h[i-1, j][:, :, k] .+
                        0.5 .* KS.pSpace.dx[i-1, j] .* sol.sh[i-1, j][:, :, k, 1],
                        sol.b[i-1, j][:, :, k] .+
                        0.5 .* KS.pSpace.dx[i-1, j] .* sol.sb[i-1, j][:, :, k, 1],
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dx[i, j] .* sol.sh[i, j][:, :, k, 1],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dx[i, j] .* sol.sb[i, j][:, :, k, 1],
                        un,
                        ut,
                        KS.vSpace.weights,
                        dt,
                        0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                        sol.sh[i-1, j][:, :, k, 1],
                        sol.sb[i-1, j][:, :, k, 1],
                        sol.sh[i, j][:, :, k, 1],
                        sol.sb[i, j][:, :, k, 1],
                    )
                    flux.fw1[i, j][:, k] .=
                        global_frame(fw1, flux.n1[i, j][1], flux.n1[i, j][2])
                end
            end
        end

        @inbounds for j = 1:KS.pSpace.ny+1
            for i = 1:KS.pSpace.nx
                vn = KS.vSpace.u .* flux.n2[i, j][1] .+ KS.vSpace.v .* flux.n2[i, j][2]
                vt = KS.vSpace.v .* flux.n2[i, j][1] .- KS.vSpace.u .* flux.n2[i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw2 = @view flux.fw2[i, j][:, k]
                    fh2 = @view flux.fh2[i, j][:, :, k]
                    fb2 = @view flux.fb2[i, j][:, :, k]

                    flux_kfvs!(
                        fw2,
                        fh2,
                        fb2,
                        sol.h[i, j-1][:, :, k] .+
                        0.5 .* KS.pSpace.dy[i, j-1] .* sol.sh[i, j-1][:, :, k, 2],
                        sol.b[i, j-1][:, :, k] .+
                        0.5 .* KS.pSpace.dy[i, j-1] .* sol.sb[i, j-1][:, :, k, 2],
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dy[i, j] .* sol.sh[i, j][:, :, k, 2],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dy[i, j] .* sol.sb[i, j][:, :, k, 2],
                        vn,
                        vt,
                        KS.vSpace.weights,
                        dt,
                        0.5 * (KS.pSpace.dx[i, j-1] + KS.pSpace.dx[i, j]),
                        sol.sh[i, j-1][:, :, k, 2],
                        sol.sb[i, j-1][:, :, k, 2],
                        sol.sh[i, j][:, :, k, 2],
                        sol.sb[i, j][:, :, k, 2],
                    )
                    flux.fw2[i, j][:, k] .=
                        global_frame(fw2, flux.n2[i, j][1], flux.n2[i, j][2])
                end
            end
        end

    elseif mode == :kcu

        @inbounds for j = 1:KS.pSpace.ny
            for i = 1:KS.pSpace.nx+1
                un = KS.vSpace.u .* flux.n1[i, j][1] .+ KS.vSpace.v .* flux.n1[i, j][2]
                ut = KS.vSpace.v .* flux.n1[i, j][1] .- KS.vSpace.u .* flux.n1[i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw1 = @view flux.fw1[i, j][:, k]
                    fh1 = @view flux.fh1[i, j][:, :, k]
                    fb1 = @view flux.fb1[i, j][:, :, k]

                    flux_kcu!(
                        fw1,
                        fh1,
                        fb1,
                        local_frame(
                            sol.w[i-1, j][:, k] .+
                            0.5 .* KS.pSpace.dx[i-1, j] .* sol.sw[i-1, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i-1, j][:, :, k] .+
                        0.5 .* KS.pSpace.dx[i-1, j] .* sol.sh[i-1, j][:, :, k, 1],
                        sol.b[i-1, j][:, :, k] .+
                        0.5 .* KS.pSpace.dx[i-1, j] .* sol.sb[i-1, j][:, :, k, 1],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.pSpace.dx[i, j] .* sol.sw[i, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dx[i, j] .* sol.sh[i, j][:, :, k, 1],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dx[i, j] .* sol.sb[i, j][:, :, k, 1],
                        un,
                        ut,
                        KS.vSpace.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                    )
                    flux.fw1[i, j][:, k] .=
                        global_frame(fw1, flux.n1[i, j][1], flux.n1[i, j][2])
                end
            end
        end

        @inbounds for j = 1:KS.pSpace.ny+1
            for i = 1:KS.pSpace.nx
                vn = KS.vSpace.u .* flux.n2[i, j][1] .+ KS.vSpace.v .* flux.n2[i, j][2]
                vt = KS.vSpace.v .* flux.n2[i, j][1] .- KS.vSpace.u .* flux.n2[i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw2 = @view flux.fw2[i, j][:, k]
                    fh2 = @view flux.fh2[i, j][:, :, k]
                    fb2 = @view flux.fb2[i, j][:, :, k]

                    flux_kcu!(
                        fw2,
                        fh2,
                        fb2,
                        local_frame(
                            sol.w[i, j-1][:, k] .+
                            0.5 .* KS.pSpace.dy[i, j-1] .* sol.sw[i, j-1][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j-1][:, :, k] .+
                        0.5 .* KS.pSpace.dy[i, j-1] .* sol.sh[i, j-1][:, :, k, 2],
                        sol.b[i, j-1][:, :, k] .+
                        0.5 .* KS.pSpace.dy[i, j-1] .* sol.sb[i, j-1][:, :, k, 2],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.pSpace.dy[i, j] .* sol.sw[i, j][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dy[i, j] .* sol.sh[i, j][:, :, k, 2],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dy[i, j] .* sol.sb[i, j][:, :, k, 2],
                        vn,
                        vt,
                        KS.vSpace.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * (KS.pSpace.dx[i, j-1] + KS.pSpace.dx[i, j]),
                    )
                    flux.fw2[i, j][:, k] .=
                        global_frame(fw2, flux.n2[i, j][1], flux.n2[i, j][2])
                end
            end
        end

    elseif mode == :ugks

        @inbounds for j = 1:KS.pSpace.ny
            for i = 1:KS.pSpace.nx+1
                un = KS.vSpace.u .* flux.n1[i, j][1] .+ KS.vSpace.v .* flux.n1[i, j][2]
                ut = KS.vSpace.v .* flux.n1[i, j][1] .- KS.vSpace.u .* flux.n1[i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw1 = @view flux.fw1[i, j][:, k]
                    fh1 = @view flux.fh1[i, j][:, :, k]
                    fb1 = @view flux.fb1[i, j][:, :, k]

                    flux_ugks!(
                        fw1,
                        fh1,
                        fb1,
                        local_frame(
                            sol.w[i-1, j][:, k] .+
                            0.5 .* KS.pSpace.dx[i-1, j] .* sol.sw[i-1, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i-1, j][:, :, k] .+
                        0.5 .* KS.pSpace.dx[i-1, j] .* sol.sh[i-1, j][:, :, k, 1],
                        sol.b[i-1, j][:, :, k] .+
                        0.5 .* KS.pSpace.dx[i-1, j] .* sol.sb[i-1, j][:, :, k, 1],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.pSpace.dx[i, j] .* sol.sw[i, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dx[i, j] .* sol.sh[i, j][:, :, k, 1],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dx[i, j] .* sol.sb[i, j][:, :, k, 1],
                        un,
                        ut,
                        KS.vSpace.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * KS.pSpace.dx[i-1, j],
                        0.5 * KS.pSpace.dx[i, j],
                        0.5 * (KS.pSpace.dy[i-1, j] + KS.pSpace.dy[i, j]),
                        sol.sh[i-1, j][:, :, k, 1],
                        sol.sb[i-1, j][:, :, k, 1],
                        sol.sh[i, j][:, :, k, 1],
                        sol.sb[i, j][:, :, k, 1],
                    )
                    flux.fw1[i, j][:, k] .=
                        global_frame(fw1, flux.n1[i, j][1], flux.n1[i, j][2])
                end
            end
        end

        @inbounds for j = 1:KS.pSpace.ny+1
            for i = 1:KS.pSpace.nx
                vn = KS.vSpace.u .* flux.n2[i, j][1] .+ KS.vSpace.v .* flux.n2[i, j][2]
                vt = KS.vSpace.v .* flux.n2[i, j][1] .- KS.vSpace.u .* flux.n2[i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw2 = @view flux.fw2[i, j][:, k]
                    fh2 = @view flux.fh2[i, j][:, :, k]
                    fb2 = @view flux.fb2[i, j][:, :, k]

                    flux_ugks!(
                        fw2,
                        fh2,
                        fb2,
                        local_frame(
                            sol.w[i, j-1][:, k] .+
                            0.5 .* KS.pSpace.dy[i, j-1] .* sol.sw[i, j-1][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j-1][:, :, k] .+
                        0.5 .* KS.pSpace.dy[i, j-1] .* sol.sh[i, j-1][:, :, k, 2],
                        sol.b[i, j-1][:, :, k] .+
                        0.5 .* KS.pSpace.dy[i, j-1] .* sol.sb[i, j-1][:, :, k, 2],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.pSpace.dy[i, j] .* sol.sw[i, j][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dy[i, j] .* sol.sh[i, j][:, :, k, 2],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.pSpace.dy[i, j] .* sol.sb[i, j][:, :, k, 2],
                        vn,
                        vt,
                        KS.vSpace.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * KS.pSpace.dy[i, j-1],
                        0.5 * KS.pSpace.dy[i, j],
                        0.5 * (KS.pSpace.dx[i, j-1] + KS.pSpace.dx[i, j]),
                        sol.sh[i, j-1][:, :, k, 2],
                        sol.sb[i, j-1][:, :, k, 2],
                        sol.sh[i, j][:, :, k, 2],
                        sol.sb[i, j][:, :, k, 2],
                    )
                    flux.fw2[i, j][:, k] .=
                        global_frame(fw2, flux.n2[i, j][1], flux.n2[i, j][2])
                end
            end
        end

    else

        throw("flux mode isn't available")

    end # if

end


function evolve!(
    KS::SolverSet,
    uq::AbstractUQ,
    ctr::AbstractArray{<:AbstractControlVolume1D,1},
    face::AbstractArray{<:AbstractInterface1D,1},
    dt::AbstractFloat;
    mode = :kfvs::Symbol,
)

    if uq.method == "collocation"
        @inbounds Threads.@threads for i in eachindex(face)
            uqflux_flow!(KS, ctr[i-1], face[i], ctr[i], dt, mode=mode)
            uqflux_em!(KS, uq, ctr[i-2], ctr[i-1], face[i], ctr[i], ctr[i+1], dt)
        end
    elseif uq.method == "galerkin"
    else
        throw("UQ method isn't available")
    end

end


function uqflux_flow!(
    KS::SolverSet,
    cellL::ControlVolume1D1F,
    face::Interface1D1F,
    cellR::ControlVolume1D1F,
    dt::AbstractFloat;
    mode = :kfvs::Symbol,
)

    if mode == :kfvs

        if ndims(cellL.f) == 2

            @inbounds for j in axes(cellL.f, 2)
                fw = @view face.fw[:, j]
                ff = @view face.ff[:, j]

                flux_kfvs!(
                    fw,
                    ff,
                    cellL.f[:, j] .+ 0.5 .* cellL.dx .* cellL.sf[:, j],
                    cellR.f[:, j] .- 0.5 .* cellR.dx .* cellR.sf[:, j],
                    KS.vSpace.u,
                    KS.vSpace.weights,
                    dt,
                    cellL.sf[:, j],
                    cellR.sf[:, j],
                )
            end

        elseif ndims(cellL.f) == 3

            @inbounds for k in axes(cellL.f, 3)
                for j in axes(cellL.f, 2)
                    fw = @view face.fw[:, j, k]
                    ff = @view face.ff[:, j, k]

                    flux_kfvs!(
                        fw,
                        ff,
                        cellL.f[:, j, k] .+ 0.5 .* cellL.dx .* cellL.sf[:, j, k],
                        cellR.f[:, j, k] .- 0.5 .* cellR.dx .* cellR.sf[:, j, k],
                        KS.vSpace.u[:, k],
                        KS.vSpace.weights[:, k],
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


function uqflux_flow!(
    KS::SolverSet,
    cellL::ControlVolume1D4F,
    face::Interface1D4F,
    cellR::ControlVolume1D4F,
    dt::AbstractFloat;
    mode = :kfvs::Symbol,
)

    if mode == :kfvs

        if ndims(cellL.h0) == 2

            @inbounds for j in axes(cellL.h0, 2)
                fw = @view face.fw[:, j]
                fh0 = @view face.fh0[:, j]
                fh1 = @view face.fh1[:, j]
                fh2 = @view face.fh2[:, j]
                fh3 = @view face.fh3[:, j]

                flux_kfvs!(
                    fw,
                    fh0,
                    fh1,
                    fh2,
                    fh3,
                    cellL.h0[:, j] .+ 0.5 .* cellL.dx .* cellL.sh0[:, j],
                    cellL.h1[:, j] .+ 0.5 .* cellL.dx .* cellL.sh1[:, j],
                    cellL.h2[:, j] .+ 0.5 .* cellL.dx .* cellL.sh2[:, j],
                    cellL.h3[:, j] .+ 0.5 .* cellL.dx .* cellL.sh3[:, j],
                    cellR.h0[:, j] .- 0.5 .* cellR.dx .* cellR.sh0[:, j],
                    cellR.h1[:, j] .- 0.5 .* cellR.dx .* cellR.sh1[:, j],
                    cellR.h2[:, j] .- 0.5 .* cellR.dx .* cellR.sh2[:, j],
                    cellR.h3[:, j] .- 0.5 .* cellR.dx .* cellR.sh3[:, j],
                    KS.vSpace.u,
                    KS.vSpace.weights,
                    dt,
                    cellL.sh0[:, j],
                    cellL.sh1[:, j],
                    cellL.sh2[:, j],
                    cellL.sh3[:, j],
                    cellR.sh0[:, j],
                    cellR.sh1[:, j],
                    cellR.sh2[:, j],
                    cellR.sh3[:, j],
                )
            end

        elseif ndims(cellL.h0) == 3

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
                    cellL.h0[:, j, :] .+ 0.5 .* cellL.dx .* cellL.sh0[:, j, :],
                    cellL.h1[:, j, :] .+ 0.5 .* cellL.dx .* cellL.sh1[:, j, :],
                    cellL.h2[:, j, :] .+ 0.5 .* cellL.dx .* cellL.sh2[:, j, :],
                    cellL.h3[:, j, :] .+ 0.5 .* cellL.dx .* cellL.sh3[:, j, :],
                    cellR.h0[:, j, :] .- 0.5 .* cellR.dx .* cellR.sh0[:, j, :],
                    cellR.h1[:, j, :] .- 0.5 .* cellR.dx .* cellR.sh1[:, j, :],
                    cellR.h2[:, j, :] .- 0.5 .* cellR.dx .* cellR.sh2[:, j, :],
                    cellR.h3[:, j, :] .- 0.5 .* cellR.dx .* cellR.sh3[:, j, :],
                    KS.vSpace.u,
                    KS.vSpace.weights,
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

        else

            throw("inconsistent distribution function size")

        end

    else

        throw("flux mode not available")

    end # if

end


function uqflux_flow!(
    KS::SolverSet,
    cellL::ControlVolume1D3F,
    face::Interface1D3F,
    cellR::ControlVolume1D3F,
    dt::AbstractFloat;
    mode = :kfvs::Symbol,
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
                    cellL.h0[:, :, j, k] .+ 0.5 .* cellL.dx .* cellL.sh0[:, :, j, k],
                    cellL.h1[:, :, j, k] .+ 0.5 .* cellL.dx .* cellL.sh1[:, :, j, k],
                    cellL.h2[:, :, j, k] .+ 0.5 .* cellL.dx .* cellL.sh2[:, :, j, k],
                    cellR.h0[:, :, j, k] .- 0.5 .* cellR.dx .* cellR.sh0[:, :, j, k],
                    cellR.h1[:, :, j, k] .- 0.5 .* cellR.dx .* cellR.sh1[:, :, j, k],
                    cellR.h2[:, :, j, k] .- 0.5 .* cellR.dx .* cellR.sh2[:, :, j, k],
                    KS.vSpace.u[:, :, k],
                    KS.vSpace.v[:, :, k],
                    KS.vSpace.weights[:, :, k],
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
                cellL.w[:, j, :] .+ 0.5 .* cellL.dx .* cellL.sw[:, j, :],
                cellL.h0[:, :, j, :] .+ 0.5 .* cellL.dx .* cellL.sh0[:, :, j, :],
                cellL.h1[:, :, j, :] .+ 0.5 .* cellL.dx .* cellL.sh1[:, :, j, :],
                cellL.h2[:, :, j, :] .+ 0.5 .* cellL.dx .* cellL.sh2[:, :, j, :],
                cellR.w[:, j, :] .- 0.5 .* cellR.dx .* cellR.sw[:, j, :],
                cellR.h0[:, :, j, :] .- 0.5 .* cellR.dx .* cellR.sh0[:, :, j, :],
                cellR.h1[:, :, j, :] .- 0.5 .* cellR.dx .* cellR.sh1[:, :, j, :],
                cellR.h2[:, :, j, :] .- 0.5 .* cellR.dx .* cellR.sh2[:, :, j, :],
                KS.vSpace.u[:, :, :],
                KS.vSpace.v[:, :, :],
                KS.vSpace.weights[:, :, :],
                KS.gas.K,
                KS.gas.γ,
                KS.gas.mi,
                KS.gas.ni,
                KS.gas.me,
                KS.gas.ne,
                KS.gas.Kn[1],
                dt,
                1.0, 
            )
        end

    else

        throw("flux mode not available")

    end # if

end


"""
Maxwell's diffusive boundary flux

"""
function evolve_boundary!(
    bc::Array,
    KS::SolverSet,
    sol::Solution2D2F,
    flux::Flux2D2F,
    dt::AbstractFloat;
    mode = :maxwell::Symbol,
)

    if mode == :maxwell

        @inbounds for j = 1:KS.pSpace.ny
            un = KS.vSpace.u .* flux.n1[1, j][1] .+ KS.vSpace.v .* flux.n1[1, j][2]
            ut = KS.vSpace.v .* flux.n1[1, j][1] .- KS.vSpace.u .* flux.n1[1, j][2]

            for k in axes(sol.w[1, 1], 2)
                bcL = local_frame(bc[1][:, k], flux.n1[1, j][1], flux.n1[1, j][2])
                bcR = local_frame(bc[2][:, k], flux.n1[1, j][1], flux.n1[1, j][2])

                fw1 = @view flux.fw1[1, j][:, k]
                fh1 = @view flux.fh1[1, j][:, :, k]
                fb1 = @view flux.fb1[1, j][:, :, k]
                flux_boundary_maxwell!(
                    fw1,
                    fh1,
                    fb1,
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
                    global_frame(flux.fw1[1, j][:, k], flux.n1[1, j][1], flux.n1[1, j][2])

                fw2 = @view flux.fw1[end, j][:, k]
                fh2 = @view flux.fh1[end, j][:, :, k]
                fb2 = @view flux.fb1[end, j][:, :, k]
                flux_boundary_maxwell!(
                    fw2,
                    fh2,
                    fb2,
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
                flux.fw1[end, j][:, k] .= global_frame(
                    flux.fw1[end, j][:, k],
                    flux.n1[end, j][1],
                    flux.n1[end, j][2],
                )

            end
        end

        @inbounds for i = 1:KS.pSpace.nx
            vn = KS.vSpace.u .* flux.n2[i, 1][1] .+ KS.vSpace.v .* flux.n2[i, 1][2]
            vt = KS.vSpace.v .* flux.n2[i, 1][1] .- KS.vSpace.u .* flux.n2[i, 1][2]

            for k in axes(sol.w[1, 1], 2)
                bcU = local_frame(bc[3][:, k], flux.n2[i, 1][1], flux.n2[i, 1][2])
                bcD = local_frame(bc[4][:, k], flux.n2[i, 1][1], flux.n2[i, 1][2])

                fw3 = @view flux.fw2[i, 1][:, k]
                fh3 = @view flux.fh2[i, 1][:, :, k]
                fb3 = @view flux.fb2[i, 1][:, :, k]
                flux_boundary_maxwell!(
                    fw3,
                    fh3,
                    fb3,
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
                    global_frame(flux.fw2[i, 1][:, k], flux.n2[i, 1][1], flux.n2[i, 1][2])

                fw4 = @view flux.fw2[i, end][:, k]
                fh4 = @view flux.fh2[i, end][:, :, k]
                fb4 = @view flux.fb2[i, end][:, :, k]
                flux_boundary_maxwell!(
                    fw4,
                    fh4,
                    fb4,
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
                flux.fw2[i, end][:, k] .= global_frame(
                    flux.fw2[i, end][:, k],
                    flux.n2[i, end][1],
                    flux.n2[i, end][2],
                )
            end
        end

    end # if

end


function uqflux_em!(
    KS::SolverSet,
    uq::AbstractUQ,
    cellLL::ControlVolume1D4F,
    cellL::ControlVolume1D4F,
    face::Interface1D4F,
    cellR::ControlVolume1D4F,
    cellRR::ControlVolume1D4F,
    dt::Real,
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
                cellL.dx,
                cellR.dx,
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
                cellL.dx,
                cellR.dx,
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


function uqflux_em!(
    KS::SolverSet,
    uq::AbstractUQ,
    cellLL::ControlVolume1D3F,
    cellL::ControlVolume1D3F,
    face::Interface1D3F,
    cellR::ControlVolume1D3F,
    cellRR::ControlVolume1D3F,
    dt::Real,
)

    if uq.method == "collocation"

        for j = 1:uq.op.quad.Nquad
            femL = @view face.femL[:, j]
            femR = @view face.femR[:, j]
            femRU = @view face.femRU[:, j]
            femRD = @view face.femRD[:, j]
            femLU = @view face.femLU[:, j]
            femLD = @view face.femLD[:, j]

            flux_emx!(
                femL,
                femR,
                femRU,
                femRD,
                femLU,
                femLD,
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
                cellL.dx,
                cellR.dx,
                KS.gas.A1p,
                KS.gas.A1n,
                KS.gas.A2p,
                KS.gas.A2n,
                KS.gas.D1,
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
                cellL.dx,
                cellR.dx,
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
