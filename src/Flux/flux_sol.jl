# ============================================================
# Flux for Solution Structures
# ============================================================

"""
Evolution of particle transport

  - kinetic flux vector splitting (KFVS)
  - kinetic central-upwind (KCU)
  - unified gas-kinetic scheme (UGKS)
"""
function KitBase.evolve!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution1F{T1,T2,T3,T4,1},
    flux::Flux1F,
    dt::AbstractFloat;
    mode=:kfvs::Symbol,
) where {T1,T2,T3,T4}
    if mode == :kfvs
        @inbounds @threads for i in eachindex(flux.fw)
            for j in axes(sol.w[1], 2) # over gPC coefficients or quadrature points
                fw = @view flux.fw[i][:, j]
                ff = @view flux.ff[i][:, j]

                KitBase.flux_kfvs!(
                    fw,
                    ff,
                    sol.f[i-1][:, j] .+ 0.5 .* KS.ps.dx[i-1] .* sol.∇f[i-1][:, j],
                    sol.f[i][:, j] .- 0.5 .* KS.ps.dx[i] .* sol.∇f[i][:, j],
                    KS.vs.u,
                    KS.vs.weights,
                    dt,
                    sol.∇f[i-1][:, j],
                    sol.∇f[i][:, j],
                )
            end
        end
    else
        throw("flux mode isn't available")
    end
end

function KitBase.evolve!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution2F{T1,T2,T3,T4,1},
    flux::Flux2F,
    dt::AbstractFloat;
    mode=:kfvs::Symbol,
) where {T1,T2,T3,T4}
    if mode == :kfvs
        @inbounds @threads for i in eachindex(flux.fw)
            for j in axes(sol.w[1], 2) # over gPC coefficients or quadrature points
                fw = @view flux.fw[i][:, j]
                fh = @view flux.fh[i][:, j]
                fb = @view flux.fb[i][:, j]

                KitBase.flux_kfvs!(
                    fw,
                    fh,
                    fb,
                    sol.h[i-1][:, j] .+ 0.5 .* KS.ps.dx[i-1] .* sol.∇h[i-1][:, j],
                    sol.b[i-1][:, j] .+ 0.5 .* KS.ps.dx[i-1] .* sol.∇b[i-1][:, j],
                    sol.h[i][:, j] .- 0.5 .* KS.ps.dx[i] .* sol.∇h[i][:, j],
                    sol.b[i][:, j] .- 0.5 .* KS.ps.dx[i] .* sol.∇b[i][:, j],
                    KS.vs.u,
                    KS.vs.weights,
                    dt,
                    sol.∇h[i-1][:, j],
                    sol.∇b[i-1][:, j],
                    sol.∇h[i][:, j],
                    sol.∇b[i][:, j],
                )
            end
        end
    else
        throw("flux mode isn't available")
    end
end

#--- 2D case ---#
function KitBase.evolve!(
    KS::SolverSet,
    uq::AbstractUQ,
    sol::Solution2F{T1,T2,T3,T4,2},
    flux::Flux2F,
    dt::AbstractFloat;
    mode=:kfvs::Symbol,
) where {T1,T2,T3,T4}
    if mode == :kfvs
        @inbounds for j in 1:KS.ps.ny
            for i in 1:(KS.ps.nx+1)
                un = KS.vs.u .* flux.n[1][i, j][1] .+ KS.vs.v .* flux.n[1][i, j][2]
                ut = KS.vs.v .* flux.n[1][i, j][1] .- KS.vs.u .* flux.n[1][i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw1 = @view flux.fw[1][i, j][:, k]
                    fh1 = @view flux.fh[1][i, j][:, :, k]
                    fb1 = @view flux.fb[1][i, j][:, :, k]

                    flux_kfvs!(
                        fw1,
                        fh1,
                        fb1,
                        sol.h[i-1, j][:, :, k] .+
                        0.5 .* KS.ps.dx[i-1, j] .* sol.∇h[i-1, j][:, :, k, 1],
                        sol.b[i-1, j][:, :, k] .+
                        0.5 .* KS.ps.dx[i-1, j] .* sol.∇b[i-1, j][:, :, k, 1],
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.ps.dx[i, j] .* sol.∇h[i, j][:, :, k, 1],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.ps.dx[i, j] .* sol.∇b[i, j][:, :, k, 1],
                        un,
                        ut,
                        KS.vs.weights,
                        dt,
                        0.5 * (KS.ps.dy[i-1, j] + KS.ps.dy[i, j]),
                        sol.∇h[i-1, j][:, :, k, 1],
                        sol.∇b[i-1, j][:, :, k, 1],
                        sol.∇h[i, j][:, :, k, 1],
                        sol.∇b[i, j][:, :, k, 1],
                    )
                    flux.fw[1][i, j][:, k] .=
                        global_frame(fw1, flux.n[1][i, j][1], flux.n[1][i, j][2])
                end
            end
        end

        @inbounds for j in 1:(KS.ps.ny+1)
            for i in 1:KS.ps.nx
                vn = KS.vs.u .* flux.n[2][i, j][1] .+ KS.vs.v .* flux.n[2][i, j][2]
                vt = KS.vs.v .* flux.n[2][i, j][1] .- KS.vs.u .* flux.n[2][i, j][2]

                for k in axes(sol.w[1, 1], 2)
                    fw2 = @view flux.fw[2][i, j][:, k]
                    fh2 = @view flux.fh[2][i, j][:, :, k]
                    fb2 = @view flux.fb[2][i, j][:, :, k]

                    flux_kfvs!(
                        fw2,
                        fh2,
                        fb2,
                        sol.h[i, j-1][:, :, k] .+
                        0.5 .* KS.ps.dy[i, j-1] .* sol.∇h[i, j-1][:, :, k, 2],
                        sol.b[i, j-1][:, :, k] .+
                        0.5 .* KS.ps.dy[i, j-1] .* sol.∇b[i, j-1][:, :, k, 2],
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.ps.dy[i, j] .* sol.∇h[i, j][:, :, k, 2],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.ps.dy[i, j] .* sol.∇b[i, j][:, :, k, 2],
                        vn,
                        vt,
                        KS.vs.weights,
                        dt,
                        0.5 * (KS.ps.dx[i, j-1] + KS.ps.dx[i, j]),
                        sol.∇h[i, j-1][:, :, k, 2],
                        sol.∇b[i, j-1][:, :, k, 2],
                        sol.∇h[i, j][:, :, k, 2],
                        sol.∇b[i, j][:, :, k, 2],
                    )
                    flux.fw[2][i, j][:, k] .=
                        global_frame(fw2, flux.n[2][i, j][1], flux.n[2][i, j][2])
                end
            end
        end

    elseif mode == :kcu
        @inbounds for j in 1:KS.ps.ny
            for i in 1:(KS.ps.nx+1)
                un = KS.vs.u .* flux.n1[i, j][1] .+ KS.vs.v .* flux.n1[i, j][2]
                ut = KS.vs.v .* flux.n1[i, j][1] .- KS.vs.u .* flux.n1[i, j][2]

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
                            0.5 .* KS.ps.dx[i-1, j] .* sol.∇w[i-1, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i-1, j][:, :, k] .+
                        0.5 .* KS.ps.dx[i-1, j] .* sol.∇h[i-1, j][:, :, k, 1],
                        sol.b[i-1, j][:, :, k] .+
                        0.5 .* KS.ps.dx[i-1, j] .* sol.∇b[i-1, j][:, :, k, 1],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.ps.dx[i, j] .* sol.∇w[i, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.ps.dx[i, j] .* sol.∇h[i, j][:, :, k, 1],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.ps.dx[i, j] .* sol.∇b[i, j][:, :, k, 1],
                        un,
                        ut,
                        KS.vs.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * (KS.ps.dy[i-1, j] + KS.ps.dy[i, j]),
                    )
                    flux.fw1[i, j][:, k] .=
                        global_frame(fw1, flux.n1[i, j][1], flux.n1[i, j][2])
                end
            end
        end

        @inbounds for j in 1:(KS.ps.ny+1)
            for i in 1:KS.ps.nx
                vn = KS.vs.u .* flux.n2[i, j][1] .+ KS.vs.v .* flux.n2[i, j][2]
                vt = KS.vs.v .* flux.n2[i, j][1] .- KS.vs.u .* flux.n2[i, j][2]

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
                            0.5 .* KS.ps.dy[i, j-1] .* sol.∇w[i, j-1][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j-1][:, :, k] .+
                        0.5 .* KS.ps.dy[i, j-1] .* sol.∇h[i, j-1][:, :, k, 2],
                        sol.b[i, j-1][:, :, k] .+
                        0.5 .* KS.ps.dy[i, j-1] .* sol.∇b[i, j-1][:, :, k, 2],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.ps.dy[i, j] .* sol.∇w[i, j][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.ps.dy[i, j] .* sol.∇h[i, j][:, :, k, 2],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.ps.dy[i, j] .* sol.∇b[i, j][:, :, k, 2],
                        vn,
                        vt,
                        KS.vs.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * (KS.ps.dx[i, j-1] + KS.ps.dx[i, j]),
                    )
                    flux.fw2[i, j][:, k] .=
                        global_frame(fw2, flux.n2[i, j][1], flux.n2[i, j][2])
                end
            end
        end

    elseif mode == :ugks
        @inbounds for j in 1:KS.ps.ny
            for i in 1:(KS.ps.nx+1)
                un = KS.vs.u .* flux.n1[i, j][1] .+ KS.vs.v .* flux.n1[i, j][2]
                ut = KS.vs.v .* flux.n1[i, j][1] .- KS.vs.u .* flux.n1[i, j][2]

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
                            0.5 .* KS.ps.dx[i-1, j] .* sol.∇w[i-1, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i-1, j][:, :, k] .+
                        0.5 .* KS.ps.dx[i-1, j] .* sol.∇h[i-1, j][:, :, k, 1],
                        sol.b[i-1, j][:, :, k] .+
                        0.5 .* KS.ps.dx[i-1, j] .* sol.∇b[i-1, j][:, :, k, 1],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.ps.dx[i, j] .* sol.∇w[i, j][:, k, 1],
                            flux.n1[i, j][1],
                            flux.n1[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.ps.dx[i, j] .* sol.∇h[i, j][:, :, k, 1],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.ps.dx[i, j] .* sol.∇b[i, j][:, :, k, 1],
                        un,
                        ut,
                        KS.vs.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * KS.ps.dx[i-1, j],
                        0.5 * KS.ps.dx[i, j],
                        0.5 * (KS.ps.dy[i-1, j] + KS.ps.dy[i, j]),
                        sol.∇h[i-1, j][:, :, k, 1],
                        sol.∇b[i-1, j][:, :, k, 1],
                        sol.∇h[i, j][:, :, k, 1],
                        sol.∇b[i, j][:, :, k, 1],
                    )
                    flux.fw1[i, j][:, k] .=
                        global_frame(fw1, flux.n1[i, j][1], flux.n1[i, j][2])
                end
            end
        end

        @inbounds for j in 1:(KS.ps.ny+1)
            for i in 1:KS.ps.nx
                vn = KS.vs.u .* flux.n2[i, j][1] .+ KS.vs.v .* flux.n2[i, j][2]
                vt = KS.vs.v .* flux.n2[i, j][1] .- KS.vs.u .* flux.n2[i, j][2]

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
                            0.5 .* KS.ps.dy[i, j-1] .* sol.∇w[i, j-1][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j-1][:, :, k] .+
                        0.5 .* KS.ps.dy[i, j-1] .* sol.∇h[i, j-1][:, :, k, 2],
                        sol.b[i, j-1][:, :, k] .+
                        0.5 .* KS.ps.dy[i, j-1] .* sol.∇b[i, j-1][:, :, k, 2],
                        local_frame(
                            sol.w[i, j][:, k] .-
                            0.5 .* KS.ps.dy[i, j] .* sol.∇w[i, j][:, k, 2],
                            flux.n2[i, j][1],
                            flux.n2[i, j][2],
                        ),
                        sol.h[i, j][:, :, k] .-
                        0.5 .* KS.ps.dy[i, j] .* sol.∇h[i, j][:, :, k, 2],
                        sol.b[i, j][:, :, k] .-
                        0.5 .* KS.ps.dy[i, j] .* sol.∇b[i, j][:, :, k, 2],
                        vn,
                        vt,
                        KS.vs.weights,
                        KS.gas.K,
                        KS.gas.γ,
                        KS.gas.μᵣ,
                        KS.gas.ω,
                        KS.gas.Pr,
                        dt,
                        0.5 * KS.ps.dy[i, j-1],
                        0.5 * KS.ps.dy[i, j],
                        0.5 * (KS.ps.dx[i, j-1] + KS.ps.dx[i, j]),
                        sol.∇h[i, j-1][:, :, k, 2],
                        sol.∇b[i, j-1][:, :, k, 2],
                        sol.∇h[i, j][:, :, k, 2],
                        sol.∇b[i, j][:, :, k, 2],
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

"""
Maxwell's diffusive boundary flux
"""
function KitBase.evolve_boundary!(
    bc::Array,
    KS::SolverSet,
    sol::Solution2F{T1,T2,T3,T4,2},
    flux::Flux2F,
    dt::AbstractFloat;
    mode=:maxwell::Symbol,
) where {T1,T2,T3,T4}
    if mode == :maxwell
        @inbounds for j in 1:KS.ps.ny
            un = KS.vs.u .* flux.n1[1, j][1] .+ KS.vs.v .* flux.n1[1, j][2]
            ut = KS.vs.v .* flux.n1[1, j][1] .- KS.vs.u .* flux.n1[1, j][2]

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
                    KS.vs.weights,
                    KS.gas.K,
                    dt,
                    KS.ps.dy[1, j],
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
                    sol.h[KS.ps.nx, j][:, :, k],
                    sol.b[KS.ps.nx, j][:, :, k],
                    un,
                    ut,
                    KS.vs.weights,
                    KS.gas.K,
                    dt,
                    KS.ps.dy[KS.ps.nx, j],
                    -1,
                )
                flux.fw1[end, j][:, k] .= global_frame(
                    flux.fw1[end, j][:, k],
                    flux.n1[end, j][1],
                    flux.n1[end, j][2],
                )
            end
        end

        @inbounds for i in 1:KS.ps.nx
            vn = KS.vs.u .* flux.n2[i, 1][1] .+ KS.vs.v .* flux.n2[i, 1][2]
            vt = KS.vs.v .* flux.n2[i, 1][1] .- KS.vs.u .* flux.n2[i, 1][2]

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
                    KS.vs.weights,
                    KS.gas.K,
                    dt,
                    KS.ps.dx[i, 1],
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
                    sol.h[i, KS.ps.ny][:, :, k],
                    sol.b[i, KS.ps.ny][:, :, k],
                    vn,
                    vt,
                    KS.vs.weights,
                    KS.gas.K,
                    dt,
                    KS.ps.dx[i, KS.ps.ny],
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
