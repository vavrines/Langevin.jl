"""
$(SIGNATURES)

Reconstruct solution

Note that cell-centered fields can directly call KitBase.reconstruct! for reconstruction
"""
function KitBase.reconstruct!(
    KS::SolverSet,
    sol::Solution1F{T1,T2,T3,T4,1},
) where {T1,T2,T3,T4}
    @inbounds @threads for i = 1:KS.ps.nx
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

"""
$(SIGNATURES)
"""
function KitBase.reconstruct!(
    KS::SolverSet,
    sol::Solution2F{T1,T2,T3,T4,1},
) where {T1,T2,T3,T4}
    @inbounds @threads for i = 1:KS.ps.nx
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

"""
$(SIGNATURES)
"""
function KitBase.reconstruct!(
    KS::SolverSet,
    sol::Solution2F{T1,T2,T3,T4,2},
) where {T1,T2,T3,T4}
    # x direction
    @inbounds @threads for j = 1:KS.ps.ny
        sw1 = @view sol.sw[1, j][:, :, 1]
        KitBase.reconstruct2!(
            sw1,
            sol.w[1, j],
            sol.w[2, j],
            0.5 * (KS.ps.dx[1, j] + KS.ps.dx[2, j]),
        )

        sh1 = @view sol.sh[1, j][:, :, :, 1]
        KitBase.reconstruct2!(
            sh1,
            sol.h[1, j],
            sol.h[2, j],
            0.5 * (KS.ps.dx[1, j] + KS.ps.dx[2, j]),
        )

        sb1 = @view sol.sb[1, j][:, :, :, 1]
        KitBase.reconstruct2!(
            sb1,
            sol.b[1, j],
            sol.b[2, j],
            0.5 * (KS.ps.dx[1, j] + KS.ps.dx[2, j]),
        )

        sw2 = @view sol.sw[KS.ps.nx, j][:, :, 1]
        KitBase.reconstruct2!(
            sw2,
            sol.w[KS.ps.nx-1, j],
            sol.w[KS.ps.nx, j],
            0.5 * (KS.ps.dx[KS.ps.nx-1, j] + KS.ps.dx[KS.ps.nx, j]),
        )

        sh2 = @view sol.sh[KS.ps.nx, j][:, :, :, 1]
        KitBase.reconstruct2!(
            sh2,
            sol.h[KS.ps.nx-1, j],
            sol.h[KS.ps.nx, j],
            0.5 * (KS.ps.dx[KS.ps.nx-1, j] + KS.ps.dx[KS.ps.nx, j]),
        )

        sb2 = @view sol.sb[KS.ps.nx, j][:, :, :, 1]
        KitBase.reconstruct2!(
            sb2,
            sol.b[KS.ps.nx-1, j],
            sol.b[KS.ps.nx, j],
            0.5 * (KS.ps.dx[KS.ps.nx-1, j] + KS.ps.dx[KS.ps.nx, j]),
        )
    end

    @inbounds @threads for j = 1:KS.ps.ny
        for i = 2:KS.ps.nx-1
            sw = @view sol.sw[i, j][:, :, 1]
            KitBase.reconstruct3!(
                sw,
                sol.w[i-1, j],
                sol.w[i, j],
                sol.w[i+1, j],
                0.5 * (KS.ps.dx[i-1, j] + KS.ps.dx[i, j]),
                0.5 * (KS.ps.dx[i, j] + KS.ps.dx[i+1, j]),
            )

            sh = @view sol.sh[i, j][:, :, :, 1]
            KitBase.reconstruct3!(
                sh,
                sol.h[i-1, j],
                sol.h[i, j],
                sol.h[i+1, j],
                0.5 * (KS.ps.dx[i-1, j] + KS.ps.dx[i, j]),
                0.5 * (KS.ps.dx[i, j] + KS.ps.dx[i+1, j]),
            )

            sb = @view sol.sb[i, j][:, :, :, 1]
            KitBase.reconstruct3!(
                sb,
                sol.b[i-1, j],
                sol.b[i, j],
                sol.b[i+1, j],
                0.5 * (KS.ps.dx[i-1, j] + KS.ps.dx[i, j]),
                0.5 * (KS.ps.dx[i, j] + KS.ps.dx[i+1, j]),
            )
        end
    end

    # y direction
    @inbounds @threads for i = 1:KS.ps.nx
        sw1 = @view sol.sw[i, 1][:, :, 2]
        KitBase.reconstruct2!(
            sol.sw[i, 1][:, :, 2],
            sol.w[i, 1],
            sol.w[i, 2],
            0.5 * (KS.ps.dy[i, 1] + KS.ps.dy[i, 2]),
        )

        sh1 = @view sol.sh[i, 1][:, :, :, 2]
        KitBase.reconstruct2!(
            sh1,
            sol.h[i, 1],
            sol.h[i, 2],
            0.5 * (KS.ps.dy[i, 1] + KS.ps.dy[i, 2]),
        )

        sb1 = @view sol.sb[i, 1][:, :, :, 2]
        KitBase.reconstruct2!(
            sb1,
            sol.b[i, 1],
            sol.b[i, 2],
            0.5 * (KS.ps.dy[i, 1] + KS.ps.dy[i, 2]),
        )

        sw2 = @view sol.sw[i, KS.ps.ny][:, :, 2]
        KitBase.reconstruct2!(
            sw2,
            sol.w[i, KS.ps.ny-1],
            sol.w[i, KS.ps.ny],
            0.5 * (KS.ps.dy[i, KS.ps.ny-1] + KS.ps.dy[i, KS.ps.ny]),
        )

        sh2 = @view sol.sh[i, KS.ps.ny][:, :, :, 2]
        KitBase.reconstruct2!(
            sh2,
            sol.h[i, KS.ps.ny-1],
            sol.h[i, KS.ps.ny],
            0.5 * (KS.ps.dy[i, KS.ps.ny-1] + KS.ps.dy[i, KS.ps.ny]),
        )

        sb2 = @view sol.sb[i, KS.ps.ny][:, :, :, 2]
        KitBase.reconstruct2!(
            sb2,
            sol.b[i, KS.ps.ny-1],
            sol.b[i, KS.ps.ny],
            0.5 * (KS.ps.dy[i, KS.ps.ny-1] + KS.ps.dy[i, KS.ps.ny]),
        )
    end

    @inbounds @threads for j = 2:KS.ps.ny-1
        for i = 1:KS.ps.nx
            sw = @view sol.sw[i, j][:, :, 2]
            KitBase.reconstruct3!(
                sw,
                sol.w[i, j-1],
                sol.w[i, j],
                sol.w[i, j+1],
                0.5 * (KS.ps.dy[i, j-1] + KS.ps.dy[i, j]),
                0.5 * (KS.ps.dy[i, j] + KS.ps.dy[i, j+1]),
            )

            sh = @view sol.sh[i, j][:, :, :, 2]
            KitBase.reconstruct3!(
                sh,
                sol.h[i, j-1],
                sol.h[i, j],
                sol.h[i, j+1],
                0.5 * (KS.ps.dy[i, j-1] + KS.ps.dy[i, j]),
                0.5 * (KS.ps.dy[i, j] + KS.ps.dy[i, j+1]),
            )

            sb = @view sol.sb[i, j][:, :, :, 2]
            KitBase.reconstruct3!(
                sb,
                sol.b[i, j-1],
                sol.b[i, j],
                sol.b[i, j+1],
                0.5 * (KS.ps.dy[i, j-1] + KS.ps.dy[i, j]),
                0.5 * (KS.ps.dy[i, j] + KS.ps.dy[i, j+1]),
            )
        end
    end
end
