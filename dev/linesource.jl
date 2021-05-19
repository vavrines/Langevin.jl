using LinearAlgebra, ProgressMeter, Langevin

cd(@__DIR__)
D = read_dict("linesource.txt")
set = set_setup(D)
ps = set_geometry(D)
# quadrature
quadratureorder = 8
points, weights = legendre_quadrature(quadratureorder)
vs = UnstructVSpace(-1.0, 1.0, length(weights), points, weights)
# material
radiation = Radiation(D[:knudsen], D[:sigmaS], D[:sigmaA])
# initial/boundary conditions
nq = size(points, 1)
f0 = ones(nq) .* 1e-4
w0 = discrete_moments(f0, weights)
ib = IB1F([w0], [w0], f0, [w0], [w0], [w0], f0, [w0])

ks = SolverSet(set, ps, vs, radiation, ib, pwd())

uq = UQ1D(3, 6, -0.1, 0.1)

function init_field(x, y)
    s2 = 0.03^2
    flr = 1e-4
    return max(flr, 1.0 / (4.0 * pi * s2) * exp(-(x^2 + y^2) / 4.0 / s2))
end

ctr = Array{ControlVolumeUS1F}(undef, size(ks.ps.cellid, 1))
for i in eachindex(ctr)
    n = Vector{Float64}[]
    for j = 1:3
        push!(
            n,
            KitBase.unit_normal(
                ps.points[ps.facePoints[ps.cellFaces[i, j], 1], :],
                ps.points[ps.facePoints[ps.cellFaces[i, j], 2], :],
            ),
        )

        if dot(
            ps.faceCenter[ps.cellFaces[i, j], 1:2] .- ps.cellCenter[i, 1:2],
            n[j],
        ) < 0
            n[j] .= -n[j]
        end
    end

    dx = [
        KitBase.point_distance(
            ps.cellCenter[i, :],
            ps.points[ps.cellid[i, 1], :],
            ps.points[ps.cellid[i, 2], :],
        ),
        KitBase.point_distance(
            ps.cellCenter[i, :],
            ps.points[ps.cellid[i, 2], :],
            ps.points[ps.cellid[i, 3], :],
        ),
        KitBase.point_distance(
            ps.cellCenter[i, :],
            ps.points[ps.cellid[i, 3], :],
            ps.points[ps.cellid[i, 1], :],
        ),
    ]

    phi = zeros(nq, uq.nq)
    for q in axes(phi, 2)
        phi[:, q] .= init_field(ks.ps.cellCenter[i, 1], ks.ps.cellCenter[i, 2])# * (1.0 + uq.pceSample[q])
    end
    prim = zeros(uq.nq)
    for p in axes(prim, 1)
        prim[p] = discrete_moments(phi[:, p], ks.vs.weights)
    end

    ctr[i] = KitBase.ControlVolumeUS1F(
        n,
        ps.cellCenter[i, :],
        dx,
        prim,
        prim,
        phi,
    )
end

face = Array{Interface2D1F}(undef, size(ks.ps.faceType))
for i in eachindex(face)
    len =
        norm(ps.points[ps.facePoints[i, 1], :] .- ps.points[ps.facePoints[i, 2], :])
    n = KitBase.unit_normal(
        ps.points[ps.facePoints[i, 1], :],
        ps.points[ps.facePoints[i, 2], :],
    )

    if !(-1 in ps.faceCells[i, :])
        n0 =
            ps.cellCenter[ps.faceCells[i, 2], :] .-
            ps.cellCenter[ps.faceCells[i, 1], :]
    else
        idx =
            ifelse(ps.faceCells[i, 1] != -1, ps.faceCells[i, 1], ps.faceCells[i, 2])
        n0 = ps.cellCenter[idx, :] .- ps.faceCenter[i, :]
    end
    if dot(n, n0[1:2]) < 0
        n .= -n
    end

    fw = zeros(uq.nq)
    ff = zeros(nq, uq.nq)

    face[i] = KitBase.Interface2D1F(len, n[1], n[2], fw, ff)
end

dt = 1.2 / 150 * ks.set.cfl
nt = ks.set.maxTime ÷ dt |> Int

@showprogress for iter = 1:nt
    @inbounds Threads.@threads for i in eachindex(face)
        velo = ks.vs.u[:, 1] .* face[i].n[1] + ks.vs.u[:, 2] .* face[i].n[2]
        if !(-1 in ps.faceCells[i, :])
            for j = 1:uq.nq
                ff = @view face[i].ff[:, j]

                KitBase.flux_kfvs!(
                    ff,
                    ctr[ps.faceCells[i, 1]].f[:, j],
                    ctr[ps.faceCells[i, 2]].f[:, j],
                    velo,
                    dt,
                )
            end
        end
    end

    @inbounds Threads.@threads for i in eachindex(ctr)
        if ps.cellType[i] == 0
            for j = 1:3
                dirc = sign(dot(ctr[i].n[j], face[ps.cellFaces[i, j]].n))
                @. ctr[i].f -=
                    dirc * face[ps.cellFaces[i, j]].ff * face[ps.cellFaces[i, j]].len /
                    ks.ps.cellArea[i]
            end
#=
            integral = zeros(uq.nq)
            for j in eachindex(integral)
                integral[j] = KitBase.discrete_moments(ctr[i].f[:, j], ks.vs.weights)
            end
            integral ./= 4.0 * π=#

            for j = 1:uq.nq
                #@. ctr[i].f[:, j] += (integral[j] - ctr[i].f[:, j]) * dt
                ctr[i].w[j] = sum(ctr[i].f[j] .* ks.vs.weights)
            end
        end
    end
end

sol = zeros(length(ctr), uq.nq)
for i in eachindex(ctr)
    sol[i, :] = ctr[i].w
end

write_vtk(ks.ps.points, ks.ps.cellid, sol)
