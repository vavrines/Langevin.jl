using LinearAlgebra, ProgressMeter, Langevin

cd(@__DIR__)
D = read_dict("linesource.txt")
set = set_setup(D)

ps = set_geometry(D)
#ps = UnstructPSpace("linesource.su2")

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

ctr = Array{KitBase.ControlVolumeUS1F}(undef, size(ps.cellid, 1))
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

        if dot(ps.faceCenter[ps.cellFaces[i, j], :] .- ps.cellCenter[i, :], n[j]) < 0
            n[j] .= -n[j]
        end
    end

    phi = zeros(nq, uq.nq)
    for j = 1:uq.nq
        phi[:, j] .=
            init_field(ps.cellCenter[i, 1], ps.cellCenter[i, 2]) * (1.0 + uq.pceSample[j])
    end

    w = zeros(uq.nq)
    for i = 1:uq.nq
        w[i] = sum(weights .* phi[:, i])
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

    ctr[i] = KitBase.ControlVolumeUS1F(n, ps.cellCenter[i, :], dx, w, w, phi)
end

face = Array{KitBase.Interface2D1F}(undef, size(ps.facePoints, 1))
for i in eachindex(face)
    len = norm(ps.points[ps.facePoints[i, 1], :] .- ps.points[ps.facePoints[i, 2], :])
    n = KitBase.unit_normal(
        ps.points[ps.facePoints[i, 1], :],
        ps.points[ps.facePoints[i, 2], :],
    )

    if !(-1 in ps.faceCells[i, :])
        n0 = ps.cellCenter[ps.faceCells[i, 2], :] .- ps.cellCenter[ps.faceCells[i, 1], :]
    else
        n0 = zero(n)
    end
    if dot(n, n0) < 0
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
        velo = vs.u[:, 1] .* face[i].n[1] + vs.u[:, 2] .* face[i].n[2]
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
                    ps.cellArea[i]
            end

            integral = zeros(uq.nq)
            for k = 1:uq.nq
                integral[k] = discrete_moments(ctr[i].f[:, k], vs.weights)
            end
            #integral = KitBase.discrete_moments(ctr[i].f, vs.weights)
            integral ./= 4.0 * π
            for k = 1:uq.nq
                @. ctr[i].f[:, k] += (integral[k] - ctr[i].f[:, k]) * dt
                ctr[i].w[k] = sum(ctr[i].f[:, k] .* vs.weights)
            end
        end
    end
end

sol = zeros(length(ctr), uq.nq)
for i in axes(sol, 1)
    sol[i, :] = ctr[i].w
end

pce = zeros(length(ctr), uq.nr + 1)
for i in axes(sol, 1)
    pce[i, :] = ran_chaos(sol[i, :], uq)
end

meanstd = zeros(length(ctr), 2)
for i in axes(sol, 1)
    meanstd[i, 1] = mean(pce[i, :], uq.op)
    meanstd[i, 2] = std(pce[i, :], uq.op)
end

write_vtk(ks.ps.points, ks.ps.cellid, meanstd)
