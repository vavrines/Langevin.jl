using Langevin
using ProgressMeter
using Plots

D = Dict{Symbol,Any}()
begin
    D[:matter] = "radiation"
    D[:case] = "lattice"
    D[:space] = "2d1f1v"
    D[:flux] = "kfvs"
    D[:collision] = "bgk"
    D[:nSpecies] = 1
    D[:interpOrder] = 1
    D[:limiter] = "vanleer"
    D[:boundary] = "fix"
    D[:cfl] = 0.5
    D[:maxTime] = 0.3

    D[:x0] = -3.5
    D[:x1] = 3.5
    D[:nx] = 50
    D[:y0] = -3.5
    D[:y1] = 3.5
    D[:ny] = 50
    D[:pMeshType] = "uniform"
    D[:nxg] = 0
    D[:nyg] = 0

    D[:quadrature] = "legendre"
    D[:nq] = 8

    D[:knudsen] = 1.0
    D[:sigmaS] = 1.0
    D[:sigmaA] = 0.0
end

set = set_setup(D)
ps = set_geometry(D)
points, weights = legendre_quadrature(D[:nq])
vs = UnstructVSpace(-1.0, 1.0, length(weights), points, weights)
radiation = Radiation(D[:knudsen], D[:sigmaS], D[:sigmaA])
ib = nothing

ks = SolverSet(set, ps, vs, radiation, ib, pwd())

uq = UQ1D(3, 6, -0.1, 0.1)

function is_absorb(x::T, y::T) where {T<:Real}
    cds = Array{Bool}(undef, 11) # conditions

    cds[1] = -2.5<x<-1.5 && 1.5<y<2.5
    cds[2] = -2.5<x<-1.5 && -0.5<y<0.5
    cds[3] = -2.5<x<-1.5 && -2.5<y<-1.5
    cds[4] = -1.5<x<-0.5 && 0.5<y<1.5
    cds[5] = -1.5<x<-0.5 && -1.5<y<-0.5
    cds[6] = -0.5<x<0.5 && -2.5<y<-1.5
    cds[7] = 0.5<x<1.5 && 0.5<y<1.5
    cds[8] = 0.5<x<1.5 && -1.5<y<-0.5
    cds[9] = 1.5<x<2.5 && 1.5<y<2.5
    cds[10] = 1.5<x<2.5 && -0.5<y<0.5
    cds[11] = 1.5<x<2.5 && -2.5<y<-1.5

    if any(cds) == true
        return true
    else
        return false
    end
end

begin
    σs = zeros(Float64, ps.nx, ps.ny)
    σa = zeros(Float64, ps.nx, ps.ny)
    for i = 1:ps.nx, j = 1:ps.ny
        if is_absorb(ps.x[i, j], ps.y[i, j])
            σs[i, j] = 0.0
            σa[i, j] = 10.0
        else
            σs[i, j] = 1.0
            σa[i, j] = 0.0
        end
    end
    σt = σs + σa
    σq = zeros(Float64, ps.nx, ps.ny)
    for i = 1:ps.nx, j = 1:ps.ny
        if -0.5<ps.x[i, j]<0.5 && -0.5<ps.y[i, j]<0.5
            σq[i, j] = 30.0  / (4.0 * π)
        else
            σq[i, j] = 0.0
        end
    end
end


phi = zeros(vs.nu, ps.nx, ps.ny)
for j = 1:ps.nx
    for i = 1:ps.ny
        phi[:, i, j] .= 1e-4
    end
end

t = 0.0
dt = 1.2 / 150 * ks.set.cfl
flux1 = zeros(vs.nu, ps.nx + 1, ps.ny)
flux2 = zeros(vs.nu, ps.nx, ps.ny + 1)

@showprogress for iter = 1:200
    for i = 2:ps.nx, j = 1:ps.ny
        tmp = @view flux1[:, i, j]
        flux_kfvs!(tmp, phi[:, i-1, j], phi[:, i, j], points[:, 1], dt)
    end
    for i = 1:ps.nx, j = 2:ps.ny
        tmp = @view flux2[:, i, j]
        flux_kfvs!(tmp, phi[:, i, j-1], phi[:, i, j], points[:, 2], dt)
    end

    for j = 1:ps.ny, i = 1:ps.nx
        integral = discrete_moments(phi[:, i, j], weights)
        integral /= 4.0 * π

        for q = 1:vs.nu
            phi[q, i, j] =
                phi[q, i, j] +
                (flux1[q, i, j] - flux1[q, i+1, j]) / ps.dx[i, j] +
                (flux2[q, i, j] - flux2[q, i, j+1]) / ps.dy[i, j] +
                dt * σs[i, j] * integral - dt * σt[i, j] * phi[q, i, j] + σq[i, j] * dt
        end
    end

    global t += dt
end

ρ = zeros(ps.nx, ps.ny)
for i = 1:ps.nx, j = 1:ps.ny
    ρ[i, j] = discrete_moments(phi[:, i, j], weights)
end
contourf(ps.x[1:end, 1], ps.y[1, 1:end], ρ', color = :PiYG_3)
