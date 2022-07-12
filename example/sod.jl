using CairoMakie, Langevin
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
ks, ctr, face, uq, t = initialize("config/sod.txt", "ctr")

t = 0.0
dt = timestep(ks, uq, ctr, t)
nt = ks.set.maxTime ÷ dt |> Int

@showprogress for iter = 1:nt
#    reconstruct!(ks, ctr)
    evolve!(ks, uq, ctr, face, dt)
    update!(ks, uq, ctr, face, dt, zeros(3))
end

sol = zeros(ks.ps.nx)
for i = 1:ks.ps.nx
    sol[i] = ctr[i].prim[1, 1]
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "n",
    title = "")
    lines!(ks.ps.x[1:ks.ps.nx], sol; label = "quantum")
    axislegend()
    fig
end
