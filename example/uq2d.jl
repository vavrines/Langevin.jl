using Langevin

uq = UQ2D(4, 8, [-1.0, 1.0, -1.0, 1.0], ["uniform", "uniform"], "collocation")

solRan = rand(49)
for i = 1:49
    solRan[i] = 5 + uq.points[i, 1] + uq.points[i, 2]
end

pce = ran_chaos(solRan, uq)
evaluatePCE(pce, uq.points, uq.op)

mean(pce, uq.op)
std(pce, uq.op)
