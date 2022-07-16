# ============================================================
# Full Solver
# ============================================================

cd(@__DIR__)

###
# ctr-face
###

# 1d1f1v
ks, ctr, face, uq, simTime = Langevin.initialize("../example/config/sod.txt", :ctr)
dt = timestep(ks, uq, ctr, simTime)
res = zeros(3)
evolve!(ks, uq, ctr, face, dt)
update!(ks, uq, ctr, face, dt, res)

# 1d2f1v
ks, ctr, face, uq, simTime = Langevin.initialize("../example/config/shock.txt", :ctr)
evolve!(ks, uq, ctr, face, dt)
update!(ks, uq, ctr, face, dt, res)

###
# sol-flux
###

# 1d1f1v
ks, sol, flux, uq, simTime = Langevin.initialize("../example/config/sod.txt", :sol)
dt = timestep(ks, uq, sol, simTime)
res = zeros(3)
evolve!(ks, uq, sol, flux, dt)
update!(ks, uq, sol, flux, dt, res)

# 1d2f1v
ks, sol, flux, uq, simTime = Langevin.initialize("../example/config/shock.txt", :sol)
dt = timestep(ks, uq, sol, simTime)
res = zeros(3)
evolve!(ks, uq, sol, flux, dt)
update!(ks, uq, sol, flux, dt, res)

# 2d2f2v
ks, sol, flux, uq, simTime = Langevin.initialize("../example/config/cavity.txt", :sol)
dt = timestep(ks, uq, sol, simTime)
simTime = 0.0
res = zeros(4)
evolve!(ks, uq, sol, flux, dt)
update!(ks, uq, sol, flux, dt, res)
