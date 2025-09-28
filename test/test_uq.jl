###
# struct
###

uq1 = UQ1D(4, 8, -1.0, 1.0, "uniform", "collocation")
uq2 = UQ2D(4, 8, [-1.0, 1.0, -1.0, 1.0], ["uniform", "uniform"], "collocation")

###
# transform
###

pce1 = ran_chaos(rand(uq1.op.quad.Nquad), uq1)
pce2 = ran_chaos(rand(uq2.nq), uq2)

chaos_ran(pce1, uq1)
chaos_ran(pce2, uq2)

lambda_tchaos(pce1, 1.0, uq1)
lambda_tchaos(pce2, 1.0, uq2)

###
# mean
###

uu = chaos_ran(pce1, uq1)
collo_mean(uu, uq1)
galerkin_mean(pce1, uq1)
@test mean(uu, uq1) ≈ mean(pce1, uq1)

###
# filter
###

LV.basis_norm(uq1)
LV.adapt_filter_strength(uq1.pce, 1e-5, 1.0, uq1)

###
# limiter
###

uquad = rand(uq1.nq)
uquad[end] = -0.05 # make negative value
uchaos = ran_chaos(uquad, uq1)
LV.positive_limiter!(uchaos, uq1)
chaos_ran(uchaos, uq1)
@test minimum(chaos_ran(uchaos, uq1)) > -1e-6

uquad2 = rand(3, uq1.nq)
uquad2[end, 1] = -0.05 # make negative value
uchaos2 = ran_chaos(uquad2, 2, uq1)
LV.positive_limiter!(uchaos2, uq1)
@test minimum(chaos_ran(uchaos2, 2, uq1)) > -1e-6
