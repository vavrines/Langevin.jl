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
# filter
###

LV.basis_norm(uq1)
LV.adapt_filter_strength(uq1.pce, 1e-5, 1.0, uq1) 
