# ============================================================
# Stochastic Kinetic Scheme for Uncertainty Quantification
# Copyright (c) Tianbai Xiao 2020
# ============================================================

module Langevin

using Reexport
using OffsetArrays
using Plots
using JLD2
using Kinetic
@reexport using PolyChaos

export AbstractUQ, 
       UQ1D, 
       ran_chaos, 
       chaos_ran, 
       lambda_tchaos, 
       t_lambdachaos, 
       filter!
export uq_moments_conserve, 
       uq_maxwellian,
       uq_prim_conserve,
       uq_conserve_prim,
       uq_conserve_prim!,
       uq_prim_conserve!,
       uq_sound_speed,
       uq_vhs_collision_time,
       uq_aap_hs_collision_time,
       uq_aap_hs_prim

include("uq.jl")
include("kinetic.jl")
include("initialize.jl")
include("flux.jl")
include("out.jl")
include("solver.jl")
include("step.jl")

end
