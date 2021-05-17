"""
Stochastic Scheme for Uncertainty Quantification
Copyright (c) 2020-2021 Tianbai Xiao
"""

module Langevin

using Reexport
@reexport using PolyChaos
@reexport using KitBase
using KitBase.OffsetArrays
using KitBase.Plots
using KitBase.JLD2

export AbstractUQ,
       UQ1D,
       UQ2D,
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

include("struct.jl")
include("uq.jl")
include("filter.jl")
include("kinetic.jl")
include("initialize.jl")
include("flux_ctr.jl")
include("flux_sol.jl")
include("out.jl")
include("solver.jl")
include("step.jl")

end
