"""
Stochastic Scheme for Uncertainty Quantification

Copyright (c) 2020-2022 Tianbai Xiao
"""

module Langevin

using Reexport
@reexport using KitBase
@reexport using PolyChaos
using KitBase.FiniteMesh.DocStringExtensions
using KitBase.JLD2
using KitBase.LinearAlgebra
using KitBase.OffsetArrays
using KitBase: AV, AM, AA
using Base.Threads: @threads

export LV
export AbstractUQ, UQ1D, UQ2D
export ran_chaos, chaos_ran, lambda_tchaos, t_lambdachaos
export uq_moments_conserve,
    uq_maxwellian,
    uq_energy_distribution,
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
include("Flux/flux.jl")
include("out.jl")
include("Solver/solver.jl")

const LV = Langevin

end
