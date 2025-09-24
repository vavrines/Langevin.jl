"""
$(README)

## Exports

$(EXPORTS)
"""
module Langevin

using Reexport
@reexport using KitBase
@reexport using PolyChaos
using KitBase.FiniteMesh.DocStringExtensions
using KitBase.JLD2
using KitBase.LinearAlgebra
using KitBase.OffsetArrays
using KitBase: AV, AM, AA, AVOM
using KitBase:
    AbstractSolution,
    AbstractSolution1D,
    AbstractPhysicalSpace1D,
    AbstractPhysicalSpace2D,
    AbstractSolverSet,
    AbstractControlVolume,
    AbstractControlVolume1D,
    AbstractInterface1D,
    AbstractFlux
using Base.Threads: @threads
using Statistics: mean

export LV
export AbstractUQ, UQ1D, UQ2D
export ran_chaos, chaos_ran, lambda_tchaos, t_lambdachaos, chaos_product, chaos_product!
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
include("limiter.jl")
include("kinetic.jl")
include("initialize.jl")
include("Flux/flux.jl")
include("out.jl")
include("Solver/solver.jl")

const LV = Langevin

end
