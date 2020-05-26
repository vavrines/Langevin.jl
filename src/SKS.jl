# ============================================================
# Stochastic Kinetic Scheme
# Copyright (c) Tianbai Xiao 2020
# ============================================================

module SKS

using Reexport
using OffsetArrays
using SpecialFunctions
using PolyChaos
using FileIO
using JLD2
using Plots
#using DifferentialEquations
#using Kinetic

include("D:\\Coding\\Github\\Kinetic.jl\\src\\Kinetic.jl")
@reexport using .Kinetic

include("uq.jl")
include("kinetic.jl")
include("initialize.jl")
include("flux.jl")
include("out.jl")
include("solver.jl")

end
