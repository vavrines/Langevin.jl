# ============================================================
# Stochastic Kinetic Scheme
# Copyright (c) Tianbai Xiao 2020
# ============================================================

module SKS

using Reexport
using OffsetArrays
using Plots
using Kinetic
@reexport using PolyChaos

include("uq.jl")
include("kinetic.jl")
include("initialize.jl")
include("flux.jl")
include("out.jl")
include("solver.jl")

end
