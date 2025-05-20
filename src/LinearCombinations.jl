"""
    $(@__MODULE__)

A Julia package to work with formal linear combinations, tensors
and linear as well as multilinear maps.
The terms appearing in a linear combination can be of any type,
and coefficients can be in any commutative ring with unit.
The overall aim of the package is to provide functions that are efficient and easy to use.

See [`AbstractLinear`](@ref), [`Tensor`](@ref), [`@linear`](@ref), [`@multilinear`](@ref).
"""
module LinearCombinations

using Base: Fix1, @propagate_inbounds

import Base: show, ==, hash, copy, copyto!, convert, promote_rule,
    zero, iszero, one, isone, isodd, iseven, +, -, *, /, ^,
    length, eltype, in, iterate, sizehint!,
    firstindex, lastindex, getindex, setindex!

using StructEqualHash

include("basics.jl")
include("abstractlinear.jl")
include("linear.jl")
include("linear1.jl")
include("bangbang.jl")
include("broadcast.jl")
include("extensions.jl")
include("tensor.jl")
include("denselinear.jl")
include("regroup.jl")

include("helpers.jl")

end
