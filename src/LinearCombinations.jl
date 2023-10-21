module LinearCombinations

using Base: Fix1, @propagate_inbounds

import Base: show, ==, hash, copy, copyto!, convert, promote_rule,
    zero, iszero, one, isone, isodd, iseven, +, -, *, /, ^,
    length, eltype, in, iterate, empty!, isempty, sizehint!,
    firstindex, lastindex, getindex, setindex!

using StructEqualHash

include("basics.jl")
include("linear.jl")
include("bangbang.jl")
include("broadcast.jl")
include("extensions.jl")
include("tensor.jl")
include("regroup.jl")

end
