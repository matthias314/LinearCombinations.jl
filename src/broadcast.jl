#
# broadcasting
#

using Base.Broadcast: Broadcasted, DefaultArrayStyle
import Base.Broadcast: BroadcastStyle, instantiate

const Tuple1 = Tuple{Any}
const Tuple2 = Tuple{Any,Any}

"""
    $(@__MODULE__).LinearStyle

The broadcasting style used for linear combinations.

See also [`@linear_broadcastable`](@ref).
"""
struct LinearStyle <: BroadcastStyle end

export @linear_broadcastable

"""
    @linear_broadcastable T

Add the type `T` to the types that participate in broadcasting for linear combinations.
By default, only the types `AbstractLinear` and `Number` are available. (A few others
happen to work as well, for example `AbstractChar`.)

See also [`$(@__MODULE__).LinearStyle `](@ref).
"""
macro linear_broadcastable(T)
    quote
        Broadcast.BroadcastStyle(::Type{<:$(esc(T))}) = LinearStyle()
        Broadcast.broadcastable(x::$(esc(T))) = x
    end
end

@linear_broadcastable AbstractLinear
@linear_broadcastable Sign

Base.axes(::AbstractLinear) = nothing

BroadcastStyle(::DefaultArrayStyle{0}, style::LinearStyle) = style
# needed for scalars

# copy

instantiate(bc::Broadcasted{LinearStyle}) = bc
# needed for copy, but not for copyto!

copy(::Broadcasted{LinearStyle}) = error("broadcasting not implemented for this operation")
# fallback with meaningful error message

# TODO: do we need to worry about broadcasting of scalars, as in .- 3 .* a ?
function copy(bc::Broadcasted{LinearStyle, Nothing, Mul, <:Tuple2})
    bca1, bca2 = bc.args
    if bca1 isa Broadcasted
        mul!(copy(bca1), bca2)
    elseif bca2 isa Broadcasted
        mul!(copy(bca2), bca1)
    else
        bca1*bca2
    end
end

function copy(bc::Broadcasted{LinearStyle, Nothing, Add, <:Tuple1})
    # copying cannot be omitted, for example in ".+ a .+ a"
    copy(bc.args[1])
end

function copy(bc::Broadcasted{LinearStyle, Nothing, Add})
# TODO: promote types
    foldl(addmul!, bc.args[2:end]; init = copy(bc.args[1]))
    # addmul!(copy(bc.args[1]), Broadcast.broadcasted(+, bc.args[2:end]...)
end

function copy(bc::Broadcasted{LinearStyle, Nothing, Sub, <:Tuple1})
    mul!(copy(bc.args[1]), -ONE)
end

function copy(bc::Broadcasted{LinearStyle, Nothing, Sub, <:Tuple2})
# TODO: promote types
    addmul!(copy(bc.args[1]), bc.args[2], -ONE)
end

# copyto!

copyto!(::AbstractLinear, ::Broadcasted{LinearStyle}) = error("broadcasting not implemented for this operation")
# fallback with meaningful error message

function copyto!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, typeof(identity), <:Tuple1})
    copyto!(a, bc.args[1])
end

function copyto!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Mul, <:Tuple2})
# we support two formats: a, b are AbstractLinear, c is a scalar
# 1st format:  a .*= c that is, a .= a .* c,  2nd format:  a .= c .* b
    if a === bc.args[1]
        mul!(a, bc.args[2])
    else
        copyto!(a, bc.args[2], bc.args[1])
    end
end

function copyto!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Add, <:Tuple1})
    copyto!(a, bc.args[1])
end

function copyto!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Sub, <:Tuple1})
    copyto!(a, bc.args[1], -ONE)
end

function copyto!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Add})
    foldl(addmul!, bc.args[2:end]; init = copyto!(a, bc.args[1]))
    # addmul!(copy(bc.args[1]), Broadcast.broadcasted(+, bc.args[2:end]...)
end

function copyto!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Sub, <:Tuple2})
    copyto!(a, bc.args[1])
    addmul!(a, bc.args[2], -ONE)
end

# addmul!

# This is for types T that are converted to Array{T, 0}
function addmul!(a::AbstractLinear, x::Array{T, 0}, c = ONE) where T
    addmul!(a, x[], c)
end

function addmul!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Mul, <:Tuple2}, c = ONE)
    bca1, bca2 = bc.args
    # we assume that bca1 is the scalar
    addmul!(a, bca2, c * bca1)
    # do we need to worry about broadcasting of scalars, as in .- 3 .* a ?
end

function addmul!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Add}, c = ONE)
    foldl((a, bca) -> addmul!(a, bca, c), bc.args; init = a)
end

function addmul!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Sub, <:Tuple1}, c = ONE)
    addmul!(a, bc.args[1], bc.f(c))
end

function addmul!(a::AbstractLinear, bc::Broadcasted{LinearStyle, Nothing, Sub, <:Tuple2}, c = ONE)
    addmul!(a, bc.args[1], c)
    addmul!(a, bc.args[2], bc.f(c))
end
