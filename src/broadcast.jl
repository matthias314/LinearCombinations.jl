#
# broadcasting
#

using Base.Broadcast: Broadcasted, BroadcastStyle, DefaultArrayStyle
import Base.Broadcast: instantiate

struct LinearStyle <: BroadcastStyle end

export @linear_broadcastable

macro linear_broadcastable(T)
    quote
        Broadcast.BroadcastStyle(::Type{<:$(esc(T))}) = LinearStyle()
        Broadcast.broadcastable(x::$(esc(T))) = x
    end
end

@linear_broadcastable Linear
@linear_broadcastable Sign

Base.axes(::Linear) = nothing

BroadcastStyle(::DefaultArrayStyle{0}, style::LinearStyle) = style
# needed for scalars

# copy

instantiate(bc::Broadcasted{LinearStyle}) = bc
# needed for copy, but not for copyto!

function copy(bc::Broadcasted{LinearStyle, Nothing, typeof(*)})
    bca1, bca2 = bc.args
    # TODO: we assume that bca1 is the scalar
    mul!(copy(bca2), bca1)
    # do we need to worry about broadcasting of scalars, as in .- 3 .* a ?
    # mul!(copy(bca2), bca1 isa Broadcasted ? copy(bca1) : bca1)
end

function copy(bc::Broadcasted{LinearStyle, Nothing, typeof(+), <:Tuple{Any}})
    bca1 = bc.args[1]
    # TOOD: always copy? (that's done for .+ [1,2,3])
    bca1 isa Broadcasted ? copy(bca1) : bca1
end

function copy(bc::Broadcasted{LinearStyle, Nothing, typeof(+), <:Tuple{Any, Any}})
# TODO: promote types
    bca1, bca2 = bc.args
    _copyto!(copy(bca1), bca2, ONE)
end

function copy(bc::Broadcasted{LinearStyle, Nothing, typeof(-), <:Tuple{Any}})
    bca1 = bc.args[1]
    -(bca1 isa Broadcasted ? copy(bca1) : bca1)
end

function copy(bc::Broadcasted{LinearStyle, Nothing, typeof(-), <:Tuple{Any, Any}})
# TODO: promote types
    bca1, bca2 = bc.args
    _copyto!(copy(bca1), bca2, -ONE)
end

# copyto!

function copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(identity)})
    bca1 = bc.args[1]
    if a !== bca1
        empty!(a)
        _copyto!(a, bca1, ONE)
    end
    a
end

function copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(*)})
    bca1, bca2 = bc.args
    if a === bca1
        mul!(bca1, bca2)
    else
        empty!(a)
        _copyto!(a, bca2, bca1)
        # TODO: we assume that bca1 is the scalar
    end
end

function copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(+)})
    bca1, bca2 = bc.args
    if a !== bca1
        empty!(a)
        _copyto!(a, bca1, ONE)
    end
    _copyto!(a, bca2, ONE)
end

function copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(-)})
    bca1, bca2 = bc.args
    if a !== bca1
        empty!(a)
        _copyto!(a, bca1, ONE)
    end
    _copyto!(a, bca2, -ONE)
end

function _copyto!(a::Linear, b::Linear, c::Sign)
    isone(c) ? add!(a, b) : sub!(a, b)
end

function _copyto!(a::Linear, b::Linear, c)
    addmul!(a, b, c)
end

function _copyto!(a::Linear{T}, x::T, c) where T
    # addcoeff!(a, x, c)
    addmul!(a, x, c)
end

# This is for types T that are converted to Array{T, 0}
function _copyto!(a::Linear{T}, x::Array{T, 0}, c) where T
    # addcoeff!(a, x[], c)
    addmul!(a, x[], c)
end

function _copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(*)}, c)
    bca1, bca2 = bc.args
    # TODO: this assumes that bca1 is the scalar
    _copyto!(a, bca2, c * bca1)
    # do we need to worry about broadcasting of scalars, as in .- 3 .* a ?
    # _copyto!(a, bca2, c * (bca1 isa Broadcasted ? copy(bca1) : bca1))
end

function _copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(+), <:Tuple{Any}}, c)
    bca = bc.args
    _copyto!(a, bca[1], c)
end

function _copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(+), <:Tuple{Any, Any}}, c)
    bca1, bca2 = bc.args
    _copyto!(a, bca1, c)
    _copyto!(a, bca2, c)
end

function _copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(-), <:Tuple{Any}}, c)
    bca = bc.args
    _copyto!(a, bca[1], -c)
end

function _copyto!(a::Linear, bc::Broadcasted{LinearStyle, Nothing, typeof(-), <:Tuple{Any, Any}}, c)
    bca1, bca2 = bc.args
    _copyto!(a, bca1, c)
    _copyto!(a, bca2, -c)
end
