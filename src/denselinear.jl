#
# AbstractBasis
#

export AbstractBasis, toindex

import Base: getindex, axes, size

abstract type AbstractBasis{T,N} <: AbstractArray{T,N} end

show(io::IO, ::MIME"text/plain", b::AbstractBasis) = show(io, b)

size(b::AbstractBasis) = map(length, axes(b))

function toindex end

#
# Basis
#

export Basis

struct Basis{T,N,V<:AbstractArray{T,N}} <: AbstractBasis{T,N}
    list::V
    invlist::Dict{T,CartesianIndex{N}}
end

function Basis(list::AbstractArray{T,N}) where {T,N}
    invlist = Dict{T,CartesianIndex{N}}(list[i] => i for i in eachindex(IndexCartesian(), list))
    Basis(list, invlist)
end

Basis(iter) = Basis(collect(iter))

show(io::IO, b::Basis) = print(io, "Basis(", repr(b.list), ')')

axes(b::Basis) = axes(b.list)

==(b1::Basis, b2::Basis) = b1.list == b2.list

@propagate_inbounds iterate(b::Basis, s...) = iterate(b.list, s...)

@propagate_inbounds getindex(b::Basis, ii::Integer...) = b.list[ii...]

in(x, b::Basis) = haskey(b.invlist, x)

toindex(b::Basis, x) = b.invlist[x]

#
# TensorBasis
#

export TensorBasis

struct TensorBasis{T,N,BT<:Tuple{Vararg{AbstractBasis}}} <: AbstractBasis{T,N}
    bases::BT
    function TensorBasis(bases::AbstractBasis...)
        TT = Tuple{map(eltype, bases)...}
        T = isconcretetype(TT) ? Tensor{TT} : Tensor{<:TT}
        N = sum(ndims, bases; init = 0)
        new{T,N,typeof(bases)}(bases)
    end
end

function show(io::IO, b::TensorBasis)
    print(io, "TensorBasis(")
    join(io, (repr(basis) for basis in b.bases), ',')
    print(io, ')')
end

axes(b::TensorBasis) = _cat(map(axes, b.bases)...)

==(b1::TensorBasis, b2::TensorBasis) = b1.bases == b2.bases

@propagate_inbounds iterate(b::TensorBasis, state...) =
    iterate(Iterators.map(Tensor, Iterators.product(b.bases...)), state...)

_tobasis(::Tuple{}, ::Tuple{}, x...) = Tensor(x)

@propagate_inbounds function _tobasis(t::Tuple, ii, x...)
    l1 = ndims(t[1])
    x1 = t[1][ii[1:l1]...]
    _tobasis(t[2:end], ii[l1+1:end], x..., x1)
end

@propagate_inbounds getindex(b::TensorBasis{T,N}, ii::Vararg{Integer,N}) where {T,N} = _tobasis(b.bases, ii)

in(t::Tensor, b::TensorBasis) = all(splat(in), zip(t, b.bases))

function toindex(b::TensorBasis{T}, t::U) where {T,U<:T}
    length(b.bases) == length(t) || error("TensorBasis and Tensor arguments must have the same length")
    CartesianIndex(ntuple(k -> toindex(b.bases[k], t[k]), length(b.bases)))
end

#
# DenseLinear
#

export DenseLinear

struct DenseLinear{T,R,B,A} <: AbstractLinear{T,R}
    v::A
    b::B
    @inline function DenseLinear{T,R}(a::A; basis::B) where {T,R,N,B<:AbstractBasis{<:T,N},A<:AbstractArray{R,N}}
        @boundscheck axes(a) == axes(basis) || error("array and basis must have the same dimensions")
        new{T,R,B,A}(a, basis)
    end
end

# @propagate_inbounds currently has no effect
@propagate_inbounds DenseLinear{T}(v::AbstractArray{R,N}; basis::AbstractBasis{<:T,N}) where {T,R,N} = DenseLinear{T,R}(v; basis)

@propagate_inbounds DenseLinear(v::AbstractArray{R,N}; basis::AbstractBasis{T,N}) where {R,T,N} = DenseLinear{T,R}(v; basis)

change_coefftype(::Type{DenseLinear{T,R,B,V}}, ::Type{S}) where {T,R,B,V,S} = DenseLinear{T,S,B,V}

function Base.:(==)(a::DenseLinear, b::DenseLinear)
    if a.b === b.b
        a.v == b.v
    else
        invoke(==, Tuple{AbstractLinear,AbstractLinear}, a, b)
    end
end

function zero(::Type{<:DenseLinear{T,R}}; basis::AbstractBasis{<:T}) where {T,R}
    @inbounds DenseLinear{T,R}(zeros(R, axes(basis)); basis)
end

zero(a::DenseLinear{T,R}) where {T,R} = @inbounds DenseLinear{T,R}(zero(a.v); basis = a.b)

iszero(a::DenseLinear) = iszero(a.v)

length(a::DenseLinear) = count(!iszero, a.v)

copy(a::DenseLinear{T,R}) where {T,R} = @inbounds DenseLinear{T,R}(copy(a.v); basis = a.b)

function zero!(a::DenseLinear{T,R}) where {T,R}
    fill!(a.v, zero(R))
    a
end

coeffs(a::DenseLinear) = Iterators.filter(!iszero, a.v)

terms(a::DenseLinear) = Iterators.map(first, a)

in(x, a::DenseLinear) = !iszero(a[x])

function iterate(a::DenseLinear{T,R}, ss...) where {T,R}
    while (iis = iterate(CartesianIndices(a.v), ss...)) !== nothing
        ii, s = iis
        c = @inbounds a.v[ii]
        if !iszero(c)
            @inbounds x = a.b[ii]
            return (Pair{T,R}(x, c), s)
        end
        ss = (s,)
    end
    nothing
end

function getindex(a::DenseLinear, x)
    i = toindex(a.b, unhash(x))
    @inbounds a.v[i]
end

function setindex!(a::DenseLinear, c, x)
    i = toindex(a.b, unhash(x))
    @inbounds a.v[i] = c
end

function modifycoeff!(op, a::DenseLinear, x, c)
    i = toindex(a.b, unhash(x))
    @inbounds a.v[i] = op(a.v[i], c)
    a
end

function modifylinear!(op::OP, a::DenseLinear, b::DenseLinear, c = missing) where OP
    if a.b !== b.b
        invoke(modifylinear!, Tuple{OP, AbstractLinear,AbstractLinear,Any}, op, a, b, c)
    else
        bc = c === missing ? b.v : Base.broadcasted(*, c, b.v)
        Base.materialize!(a.v, Base.broadcasted(op, a.v, bc))
    end
    a
end

function -(a::DenseLinear{T,R}) where {T,R}
    has_char2(R) ? a : @inbounds DenseLinear{T,R}(-a.v; basis = a.b)
end

function mul!(a::DenseLinear{T,R}, c) where {T,R}
    c1::R = c
    if iszero(c1)
        zero!(a)
    elseif !isone(c1)
        a.v .*= c isa Sign ? c*1 : c
    end
    a
end

function copyto!(a::DenseLinear, b::DenseLinear, c = ONE)
    if iszero(c)
        zero!(a)
    elseif a.b !== b.b
        invoke(copyto!, Tuple{AbstractLinear,AbstractLinear,Any}, a, b, c)
    elseif isone(c)
        copyto!(a.v, b.v)
    elseif c == -ONE
        a.v .= .- b.v
    else
        a.v .= c .* b.v
    end
    a
end

#
# matrix representation
#

export matrixrepr!, matrixrepr

using Base: OneTo

function matrixrepr!(a::AbstractMatrix, f, b1::AbstractBasis, b0::AbstractBasis{T}) where T
    axes(a) == (OneTo(length(b1)), OneTo(length(b0))) || error("matrix has wrong dimensions")
    axes1 = axes(b1)
    for (i, x) in enumerate(b0)
        addto = @inbounds DenseLinear(reshape(view(a, :, i), axes1); basis = b1)
        if has_addto_coeff(f, T)
            f(x; addto)
        else
            add!(addto, f(x))
        end
    end
    a
end

function matrixrepr(f, b1::AbstractBasis, b0::AbstractBasis{T}, ::Type{R}) where {T,R}
    a = zeros(R, length(b1), length(b0))
    matrixrepr!(a, f, b1, b0)
end
