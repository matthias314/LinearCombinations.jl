#
# AbstractBasis
#

export AbstractBasis, tobasis, toindex

abstract type AbstractBasis{T} end

eltype(::Type{<:AbstractBasis{T}}) where T = T

function tobasis end
function toindex end

#
# Basis
#

export Basis

struct Basis{T,V<:AbstractVector{T}} <: AbstractBasis{T}
    list::V
    invlist::Dict{T,Int}
end

function Basis(iter)
    list = iter isa AbstractVector ? iter : collect(iter)
    T = eltype(list)
    invlist = Dict{T,Int}(list[i] => i for i in eachindex(list))
    Basis(list, invlist)
end

show(io::IO, b::Basis{T}) where T = print(io, "Basis{", T, "}(", repr(b.list), ')')

length(b::Basis) = length(b.list)

@propagate_inbounds iterate(b::Basis, s...) = iterate(b.list, s...)

in(b::Basis, x) = haskey(b.invlist, x)

@propagate_inbounds tobasis(b::Basis, i) = b.list[i]

toindex(b::Basis, x) = b.invlist[x]

@propagate_inbounds function tobasis(bs::Tuple{Vararg{Basis}}, is)
    Tensor(ntuple(k -> tobasis(bs[k], is[k]), length(bs)))
end

function toindex(bs::Tuple{Vararg{Basis}}, xs::Tensor)
    CartesianIndex(ntuple(k -> toindex(bs[k], xs[k]), length(bs)))
end

#
# DenseLinear
#

export DenseLinear

struct DenseLinear{T,R,B,V} <: AbstractLinear{T,R}
    v::V
    b::B
    @inline function DenseLinear{T,R}(v::V; basis::B) where {T,R,B<:Union{Basis,Tuple{Vararg{Basis}}},V<:AbstractArray}
        @boundscheck if B <: Basis
            eltype(basis) <: T || error("term type and basis are not compatible")
            axes(v) == axes(basis.list) || error("vector and basis must have the same indices")
        else # B <: Tuple{Vararg{Basis}}
            T <: Tensor || error("term type must be <:Tensor")
            TT = T.parameters[1].parameters
            (length(TT) == length(basis) && all(ntuple(i -> eltype(basis[i]) <: TT[i], length(TT)))) ||
                error("term type and bases are not compatible")
            axes(v) == map(b -> only(axes(b.list)), basis) ||
                error("array and bases must have the same indices")
        end

        new{T,R,B,V}(v, basis)
    end
end

# @propagate_inbounds currently has no effect
@propagate_inbounds DenseLinear{T}(v::AbstractArray{R}; basis::Union{Basis,Tuple{Vararg{Basis}}}) where {T,R} = DenseLinear{T,R}(v; basis)

@propagate_inbounds DenseLinear(v::AbstractVector{R}; basis::Basis{T}) where {R,T} = DenseLinear{T,R}(v; basis)

@propagate_inbounds function DenseLinear(v::AbstractArray{R}; basis::Tuple{Vararg{Basis}}) where R
    TT = Tensor{Tuple{map(b -> typeof(b).parameters[1], basis)...}}
    DenseLinear{TT,R}(v; basis)
end

change_coefftype(::Type{DenseLinear{T,R,B,V}}, ::Type{S}) where {T,R,B,V,S} = DenseLinear{T,S,B,V}

function Base.:(==)(a::DenseLinear, b::DenseLinear)
    if a.b === b.b
        a.v == b.v
    else
        invoke(==, Tuple{AbstractLinear,AbstractLinear}, a, b)
    end
end

function zero(::Type{<:DenseLinear{T,R}}; basis::Union{Basis,Tuple{Vararg{Basis}}}) where {T,R}
    l = basis isa Basis ? length(basis) : map(length, basis)
    @inbounds DenseLinear{T,R}(zeros(R, l); basis)
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

#=
tocartesian(l::Tuple{}, ii::Tuple) = CartesianIndex()
tocartesian(l::Tuple{Integer}, ii::Tuple) = CartesianIndex(ii)

function tocartesian(l::Tuple, ii::Tuple)
    d, r = divrem(ii[end]-1, l[1])
    tocartesian(l[2:end], (ii[1:end-1]..., r+1, d+1))
end

function iterate(a::DenseLinear{T,R}, i = firstindex(a.v)) where {T,R}
    while i <= lastindex(a.v)
        @inbounds c = a.v[i]
        if a.b isa Tuple && i isa Integer
            ii = tocartesian(size(a.v), (i,))
        else
            ii = i
        end
        if !iszero(c)
            @inbounds x = tobasis(a.b, ii)
            return (Pair{T,R}(x, c), i+1)
        end
        i += 1
    end
    nothing
end
=#

function iterate(a::DenseLinear{T,R}, ss...) where {T,R}
    while (iis = iterate(CartesianIndices(a.v), ss...)) !== nothing
        ii, s = iis
        c = @inbounds a.v[ii]
        if !iszero(c)
            @inbounds x = tobasis(a.b, ii)
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
    if iszero(convert(R, c))
        zero!(a)
    else
        a.v .*= c
    end
    a
end
