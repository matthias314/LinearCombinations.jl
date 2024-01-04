#
# AbstractBasis
#

export AbstractBasis, toindex

import Base: getindex, axes, size

"""
    AbstractBasis{T,N} <: AbstractArray{T,N}

The supertype of all types representing bases whose elements are of type `T`.
Bases are needed for linear combinations of type `DenseLinear`.

All subtypes of `AbstractBasis` must implement the
[abstract arrays interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
For a given basis `b`, `b[i]` is the i-th basis element. Mapping from basis elements to indices
is one via the `toindex` function. The parameter `N == ndims(b)` specifies how many indices are used
to index elements of a basis `b`.

See also [`Basis`](@ref), [`TensorBasis`](@ref), [`toindex`](@ref), `Base.ndims`.
"""
abstract type AbstractBasis{T,N} <: AbstractArray{T,N} end

show(io::IO, ::MIME"text/plain", b::AbstractBasis) = show(io, b)

size(b::AbstractBasis) = map(length, axes(b))

"""
    toindex(b::AbstractBasis{T,N}, x) where {T,N} -> CartesianIndex{N}

Return the Cartesian index of the element `x` in the basis `b`.

See also [`AbstractBasis`](@ref), `Base.CartesianIndex`.
"""
toindex(b::AbstractBasis, x) = error_missing(typeof(b))

"""
    getindex(b::AbstractBasis{T,N}, ii::Vararg{Int,N}) -> T

    b[ii...] -> T

Return the basis element indexed by the indices `ii`.

See also [`AbstractBasis`](@ref), [`toindex`](@ref), `Base.CartesianIndex`.
"""
getindex(b::AbstractBasis{T,N}, ii::Vararg{Int,N}) where {T,N} = error_missing(typeof(b))

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

"""
    Basis{T,N} <: AbstractBasis{T,N}

    Basis(iter)

Construct a `Basis` whose elements are the elements of the `AbstractArray` or iterator `iter`. Internally,
basis elements are stored in the given `AbstractArray` or otherwise in the `Array` obtained from `collect(iter)`.

See also [`AbstractBasis`](@ref), [`TensorBasis`](@ref).

# Examples
```jldoctest
julia> b = Basis(['a', 'b', 'x', 'y', 'z'])
Basis(['a', 'b', 'x', 'y', 'z'])

julia> length(b)
5

julia> i = toindex(b, 'x')
CartesianIndex(3,)

julia> b[i], b[3]
('x', 'x')

julia> length(Basis(Char[]))
0
```
"""
Basis(iter) = Basis(collect(iter))

show(io::IO, b::Basis) = print(io, "Basis(", repr(b.list), ')')

axes(b::Basis) = axes(b.list)

==(b1::Basis, b2::Basis) = b1.list == b2.list

@propagate_inbounds iterate(b::Basis, s...) = iterate(b.list, s...)

@propagate_inbounds getindex(b::Basis{T,N}, ii::Vararg{Int,N}) where {T,N} = b.list[ii...]

in(x, b::Basis) = haskey(b.invlist, x)

toindex(b::Basis, x) = b.invlist[x]

#
# TensorBasis
#

export TensorBasis

"""
    TensorBasis{T,N} <: AbstractBasis{T,N}

    TensorBasis(bases...)

Construct a `TensorBasis` out of the given bases. The elements of the `TensorBasis` are of type `Tensor`,
where the `i`-th tensor component is from the `i`-th basis.

See also [`AbstractBasis`](@ref), [`Basis`](@ref).

# Examples
```jldoctest
julia> b1, b2 = Basis('a':'c'), Basis(["x", "y", "z"])
(Basis('a':1:'c'), Basis(["x", "y", "z"]))

julia> b3 = TensorBasis(b1, b2)
TensorBasis(Basis('a':1:'c'), Basis(["x", "y", "z"]))

julia> length(b3)
9

julia> toindex(b3, Tensor('b', "z"))
CartesianIndex(2, 3)

julia> b4 = TensorBasis(b3, b1)
TensorBasis(TensorBasis(Basis('a':1:'c'), Basis(["x", "y", "z"])), Basis('a':1:'c'))

julia> ndims(b4)
3

julia> x = first(b4)
(a⊗x)⊗a

julia> toindex(b4, x)
CartesianIndex(1, 1, 1)

julia> b0 = TensorBasis(); length(b0), Tensor() in b0
(1, true)
```
"""
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
    join(io, (repr(basis) for basis in b.bases), ", ")
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

@propagate_inbounds getindex(b::TensorBasis{T,N}, ii::Vararg{Int,N}) where {T,N} = _tobasis(b.bases, ii)

in(t::Tensor, b::TensorBasis) = all(splat(in), zip(t, b.bases))

function toindex(b::TensorBasis{T}, t::U) where {T,U<:T}
    length(b.bases) == length(t) || error("TensorBasis and Tensor arguments must have the same length")
    CartesianIndex(ntuple(k -> toindex(b.bases[k], t[k]), length(b.bases)))
end

#
# DenseLinear
#

export DenseLinear

"""
    DenseLinear{T,R} <: AbstractLinear{T,R}

    DenseLinear{T,R}(itr; basis::Basis)

Construct a linear combination of type `DenseLinear` with term type `T` and
coefficient type `R` out of the term-coefficient pairs provided by the iterator `itr`.

Linear combinations of this type are internally stored as a `Vector` (or, more generally,
an `AbstractArray`). The mandatory keyword argument `basis` is used to translate between
terms and entries of the `Vector` (or `Array`). Operations involving two `DenseLinear`
elements are much faster when the two bases are identical (in the sense of `===`).

Other ways to use this constructor are discussed under `AbstractLinear`.

See also [`Basis`](@ref), [`AbstractLinear`](@ref), [`Linear`](@ref), [`Linear1`](@ref).

# Examples

```jldoctest
julia> azbasis = Basis('a':'z')
Basis('a':1:'z')

julia> a = DenseLinear('x' => 1, 'y' => 2; basis = azbasis)
x+2*y

julia> a + 'z'
x+2*y+z

julia> typeof(ans)
DenseLinear{Char, Int64, Basis{Char, 1, StepRange{Char, Int64}}, Vector{Int64}}

julia> a + 'X'
ERROR: KeyError: key 'X' not found
[...]

julia> b = DenseLinear('x' => -1, 'z' => 3; basis = azbasis)
-x+3*z

julia> a + b
2*y+3*z

julia> typeof(ans)
Linear{Char, Int64}

julia> c = DenseLinear('a' => 5; basis = Basis('a':'c'))
5*a

julia> add!(a, c)
5*a+x+2*y

julia> add!(c, a)
ERROR: KeyError: key 'x' not found
```
"""
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

function getcoeff(a::DenseLinear, x)
    i = toindex(a.b, unhash(x))
    @inbounds a.v[i]
end

function setcoeff!(a::DenseLinear, c, x)
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

"""
    matrixrepr!(f, a::AbstractMatrix, b1::AbstractBasis, b0::AbstractBasis; iszero::Bool = false) -> a

Compute a matrix representing the linear map `f` with with respect to the bases `b0` (source)
and `b1` (target). Store the result in `a` and return `a`. If the keyword argument `iszero` is
`true`, then the matrix `a` is not first initialized with zeros.

See also [`matrixrepr`](@ref).
"""
function matrixrepr!(f, a::AbstractMatrix{R}, b1::AbstractBasis, b0::AbstractBasis{T}; iszero::Bool = false) where {R,T}
    axes(a) == (OneTo(length(b1)), OneTo(length(b0))) || error("matrix has wrong dimensions")
    iszero || fill!(a, zero(R))
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

"""
    matrixrepr(f, b1::AbstractBasis, b0::AbstractBasis, ::Type{R})) where R

Return a matrix representing the linear map `f` with respect to the bases `b0` (source)
and `b1` (target). Coefficients have the type `R`.

See also [`matrixrepr!`](@ref).

# Example
```jldoctest
julia> @linear f; f(x) = Linear(uppercase(x) => 1, 'A' => 1)
f (generic function with 2 methods)

julia> matrixrepr(f, Basis('A':'C'), Basis('a':'c'), Int)
3×3 Matrix{Int64}:
 2  1  1
 0  1  0
 0  0  1
```
"""
function matrixrepr(f, b1::AbstractBasis, b0::AbstractBasis{T}, ::Type{R}) where {T,R}
    a = zeros(R, length(b1), length(b0))
    matrixrepr!(f, a, b1, b0; iszero = true)
end
