#
# AbstractLinear abstract data type
#

export linear_filter, AbstractLinear,
    coefftype, coeffs, termtype, terms,
    addmul!, add!, sub!, mul!, addmul, zero!

"""
    L(xc::Pair...; is_filtered = false; kw...) where L <: AbstractLinear
    L(itr; is_filtered = false; kw...) where L <: AbstractLinear

`AbstractLinear{T,R}` is the supertype of all linear combinations
with term type `T` and coefficient type `R`.

A constructor for a subtype `L <: AbstractLinear` constructs
a linear combination of type `L` out of the given term-coefficient pairs
of the form `x => c` where `x` is the term and `c` the coefficient,
or out of the pairs provided by the iterator `itr`. It must be possible
to convert all terms and coefficients to the chosen term type and
coefficient type, respectively.

Neither the term type nor the coefficient type need to be concrete.
(Of course, concrete types lead to better performance.)
If the coefficient type and possibly also the term type are not
specified, the constructor tries to determine them using
`promote_type` (for coefficients) and `promote_typejoin` (for terms).

If two or more term-coefficient pairs are given with the same term,
then the corresponding coefficients are added up. This is different
from dictionaries, where any key-value pair overrides previous pairs
with the same key. However, the implemented behavior is more useful
for linear combinations.

For specialized applications, terms and coefficients can be processed
with `linear_filter` and `termcoeff` before being stored
in a linear combination. The keyword argument `is_filtered` controls
whether `linear_filter` is called for each term.

By default, linear combinations are displayed in a human-readable form
with a limited number of terms. Dedicated `show` methods display all
terms of a linear combination or return an expression that Julia can
parse as input. See the examples below.

See also [`Linear`](@ref), [`DenseLinear`](@ref), [`Linear1`](@ref),
[`linear_filter`](@ref), [`$(@__MODULE__).termcoeff`](@ref),
`Base.show`

# Examples
```jldoctest abstractlinear
julia> Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> Linear(x => c for (c, x) in enumerate('u':'z'))
Linear{Char, Int64} with 6 terms:
3*'w'+2*'v'+4*'x'+'u'+5*'y'+6*'z'

julia> Linear{Char,Int}('x' => 1, 'y' => Int8(0), 'x' => 3.0)
Linear{Char, Int64} with 1 term:
4*'x'

julia> a = Linear('x' => BigInt(1), "yz" => 2.0)
Linear{Any, BigFloat} with 2 terms:
'x'+2.0*"yz"
```
Iterating over a linear combination yields all non-zero term-coefficient
pairs. Hence a linear combination can itself be used an argument to an
`AbstractLinear` constructor.
```jldoctest abstractlinear
julia> Linear{Union{Char,String}}(a)   # same a as before
Linear{Union{Char, String}, BigFloat} with 2 terms:
'x'+2.0*"yz"
```
Various forms to display a linear combination.
```jldoctest
julia> a = Linear(x => 1 for x in 'a':'z')
Linear{Char, Int64} with 26 terms:
'n'+'f'+'w'+'d'+'e'+'o'+'h'+'j'+'i'+'k'+'r'+'s'+'t'+'q'+'y'+'a'+'c'+'p'+'m'+'z'±⋯

julia> show(stdout, MIME"text/plain"(), a)  # all terms
Linear{Char, Int64} with 26 terms:
'n'+'f'+'w'+'d'+'e'+'o'+'h'+'j'+'i'+'k'+'r'+'s'+'t'+'q'+'y'+'a'+'c'+'p'+'m'+'z'+'g'+'v'+'l'+'u'+'x'+'b'

julia> b = Linear('a' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'a'+2*'y'

julia> show(b)  # can be parsed as input
Linear{Char, Int64}('a' => 1, 'y' => 2)
```
"""
abstract type AbstractLinear{T,R} end

function (::Type{L})(itr;
        is_filtered = itr isa AbstractLinear && termtype(itr) <: T,
        kw...) where {T,R,L<:AbstractLinear{T,R}}
    a = zero(L; kw...)
    Base.haslength(itr) && sizehint!(a, length(itr))
    for (x, c) in hashed_iter(itr)
        addmul!(a, x, c; is_filtered)
    end
    a
end

# mandatory AbstractLinear interface

"""
    zero(::Type{L}; kw...) where L <: AbstractLinear -> L
    zero(a::L) where L <: AbstractLinear -> L

Return a zero linear combination of type `L`. If `zero` is called with a type `L <: AbstractLinear`
as argument, then keyword arguments may be accepted or required.

See also [`zero!`](@ref).
"""
zero(a::Type{L}) where L <: AbstractLinear = error_missing(L)

"""
    $(@__MODULE__).getcoeff(a::AbstractLinear{T,R}, x) where {T,R} -> R

Return the coefficient of `x` in the linear combination `a`.
This is `zero(R)` if `x` does not appear in `a`.

This function is part of the `AbstractLinear` interface. When it is called,
the term `x` has already been transformed via `termcoeff`, and `linear_filter(x)` is `true`.

See also [`$(@__MODULE__).setcoeff!`](@ref), [`linear_filter`](@ref), [`$(@__MODULE__).termcoeff`](@ref).
"""
getcoeff(a::AbstractLinear, x) = error_missing(typeof(a))

"""
    $(@__MODULE__).setcoeff!(a::AbstractLinear{T,R}, c, x) where {T,R} -> c

Set the coefficient of `x` in the linear combination `a` equal to `c` and return `c`.

This function is part of the `AbstractLinear` interface. When it is called,
both `x` and `c` have already been transformed via `termcoeff`, and `linear_filter(x)` is `true`.

See also [`$(@__MODULE__).getcoeff`](@ref), [`linear_filter`](@ref), [`$(@__MODULE__).termcoeff`](@ref).
"""
setcoeff!(a::AbstractLinear, c, x) = error_missing(typeof(a))

"""
    length(a::AbstractLinear) -> Int

Return the number of non-zero terms in `a`.
"""
length(a::AbstractLinear) = error_missing(typeof(a))

"""
    iterate(a::AbstractLinear [, state])

Iterating over a linear combination yields all non-zero term-coefficient pairs.

# Examples
```jldoctest
julia> a = Linear('x' => 1, 'y' => 2, 'z' => 0)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> collect(a)
2-element Vector{Pair{Char, Int64}}:
 'x' => 1
 'y' => 2

julia> Linear(x => c^2 for (x, c) in a)
Linear{Char, Int64} with 2 terms:
'x'+4*'y'
```
"""
@propagate_inbounds function iterate(a::AbstractLinear{T,R}, state...) where {T,R}
    # NOTE: we use @inbounds although the user may provide an ivalid state
    @inbounds xcs = iterate(hashed_iter(a), state...)
    if xcs === nothing
        nothing
    else
        (x, c), s = xcs
        (Pair{T,R}(unhash(x), c), s)
        # "Pair{T,R}" is important for performance
    end
end

# additional methods for AbstractLinear with default implementations

linear_type(::Type{L}, ::Type{T}, ::Type{R}) where {L<:AbstractLinear,T,R} = L{T,R}
linear_type(::Type{L}, ::Type, ::Type{R}) where {T,L<:AbstractLinear{T},R} = L{R}

function (::Type{L})(itr; kw...) where L <: AbstractLinear
    TR = element_type(itr)
    TR <: Pair || error("elements of the given iterator must be of type Pair")
    T = unhash(element_type(x for (x, c) in itr))
    R = element_type(c for (x, c) in itr)
    LTR = linear_type(L, T, R)
    LTR(itr; kw...)
end

(::Type{L})(xc::Pair...; kw...) where {T,R,L<:AbstractLinear{T,R}} = L(xc; kw...)

function (::Type{L})(xc::Pair...; kw...) where {T,L<:AbstractLinear{T}}
    isempty(xc) && error("specify coefficient type or give at least one term-coefficient pair")
    R = promote_type(map(p -> typeof(p.second), xc)...)
    LTR = linear_type(L, Any, R)
    LTR(xc; kw...)
end

function (::Type{L})(xc::Pair...; kw...) where L<:AbstractLinear
    isempty(xc) && error("specify term and coefficient types or give at least one term-coefficient pair")
    T = promote_typejoin(map(p -> typeof(p.first), xc)...)
    R = promote_type(map(p -> typeof(p.second), xc)...)
    LTR = linear_type(L, T, R)
    LTR(xc; kw...)
end

eltype(::Type{<:AbstractLinear{T,R}}) where {T,R} = Pair{T,R}
eltype(a::L) where L <: AbstractLinear = eltype(L)

"""
    termtype(::Type{L}) where L <: AbstractLinear{T,R} = T
    termtype(a::L) where L <: AbstractLinear{T,R} = T

Return the type of the terms (basis elements) in a linear combination.

See also [`coefftype`](@ref).
"""
termtype(::Type{<:AbstractLinear{T,R}}) where {T,R} = T
termtype(::L) where L <: AbstractLinear = termtype(L)

function _termtype(::Type{T}) where T
    T isa Union ? Union{_termtype(T.a), _termtype(T.b)} : T
end

_termtype(::Type{<:AbstractLinear{T,R}}) where {T,R} = T
_termtype(::T) where T = _termtype(T)

"""
    const $(@__MODULE__).DefaultCoefftype = Int

The coefficient type use by `$(@__MODULE__)` if no other coefficient type information is available.
"""
const DefaultCoefftype = Int

"""
    coefftype(::Type{L}) where L <: AbstractLinear{T,R} = R
    coefftype(a::L) where L <: AbstractLinear{T,R} = R

Return the type of the coefficients in a linear combination.

See also [`termtype`](@ref).
"""
coefftype(::Type{<:AbstractLinear{T,R}}) where {T,R} = R
coefftype(::L) where L <: AbstractLinear = coefftype(L)

function _coefftype(::Type{T}) where T
    if T isa Union
        promote_type(_coefftype(T.a), _coefftype(T.b))
    else
        Sign
    end
end

_coefftype(::Type{<:AbstractLinear{T,R}}) where {T,R} = R
_coefftype(::T) where T = _coefftype(T)
_coefftype(::Missing) = missing

function ==(a::AbstractLinear, b::AbstractLinear)
    length(a) == length(b) && all(hashed_iter(a)) do (x, c)
        getcoeff(a, unhash(x)) == c
    end
end

function Base.hash(a::AbstractLinear, h0::UInt)
    h = StructEqualHash.typehash(AbstractLinear, h0)
    for (x, c) in hashed_iter(a)
        h ⊻= hash(c, hash(x))
    end
    h
end

zero(::T) where T <: AbstractLinear = zero(T)

iszero(a::AbstractLinear) = iszero(length(a))

"""
    zero!(a::AbstractLinear) -> a

Set `a` equal to the zero linear combination and return `a`.

See also [`zero`](@ref).
"""
zero!(a::AbstractLinear) = mul!(a, 0)

"""
    coeffs(a::AbstractLinear)

Return an iterator over the non-zero coefficients appearing in `a`.
"""
coeffs(a::AbstractLinear) = (c for (x, c) in a)

"""
    terms(a::AbstractLinear)

Return an iterator over the terms appearing in `a` (with a non-zero coefficient).
"""
terms(a::AbstractLinear) = (x for (x, c) in a)

"""
    x in a::AbstractLinear -> Bool

Return `true` if the term `x` appears in the linear combination `a`
with a non-zero coefficient, and `false` otherwise.
"""
in(x, a::AbstractLinear) = !iszero(a[unhash(x)])

"""
    mul!(a::AbstractLinear, c) -> a

Multiply `a` by the scalar `c`. This functions modifies `a`.

See also [`addmul!`](@ref).
"""
function mul!(a::AbstractLinear, c)
    for (x, d) in a
        setcoeff!(a, c*d, x)
    end
    a
end

"""
    $(@__MODULE__).termcoeff(xc::Pair) -> Pair

Transform a term-coefficient pair into the format that is stored in a linear combination.

This function is called for every term-coefficient pair that enters a linear combination.
The default is to return a pair `x => c` unchanged. This can be changed in special situations
where one wants to normalize terms and coefficients. Note that `termcoeff` is called
only if `linear_filter(x)` is `true`.

See also [`linear_filter`](@ref).

# Example

We show how to convert term-coefficient pairs with an uppercase letter to lowercase letters
together with the negative coefficient.
```jldoctest
julia> function $(@__MODULE__).termcoeff((x, c)::Pair{Char})
           isuppercase(x) ? lowercase(x) => -c : x => c
       end

julia> a = Linear('x' => 1, 'Y' => 2)
Linear{Char, Int64} with 2 terms:
'x'-2*'y'

julia> a['y'], a['Y']
(-2, 2)

julia> a['Y'] = 3; a
Linear{Char, Int64} with 2 terms:
'x'-3*'y'

julia> a + 'X'
Linear{Char, Int64} with 1 term:
-3*'y'
```
"""
termcoeff(xc::Pair) = xc

const LINEAR_SHOW_MAX = 10

show_extra_params(io::IO, a::AbstractLinear) = nothing

function show(io::IO, a::L) where L <: AbstractLinear
    print(io, typename(L))
    if get(io, :typeinfo, Any) != L || iszero(a)
        print(io, '{', termtype(L), ", ", coefftype(L), '}')
    end
    print(io, '(')
    if get(io, :limit, false) && length(a) > LINEAR_SHOW_MAX
        join(io, Iterators.take(a, LINEAR_SHOW_MAX), ", ")
        print(io, " …")
    else
        join(io, a, ", ")
    end
    show_extra_params(io, a)
    print(io, ')')
end

show_term(io::IO, x) = show(io, MIME"text/plain"(), x)
show_term(io::IO, x::Union{AbstractArray,AbstractDict}) = show(io, x)

show_needsplus(c) = true
show_needsplus(c::Real) = !signbit(c)
show_needsplus(a::AbstractLinear) = length(a) != 1 || show_needsplus(first(coeffs(a)))

show_needsparen(c) = false
show_needsparen(c::Complex) = true
show_needsparen(c::Complex{Bool}) = false
show_needsparen(a::AbstractLinear) = length(a) > 1

function show_coeff(io::IO, c)
    show_needsparen(c) && print(io, '(')
    show(io, MIME"text/plain"(), c)
    show_needsparen(c) && print(io, ')')
end

function show_summand(io::IO, x, c)
    show_coeff(io, c)
    print(io, '*')
    show_term(io, x)
end

function show_summands(io, a)
    io = IOContext(io, :compact => true)
    isfirst = true
    for (x, c) in a
        if isone(c)
            isfirst || print(io, '+')
            show_term(io, x)
        elseif c == -1
            print(io, '-')
            show_term(io, x)
        else
            isfirst || !show_needsplus(c) || print(io, '+')
            show_summand(io, x, c)
        end
        isfirst = false
    end
end

function show(io::IO, ::MIME"text/plain", a::L) where L <: AbstractLinear
    if !get(io, :compact, false)
        print(io, typename(L), '{', termtype(L), ", ", coefftype(L), "} with ", length(a), " term")
        length(a) != 1 && print(io, 's')
        println(io, ':')
    end
    if iszero(a)
        # print(io, zero(R))
        print(io, '0')
    elseif get(io, :limit, false) && length(a) > 2*LINEAR_SHOW_MAX
        show_summands(io, Iterators.take(a, 2*LINEAR_SHOW_MAX))
        print(io, "±⋯")
    else
        show_summands(io, a)
    end
end

"""
    convert(::Type{L}, a::AbstractLinear; kw...) where L <: AbstractLinear -> L
    convert(::Type{L}, x; kw...) where L <: AbstractLinear -> L

Convert the linear combination `a` or the term `x` to a linear combination of type `L`.
Keyword arguments are passed to the constructor for `L`.

# Examples
```jldoctest
julia> a = Linear{AbstractChar,Int}('x' => 2)
Linear{AbstractChar, Int64} with 1 term:
2*'x'

julia> b = convert(Linear{Char,Float64}, a)
Linear{Char, Float64} with 1 term:
2.0*'x'

julia> convert(Linear{Char,Int}, 'x') == Linear('x' => 1)
true

julia> convert(DenseLinear, a; basis = Basis('a':'z'))
DenseLinear{AbstractChar, Int64} with 1 term:
2*'x'
```
"""
convert(::Type{L}, x; kw...) where L <: AbstractLinear = L(x => one(coefftype(L)); kw...)

convert(::Type{L}, a::AbstractLinear; kw...) where L <: AbstractLinear = linear_convert(L, a; kw...)

linear_convert(::Type{L}, a::L) where L <: AbstractLinear = a
linear_convert(::Type{L}, a::AbstractLinear; kw...) where L <: AbstractLinear = L(a; kw...)

hashed_iter(a) = a
# to possibly switch to an iterator (y::Hashed, c)
# we also apply this function to iterators that are not <: AbstractLinear

"""
    sizehint!(a::AbstractLinear, n::Integer) -> a

Try to make room for in total `n` non-zero term-coefficient pairs in the linear combination `a`.

This can speed up computations. At present, `sizehint!` has an effect for elements of type `Linear`
(which internally use a dictionary) and is ignored for all other subtypes of `AbstractLinear`.

See also [`Linear`](@ref), [`$(@__MODULE__).has_sizehint`](@ref), `Base.sizehint!(d::AbstractDict)`.
"""
sizehint!(a::AbstractLinear, ::Integer) = a
# default: no sizehint!

# default filter, also used for type Hashed
"""
    $(@__MODULE__).linear_filter(x) -> Bool

Return `true` if the term `x` is to be stored in linear combinations and `false` if it is to be dropped.

The effect of this is that linear combinations don't live in the vector space (or free module)
spanned by all possible terms, but rather in the quotient by the subspace (or submodule) spanned by the terms
for which `linear_filter` returns `false`. Setting coefficients for terms that are divided out is allowed,
but has no effect.

The default return value of `linear_fiilter` is `true` for all arguments, so that nothing is divided out.

See also [`keeps_filtered`](@ref), [`$(@__MODULE__).termcoeff`](@ref).

# Example
```julia-repl
julia> $(@__MODULE__).linear_filter(x::Char) = islowercase(x)

julia> a = Linear('x' => 1, 'Y' => 2)
x

julia> a['Z'] = 3
3

julia> a
x
```
"""
linear_filter(x) = true

function getindex(a::AbstractLinear{T,R}, x) where {T,R}
    y, c = termcoeff(x => ONE)
    linear_filter(y) ? inv(c)*getcoeff(a, y) : zero(R)
end

function setindex!(a::AbstractLinear{T,R}, c, x) where {T,R}
    y, d = termcoeff(x => c)
    linear_filter(y) && setcoeff!(a, d, y)
    c
end

"""
    $(@__MODULE__).modifycoeff!(op, a::AbstractLinear, x, c) -> a

Replace the coefficient of `x` in `a` by `op(getcoeff(a, x), c)` and return `a`.
Here `op` is either `+` or `-`.

This function is called after `termcoeff` and `linear_filter`.

See also [`linear_filter`](@ref), [`$(@__MODULE__).termcoeff`](@ref), [`$(@__MODULE__).modifylinear!`](@ref).
"""
function modifycoeff!(op::AddSub, a::AbstractLinear, x, c)
    x = unhash(x)
    setcoeff!(a, op(getcoeff(a, x), c), x)
end

"""
    $(@__MODULE__).modifylinear!(op, a::AbstractLinear, b::AbstractLinear, c = missing) -> a

If `op` is `+`, add `c*b` to `a`, or just `b` if `c` is missing.
If `op` is `-`, subtract `b` or `c*b` from `a`.
Store the new value in `a` and return it.

See also [`$(@__MODULE__).modifycoeff!`](@ref).
"""
function modifylinear!(op::F, a::AbstractLinear, b::AbstractLinear, c = missing) where F <: AddSub
# TODO: is this dangerous if a === b?
    sizehint!(a, length(a) + length(b))
    @inbounds for (x, d) in hashed_iter(b)
        modifycoeff!(op, a, x, c === missing ? d : c*d)
    end
    a
end

"""
    addmul!(a::AbstractLinear, b::AbstractLinear, c) -> a
    addmul!(a::AbstractLinear, x, c) -> a

Add the `c`-fold multiple of the linear combination `b` or of the term `x` to `a`,
where `c` is a scalar. This function modifies `a`.

See also [`addmul`](@ref), [`add!`](@ref), [`sub!`](@ref), [`mul!`](@ref).
"""
@inline function addmul!(a::AbstractLinear, x, c = 1; is_filtered::Bool = false)
    if !is_filtered
        linear_filter(x) || return a
        x, c = termcoeff(x => c)
    end
    modifycoeff!(+, a, x, c)
end

add!(a::AbstractLinear, x) = addmul!(a, x, 1)
add!(a::AbstractLinear{T}, c::Number) where T = addmul!(a, one(T), c)

sub!(a::AbstractLinear, x) = addmul!(a, x, -1)
sub!(a::AbstractLinear{T}, c::Number) where T = addmul!(a, one(T), -c)

function addmul!(a::AbstractLinear, b::AbstractLinear, c; is_filtered::Bool = false)
# we support "is_filtered" to be compatible with the method for terms
    iszero(c) || modifylinear!(+, a, b, c)
    a
end

function addmul!(a::AbstractLinear, b::AbstractLinear, c::Sign = ONE; is_filtered::Bool = false)
# c is optional to be compatible with addmul! for broadcasting
    isone(c) ? add!(a, b) : sub!(a, b)
end

"""
    addmul(a::AbstractLinear, b::AbstractLinear, c)
    addmul(a::AbstractLinear, x, c)

Add the `c`-fold multiple of the linear combination `b` or of the term `x` to `a`,
where `c` is a scalar.

See also [`addmul!`](@ref).
"""
addmul(a::AbstractLinear, b, c) = addmul!(copy(a), b, c)

"""
    add!(a::AbstractLinear, b::AbstractLinear) -> a
    add!(a::AbstractLinear, x) -> a

Add the linear combination `b` or the term `x` to `a`. This function modifies `a`.

See also [`addmul!`](@ref), [`sub!`](@ref).
"""
add!(a::AbstractLinear, b::AbstractLinear) = modifylinear!(+, a, b)

"""
    sub!(a::AbstractLinear, b::AbstractLinear) -> a
    sub!(a::AbstractLinear, x) -> a

Subtract the linear combination `b` or the term `x` from `a`. This function modifies `a`.

See also [`addmul!`](@ref), [`add!`](@ref).
"""
sub!(a::AbstractLinear, b::AbstractLinear) = modifylinear!(-, a, b)

"""
    copyto!(a::AbstractLinear, b::AbstractLinear, c = 1) -> a
    copyto!(a::AbstractLinear, x, c = 1) -> a

Set `a` equal to the `c`-fold multiple of the linear combination `b`
or of the term `x`. If the scalar `c` is omitted, it is taken to be `1`.
"""
function copyto!(a::AbstractLinear, b, c = ONE)
# b can be of type AbstractLinear or some term
    a === b ? mul!(a, c) : addmul!(zero!(a), b, c)
end

"""
    copy(a::L) where L <: AbstractLinear -> L

Return a copy of `a`.
"""
copy(a::L) where L <: AbstractLinear = L(xc for xc in a)

function +(as::AbstractLinear...)
    T = promote_typejoin(map(termtype, as)...)
    R = promote_type(map(coefftype, as)...)
    i = findfirst(a -> a isa Linear{T,R}, as)
    if i === nothing
        b = zero(Linear{T,R})
        foldl(add!, as; init = b)
    else
        b = copy(as[i])
        for (j, a) in enumerate(as)
            j != i && add!(b, a)
        end
    end
    b
end

+(a::AbstractLinear, x) = add!(copy(a), x)
+(x, a::AbstractLinear) = a + x
+(a::AbstractLinear, ::Zero) = a
+(::Zero, a::AbstractLinear) = a
# +(a::AbstractLinear) = copy(a)
+(a::AbstractLinear) = a

function -(a::AbstractLinear{T,R}) where {T,R}
    has_char2(R) ? a : mul!(copy(a), MINUSONE)
end

function -(a::AbstractLinear, x)
    if a isa Linear
        sub!(copy(a), x)
    else
        sub!(convert(Linear, a), x)
    end
end

function -(x, a::AbstractLinear)
    b = -a
    # b === a may be possible in characteristic 2
    add!(b === a ? copy(a) : b, x)
end

function -(a::AbstractLinear{T,R}, b::AbstractLinear{U,S}) where {T,R,U,S}
    TU = promote_typejoin(T, U)
    RS = promote_type(R, S)
    if a isa Linear && TU == U && RS == R
        sub!(copy(a), b)
    else
        sub!(convert(Linear{TU,RS}, a), b)
    end
end

function *(c::S, a::L) where {S,T,R,L<:AbstractLinear{T,R}}
    RS = promote_type(R, S)
    if RS == R
        mul!(copy(a), c)
    else
        change_coefftype(L, RS)(x => c*d for (x, d) in hashed_iter(a))
    end
end

*(a::AbstractLinear, c) = c*a

-(a::AbstractLinear, ::Zero) = a
-(::Zero, a::AbstractLinear) = -a

*(s::Sign, x::AbstractLinear) = isone(s) ? x : -x
*(x::AbstractLinear, s::Sign) = s*x

"""
    deg(a::AbstractLinear)

Return `deg(x)` where `x` is the first term appearing in `a` (as determined by `first(a)`).

The linear combination `a` must not be zero.
If `a` is homogeneous, then `deg(a)` is the common degree of all terms in it.
"""
function deg(a::AbstractLinear)
    if iszero(a)
        error("degree is only defined for non-zero linear combinations")
    else
        deg(first(a).first)
    end
end
