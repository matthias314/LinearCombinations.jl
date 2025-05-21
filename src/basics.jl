#
# return type and element type
#

# using ReturnType: ReturnType, has_method

# import Base: promote_op as return_type

return_type(f, types...) = Core.Compiler.return_type(f, Tuple{types...})

element_type(itr) = eltype(itr)
element_type(g::Base.Generator) = return_type(g.f, element_type(g.iter))

#
# basics
#

error_missing(::Type{T}) where T = error("implementation missing for type ", T)

macro Function(T)
# turns a type (constructor) into a function
    :( (x...; kw...) -> $(esc(T))(x...; kw...) )
end

# function evaluation
Eval(f, x...; kw...) = f(x...; kw...)

# unwrapping Val
"""
    $(@__MODULE__).unval(x)

Return `c` if the argument `x` is of type `Val{c}` and `x` itself otherwise.

This can be used to write type-stable code for the `coefftype` keyword argument
to linear and multilinear functions.

See also [`@linear`](@ref), [`@linear_kw`](@ref), [`@multilinear`](@ref).

# Examples
```jldoctest
julia> $(@__MODULE__).unval(Char)
Char

julia> $(@__MODULE__).unval(Val(Char))
Char
```
"""
unval(x) = x
unval(::Val{c}) where c = c

promote_typejoin(T...) = foldl(Base.promote_typejoin, T)

push_kw(nt::Union{NamedTuple, Base.Pairs}; kw...) = (; nt..., kw...)

#
# Add and Sub
#

const Add = typeof(+)
const Sub = typeof(-)
const Mul = typeof(*)
const AddSub = Union{Add,Sub}

#
# Sign and Zero
#

# export Sign, Zero

# Sign

struct Sign
    neg::Bool
end

const ONE = Sign(false)

one(::Type{Sign}) = ONE
one(::Sign) = one(Sign)

isone(s::Sign) = !s.neg

iszero(s::Sign) = false

+(s::Sign) = s
-(s::Sign) = Sign(!s.neg)
*(s::Sign, x) = isone(s) ? x : -x
*(x, s::Sign) = s*x
*(s::Sign, t::Sign) = Sign(xor(s.neg, t.neg))

Base.inv(s::Sign) = s

==(s::Sign, x) = isone(s*x)
==(x, s::Sign) = s == x
==(s::Sign, t::Sign) = s.neg == t.neg

hash(s::Sign, h::UInt) = hash(isone(s) ? 1 : -1, h)

show(io::IO, s::Sign) = show(io, s*1)

convert(::Type{Sign}, s::Sign) = s
convert(::Type{R}, s::Sign) where R <: Number = s*one(R)
    # added "<: Number" to reduce invalidations

function convert(::Type{Sign}, x)
    if isone(x)
        Sign(false)
    elseif isone(-x)
        Sign(true)
    else
        error("cannot convert $x to Sign")
    end
end

promote_rule(::Type{Sign}, ::Type{R}) where R = R

signtype(::Type{T}) where T <: Integer = Sign

# Zero

"""
    Zero

`Zero` is a type whose only value `Zero()` behaves like `0`, but allows
for simplification at compile time.
"""
struct Zero end

zero(::Type{Zero}) = Zero()
zero(::Zero) = zero(Zero)

iszero(::Zero) = true

iseven(::Zero) = true
isodd(::Zero) = false

+(::Zero) = Zero()
+(::Zero, x) = x
+(x, ::Zero) = x
+(::Zero, ::Zero) = Zero()

-(::Zero) = Zero()
-(::Zero, x) = -x
-(x, ::Zero) = x
-(::Zero, ::Zero) = Zero()

*(::Zero, x) = Zero()
*(x, ::Zero) = Zero()
*(::Zero, ::Zero) = Zero()

promote_rule(::Type{Zero}, ::Type{R}) where R = R

convert(::Type{R}, ::Zero) where R <: Number = zero(R)
    # added "<: Number" to reduce invalidations

signtype(::Type{Zero}) = Sign

#
# macro-like functions
#

export is_domain, has_char2

"""
    is_domain(::Type{R}) where R -> Bool

Return `true` if the ring `R` is known to be integral domain and `false` otherwise.
An integral domain is a commutative ring without zero divisors.

By default, `is_domain` returns `true` only for subtypes of `Real` and `Complex`.
If `is_domain(R) == true`, then sometimes more efficient algorithms can be chosen.

See also [`has_char2`](@ref).
"""
is_domain(::Type) = false
is_domain(::Type{<:Union{Real,Complex}}) = true

"""
    has_char2(::Type{R}) where R -> Bool
    has_char2(x::R) where R -> Bool

Return `true` if the ring `R` is known to have characteristic `2` and `false` otherwise.

By default, `has_char2` returns `false` for all arguments. Changing it to `true` for a ring `R`
avoids (possibly expensive) sign computations.

See also [`is_domain`](@ref).
"""
has_char2

has_char2(::Type) = false
has_char2(::T) where T = has_char2(T)
has_char2(::Missing) = false  # TODO: needed?

"""
    $(@__MODULE__).withsign(k, x)

Return a value representing `(-1)^k*x`. The default definition is
```julia
withsign(k, x) = has_char2(x) || iseven(k) ? x : -x
```
Additional methods may be needed to support more exotic coefficient types.
"""
withsign(k, x) = has_char2(x) || iseven(k) ? x : -x

sum0(itr) = sum(itr; init = Zero())
sum0(f, itr) = sum(f, itr; init = Zero())

sum0(::Tuple{}) = Zero()
sum0(x::Tuple) = x[1] + sum0(x[2:end])

#
# Hashed datatype
#

"""
    $(@__MODULE__).Hashed{T}

`Hashed{T}(x::T)` is a wrapper that stores `hash(x)` along with `x`. Computing the hash of such
an element returns the stored hash value.

See also [`$(@__MODULE__).unhash`](@ref).
"""
struct Hashed{T}
    var::T
    hash::UInt
end

Hashed{T}(x) where T = Hashed{T}(x, hash(x))
Hashed{T}(x::Hashed{T}) where T = x
Hashed{T}(x::Hashed) where T = Hashed{T}(x.var, x.hash)

show(io::IO, ::MIME"text/plain", x::Hashed) = show(io, MIME"text/plain"(), x.var)

convert(::Type{Hashed{T}}, x::Hashed) where T = Hashed{T}(x)

==(x::Hashed, y::Hashed) = x.var == y.var

hash(x::Hashed) = x.hash
hash(x::Hashed, h::UInt) = hash(x.var, h)

"""
    $(@__MODULE__).unhash(x)

Return the argument unless it is of the form `Hashed{T}(y)`, in which case `y` is returned.

See also [`$(@__MODULE__).Hashed`](@ref).
"""
unhash(x) = x
unhash(x::Hashed) = x.var

unhash_type(::Type{T}) where T = T
unhash_type(::Type{Hashed{T}}) where T = T

#
# degree
#

export deg

"""
    deg(x)

Return the degree of `x`. The default value of `deg(x)` is `0`.
(More precisely, it is `$(@__MODULE__).Zero()`, which behaves like `0`.)

See also [`deg(::AbstractTensor)`](@ref), [`$(@__MODULE__).Zero`](@ref).
"""
deg(_) = Zero()

deg(f::ComposedFunction) = deg(f.outer) + deg(f.inner)
# TODO: can one do this more efficiently?
