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

macro Function(T)
# turns a type (constructor) into a function
    :( (x...; kw...) -> $(esc(T))(x...; kw...) )
end

# function evaluation
Eval(f, x...; kw...) = f(x...; kw...)

# unwrapping Val
unval(x) = x
unval(::Val{x}) where x = x

struct ComposedFunctionOuterKw{O,I}
    outer::O
    inner::I
end

(f::ComposedFunctionOuterKw)(x...; kw...) = f.outer(f.inner(x...); kw...)

promote_typejoin(T...) = foldl(Base.promote_typejoin, T)

#
# Sign and Zero
#

# export Sign, Zero

# Sign

struct Sign
    neg::Bool
end

const ONE = Sign(false)
const MINUSONE = Sign(true)

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

sign_type(::Type{T}) where T <: Integer = Sign

# Zero

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

sign_type(::Type{Zero}) = Sign

#
# macro-like functions
#

export is_domain, has_char2

is_domain(::Type) = false
is_domain(::Type{<:Union{Real,Complex}}) = true

has_char2(::Type) = false

signed(k, x::R) where R = has_char2(R) || iseven(k) ? x : -x

sum0(itr) = sum(itr; init = Zero())
sum0(f, itr) = sum(f, itr; init = Zero())

#
# Hashed datatype
#

struct Hashed{T}
    var::T
    hash::UInt
end

Hashed{T}(x) where T = Hashed{T}(x, hash(x))
Hashed{T}(x::Hashed{T}) where T = x
Hashed{T}(x::Hashed) where T = Hashed{T}(x.var, x.hash)

convert(::Type{Hashed{T}}, x::Hashed) where T = Hashed{T}(x)

==(x::Hashed, y::Hashed) = x.var == y.var

hash(x::Hashed) = x.hash
hash(x::Hashed, h::UInt) = hash(x.var, h)

unhash(x) = x
unhash(x::Hashed) = x.var

unhash_type(::Type{T}) where T = T
unhash_type(::Type{Hashed{T}}) where T = T

show(io::IO, x::Hashed) = print(io, x.var)

#
# degree
#

export deg

deg(_) = Zero()

deg(f::ComposedFunction) = deg(f.outer) + deg(f.inner)
# TODO: can one do this more efficiently?
