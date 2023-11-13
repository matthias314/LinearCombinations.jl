#
# Linear1 datatype
#

export Linear1

mutable struct Linear1{T,R} <: AbstractLinear{T,R}
    iszero::Bool
    x::T
    c::R
    Linear1{T,R}() where {T,R} = new{T,R}(true)
end

change_coefftype(::Type{Linear1{T,R}}, ::Type{S}) where {T,R,S} = Linear1{T,S}

function Base.:(==)(a::Linear1, b::Linear1)
    a.iszero == b.iszero && (a.iszero || (a.c == b.c && a.x == b.x))
end

zero(::Type{Linear1{T,R}}) where {T,R} = Linear1{T,R}()

iszero(a::Linear1) = a.iszero

length(a::Linear1) = Int(!iszero(a))

copy(a::Linear1{T,R}) where {T,R} = iszero(a) ? zero(a) : Linear1{T,R}(a.x => a.c)

function zero!(a::Linear1)
    a.iszero = true
    a
end

coeffs(a::Linear1) = iszero(a) ? () : (a.c,)

terms(a::Linear1) = iszero(a) ? () : (a.x,)

in(x, a::Linear1) = !iszero(a) && unhash(x) == a.x

iterate(a::Linear1{T,R}) where {T,R} = iszero(a) ? nothing : (Pair{T,R}(a.x, a.c), 2)
# "Pair{T,R}" is important for performance
iterate(a::Linear1, ::Int) = nothing

getindex(a::Linear1{T,R}, x) where {T,R} = x in a ? a.c : zero(R)

function setindex!(a::Linear1{T}, c, x) where T
    if iszero(a)
        a.x = x
    elseif a.x != x
        error("Linear1 cannot store linear combinations of two or more elements")
    end
    a.iszero = iszero(c)
    a.c = c
end

function modifycoeff!(op, a::Linear1{T,R}, x, c) where {T,R}
    if iszero(c)
        return a
    elseif x isa Hashed
        x = unhash(x)
    end
    if iszero(a)
        a.iszero = false
        a.x = x
        a.c = c
    elseif a.x == x
        a.c = op(a.c, c)
        a.iszero = iszero(a.c)
    else
        error("Linear1 cannot store linear combinations of two or more elements")
    end
    a
end

function -(a::Linear1{T,R}) where {T,R}
    if has_char(2) || iszero(a)
        a
    else
        Linear1{T,R}(a.x => -a.c)
    end
end

function mul!(a::Linear1{T,R}, c) where {T,R}
    if !iszero(a)
        if iszero(c)
            a.iszero = true
        else
            a.c *= c
            a.iszero = iszero(a.c)
        end
    end
    a
end