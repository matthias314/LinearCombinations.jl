#
# Linear datatype
#

export Linear

"""
    Linear{T,R} <: AbstractLinear{T,R}

    Linear{T,R}(itr)

Construct a linear combination of type `Linear` with term type `T` and
coefficient type `R` out of the term-coefficient pairs provided by the iterator `itr`.

Linear combinations of this type are sparse in the sense
that terms and (non-zero) coefficients are internally stored in a dictionary.

Other ways to use this constructor are discussed under `AbstractLinear`.

See also [`AbstractLinear`](@ref), [`DenseLinear`](@ref), [`Linear1`](@ref).
"""
struct Linear{T,R} <: AbstractLinear{T,R}
    ht::Dict{Hashed{T},R}
    # this inner constructor prevents a conflict with the outer constructor Linear(itr) below
    Linear{T,R}(ht::Dict{Hashed{T},R}) where {T,R} = new{T,R}(ht)
end

change_coefftype(::Type{Linear{T,R}}, ::Type{S}) where {T,R,S} = Linear{T,S}

Base.:(==)(a::Linear, b::Linear) = a.ht == b.ht

zero(::Type{Linear{T,R}}) where {T,R} = Linear{T,R}(Dict{Hashed{T},R}())

iszero(a::Linear) = isempty(a.ht)

length(a::Linear) = length(a.ht)

copy(a::Linear{T,R}) where {T,R} = Linear{T,R}(copy(a.ht))

function sizehint!(a::Linear, n::Integer)
    sizehint!(a.ht, n)
    a
end

function zero!(a::Linear)
    empty!(a.ht)
    a
end

coeffs(a::Linear) = values(a.ht)

terms(a::Linear) = Iterators.map(unhash, keys(a.ht))

in(x, a::Linear{T}) where T = haskey(a.ht, Hashed{T}(x))

hashed_iter(a::Linear) = a.ht

function getindex(a::Linear, x)
    y, c = termcoeff(x => ONE)
    inv(c)*getcoeff(a, y)
end

getcoeff(a::Linear{T,R}, x) where {T,R} = get(a.ht, Hashed{T}(x), zero(R))

function setcoeff!(a::Linear{T}, c, x) where T
    if !iszero(c)
        a.ht[Hashed{T}(x)] = c
    end
    c
end

function modifycoeff!(op, a::Linear{T,R}, x::Hashed, c) where {T,R}
# function modifycoeff!(op, a::Linear{T,R}, @nospecialize(x::Hashed), c) where {T,R}
    ht = a.ht
    i = Base.ht_keyindex2!(ht, x)
    local v::R
    if i > 0
        @inbounds v = op(ht.vals[i], c)
        if iszero(v)
            # we don't keep zero entries in the dictionary
            Base._delete!(ht, i)
        else
            @inbounds ht.vals[i] = v
        end
    else
        v = op(c)
        iszero(v) || @inbounds Base._setindex!(ht, v, x, -i)
    end
    a
end

function -(a::Linear{T,R}) where {T,R}
    if has_char2(R)
        a
    else
        b = copy(a)
        map!(-, values(b.ht))
        b
    end
end

function mul!(a::Linear{T,R}, c) where {T,R}
    c1::R = c
    if iszero(c1)
        zero!(a)
    elseif !isone(c1)
        map!(x -> c*x, values(a.ht))
        is_domain(R) || filter!(xc -> !iszero(xc[2]), a.ht)
    end
    a
end
