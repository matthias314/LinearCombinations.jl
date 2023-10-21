#
# Linear datatype
#

export linear_filter, Linear, coefftype, coeffs, termtype, terms,
    addmul!, add!, sub!, mul!, addmul, zero!,
    keeps_filtered

# default filter, also used for type Hashed
linear_filter(x) = true

struct Linear{T,R} # <: AbstractDict{T,R}
  ht::Dict{Hashed{T},R}
  # this inner constructor prevents a conflict with the outer constructor Linear(itr) below
  Linear{T,R}(ht::Dict{Hashed{T},R}) where {T,R} = new{T,R}(ht)
end

# ==(a::Linear, b::Linear) = a.ht == b.ht
# hash(a::Linear, h::UInt) = hash(a.ht, h)

@struct_equal_hash Linear

termcoeff(xc::Pair) = xc

function Linear{T,R}(itr) where {T,R}
# note that itr may be a Dict{Hashed{U},S}
    addto = zero(Linear{T,R})
    Base.haslength(itr) && sizehint!(addto, length(itr))
    for (x, c) in itr
        addcoeff!(addto, x, c)
    end
    addto
end

pairtype_linear(::Type{Pair{T,R}}) where {T,R} = Linear{T,R}
pairtype_linear(::Type) = error("Cannot determine element type of the given iterator. Use 'Linear{T,R}(itr)' instead.")

function Linear(itr)
    L = pairtype_linear(element_type(itr))
    L(itr)
end

Linear{T,R}(xc::Pair...) where {T,R} = Linear{T,R}(xc)

function Linear(xc::Pair...)
    isempty(xc) && error("Use 'Linear{T,R}()' or give at least one term-coefficient pair")
    T = promote_typejoin(map(p -> typeof(p.first), xc)...)
    R = promote_type(map(p -> typeof(p.second), xc)...)
    Linear{T,R}(xc)
end

zero(::Type{Linear{T,R}}) where {T,R} = Linear{T,R}(Dict{Hashed{T},R}())

zero(::T) where T <: Linear = zero(T)

length(a::Linear) = length(a.ht)

iszero(a::Linear) = isempty(a)

copy(a::Linear{T,R}) where {T,R} = Linear{T,R}(copy(a.ht))

function sizehint!(a::Linear, n)
    sizehint!(a.ht, n)
    a
end

function empty!(a::Linear)
    empty!(a.ht)
    a
end

zero!(a::Linear) = empty!(a)

isempty(a::Linear) = isempty(a.ht)

function deg(a::Linear)
    if isempty(a)
        error("degree is only defined for non-zero linear combinations")
    else
        deg(first(a).first)
    end
end

coefftype(::Type{Linear{T,R}}) where {T,R} = R
coefftype(::L) where L <: Linear = coefftype(L)

_coefftype(::Type) = Sign
_coefftype(::Type{Linear{T,R}}) where {T,R} = R
_coefftype(::T) where T = _coefftype(T)

change_coefftype(::Type{Linear{T,R}}, ::Type{S}) where {T,R,S} = Linear{T,S}

promote_coefftype(::Type{Linear{T,R}}, ::Type{S}) where {T,R,S} = Linear{T, promote_type(R,S)}

termtype(::Type{Linear{T,R}}) where {T,R} = T
termtype(::L) where L <: Linear = termtype(L)

_termtype(::Type{T}) where T = T
_termtype(::Type{Linear{T,R}}) where {T,R} = T
_termtype(::T) where T = _termtype(T)

# already defined for AbstractDict (not the supertype of Linear)
eltype(::Type{Linear{T,R}}) where {T,R} = Pair{T,R}
eltype(a::L) where L <: Linear = eltype(L)

# values already defined for AbstractDict (not the supertype of Linear)
coeffs(a::Linear) = values(a.ht)

terms(a::Linear) = Iterators.map(unhash, keys(a.ht))

in(x, a::Linear{T}) where T = Hashed{T}(x) in keys(a.ht)

@propagate_inbounds function iterate(a::Linear, state...)
    # NOTE: we use @inbounds although the user may provide an ivalid state
    @inbounds xcs = iterate(a.ht, state...)
    if xcs === nothing
        xcs
    else
        (x, c), s = xcs
        (Pair(unhash(x), c), s)
    end
end

repr_coeff(c) = repr(c)
repr_coeff(a::Linear) = length(a) == 1 ? repr(a) : string('(', repr(a), ')')

show_summand(io::IO, x, cs) = print(io, cs, '*', x)

function show(io::IO, a::Linear{T,R}) where {T,R}
    if iszero(a)
        # print(io, zero(R))
        print(io, '0')
    else
        isfirst = true
        for (x, c) in a
            if isone(c)
                if isfirst
                    print(io, x)
                else
                    print(io, '+', x)
                end
            elseif isone(-c)
                print(io, '-', x)
            else
                cs = repr_coeff(c)
                isfirst || first(cs) in "+-±" || print(io, '+')
                show_summand(io, x, cs)
            end
            isfirst = false
        end
    end
end

show(io::IO, ::MIME"text/plain", a::Linear) = show(io, a)

convert(::Type{L}, a::L) where L <: Linear = a

convert(::Type{Linear{T,R}}, x::T) where {T,R} = Linear(x => one(R))

linear_convert(::Type{L}, a) where L <: Linear = L(a.ht)
# we call linear_filter with Hashed argument, which is a no-op
# Linear{T,S}(x => convert(S, c) for (x,c) in a.ht)

convert(::Type{Linear{U,S}}, a::Linear{T,R}) where {T,R,U,S} = linear_convert(Linear{U,S}, a)

getindex(a::Linear{T,R}, x::T) where {T,R} = get(a.ht, Hashed{T}(x), zero(R))

function setindex!(a::Linear{T}, c, x::T) where T
# TODO: why x::T ?
    if !iszero(c) && linear_filter(x)
        a.ht[Hashed{T}(x)] = c
    end
    c
end

function modifycoeff!(op, a::Linear{T,R}, x::Hashed{U}, c) where {T,R,U<:T}
# function modifycoeff!(op, a::Linear{T,R}, @nospecialize(x::Hashed{U}), c) where {T,R,U<:T}
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

addcoeff!(a::Linear{T}, x::Hashed{U}, c) where {T,U<:T} = modifycoeff!(+, a, x, c)
# addcoeff!(a::Linear{T}, @nospecialize(x::Hashed{U}), c) where {T,U<:T} = modifycoeff!(+, a, x, c)
# subcoeff!(a::Linear{T}, x::Hashed{U}, c) where {T,U<:T} = modifycoeff!(-, a, x, c)

@inline function addcoeff!(a::Linear{T,R}, x, c; is_filtered::Bool = false) where {T,R}
# @inline function addcoeff!(a::Linear{T}, @nospecialize(x), c; is_filtered::Bool = false) where T
    if !is_filtered
        linear_filter(x) || return a
        x, c = termcoeff(x => c)
    end
    addcoeff!(a, Hashed{T}(x), c)
end

addmul!(a::Linear, x, c; kw...) = addcoeff!(a, x, c; kw...)

add!(a::Linear, x) = addmul!(a, x, 1)
add!(a::Linear{T}, c::Number) where T = addmul!(a, one(T), c)

sub!(a::Linear, x) = addmul!(a, x, -1)
sub!(a::Linear{T}, c::Number) where T = addmul!(a, one(T), -c)

# TODO: is this dangerous if a === b?
function modifylinear!(op::F, a::Linear{T}, b::Linear{T}, c = nothing) where {F,T}
    sizehint!(a, length(a) + length(b))
    @inbounds for (x, d) in b.ht
        # we don't need to use is_filtered because x is Hashed
        modifycoeff!(op, a, x, c === nothing ? d : c*d)
    end
    a
end

function addmul!(a::Linear{T}, b::Linear{T}, c; is_filtered::Bool = false) where T
    iszero(c) || modifylinear!(+, a, b, c)
    a
end

addmul(a::Linear, b, c) = addmul!(copy(a), b, c)

add!(a::Linear, b::Linear) = modifylinear!(+, a, b)
sub!(a::Linear, b::Linear) = modifylinear!(-, a, b)

function +(a::Linear{T,R}, b::Linear{T,S}) where {T,R,S}
    RS = promote_type(R, S)
    if RS == R
        add!(copy(a), b)
    elseif RS == S
        add!(copy(b), a)
    else
        add!(convert(Linear{T,RS}, a), b)
    end
end

# +(a::Linear{T,R}, x::T) where {T,R} = addmul(a, x, 1)
+(a::Linear, x) = add!(copy(a), x)
+(x, a::Linear) = a + x
+(a::Linear, ::Zero) = a
+(::Zero, a::Linear) = a
# +(a::Linear) = copy(a)
+(a::Linear) = a

function -(a::Linear{T,R}, b::Linear{T,S}) where {T,R,S}
    RS = promote_type(R, S)
    if RS == R
        sub!(copy(a), b)
    else
        sub!(convert(Linear{T,RS}, a), b)
    end
end

# -(a::Linear{T,R}, x::T) where {T,R} = addmul(a, x, -1)
-(a::Linear, x) = sub!(copy(a), x)
# -(x::T, a::Linear{T,R}) where {T,R} = addcoeff!(-a, x, 1)
function -(x, a::Linear)
    b = -a   # b === a may be possible in characteristic 2
    add!(b === a ? copy(a) : b, x)
end
-(a::Linear, ::Zero) = a
-(::Zero, a::Linear) = -a

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
    if iszero(convert(R, c))
        empty!(a.ht)
    else
        map!(x -> c*x, values(a.ht))
	is_domain(R) || filter!(xc -> !iszero(xc[2]), a.ht)
    end
    a
end

# *(c::R, a::Linear{T,R}) where {T,R} = mul!(copy(a), c)

function *(c::S, a::Linear{T,R}) where {S,T,R}
    RS = promote_type(R, S)
    if RS == R
        mul!(copy(a), c)
    else
        Linear{T,RS}(x => c*d for (x, d) in a.ht)
    end
end

*(a::Linear, c) = c*a

*(s::Sign, x::Linear) = isone(s) ? x : -x
*(x::Linear, s::Sign) = s*x
