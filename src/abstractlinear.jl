#
# AbstractLinear abstract data type
#

export linear_filter, AbstractLinear,
    coefftype, coeffs, termtype, terms,
    addmul!, add!, sub!, mul!, addmul, zero!

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

linear_type(::Type{L}, ::Type{T}, ::Type{R}) where {L<:AbstractLinear,T,R} = L{T,R}
linear_type(::Type{L}, ::Type, ::Type{R}) where {T,L<:AbstractLinear{T},R} = L{R}

function (::Type{L})(itr; kw...) where L <: AbstractLinear
    TR = element_type(itr)
    TR <: Pair || error("elements of the given iterator must be of type Pair")
    T, R = TR.parameters
    T = unhash(T)
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

termtype(::Type{<:AbstractLinear{T,R}}) where {T,R} = T
termtype(::L) where L <: AbstractLinear = termtype(L)

function _termtype(::Type{T}) where T
    if T isa Union
        promote_typejoin(_termtype(T.a), _termtype(T.a))
    else
        T
    end
end

_termtype(::Type{<:AbstractLinear{T,R}}) where {T,R} = T
_termtype(::T) where T = _termtype(T)

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

function ==(a::AbstractLinear, b::AbstractLinear)
    length(a) == length(b) && all(hashed_iter(a)) do (x, c)
        b[x] == c
    end
end

function Base.hash(a::AbstractLinear, h0::UInt)
    h = StructEqualHash.typehash(AbstractLinear)
    for (x, c) in hashed_iter(a)
        h ⊻= hash(c, hash(x))
    end
    hash(h, h0)
end

zero(::T) where T <: AbstractLinear = zero(T)

function zero! end

function coeffs end
function terms end

function addmul! end
function mul! end
function addmul end

termcoeff(xc::Pair) = xc

repr_coeff(c) = repr(c)
repr_coeff(a::AbstractLinear) = length(a) == 1 ? repr(a) : string('(', repr(a), ')')

show_summand(io::IO, x, cs) = print(io, cs, '*', x)

function show(io::IO, a::AbstractLinear{T,R}) where {T,R}
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

convert(::Type{L}, x; kw...) where L <: AbstractLinear = L(x => one(coefftype(L)); kw...)

convert(::Type{L}, a::AbstractLinear; kw...) where L <: AbstractLinear = linear_convert(L, a; kw...)

linear_convert(::Type{L}, a::L) where L <: AbstractLinear = a
linear_convert(::Type{L}, a::AbstractLinear; kw...) where L <: AbstractLinear = L(a; kw...)

hashed_iter(a) = a
# to possibly switch to an iterator (y::Hashed, c)
# we also apply this function to iterators that are not <: AbstractLinear

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

# default: no sizehint!
sizehint!(a::AbstractLinear, ::Integer) = a

# default filter, also used for type Hashed
linear_filter(x) = true

modifycoeff!(op, a::AbstractLinear{T}, x, c) where T = modifycoeff!(op, a, Hashed{T}(x), c)

function modifylinear!(op::F, a::AbstractLinear, b::AbstractLinear, c = missing) where F
# TODO: is this dangerous if a === b?
    sizehint!(a, length(a) + length(b))
    @inbounds for (x, d) in hashed_iter(b)
        modifycoeff!(op, a, x, c === missing ? d : c*d)
    end
    a
end

@inline function addmul!(a::AbstractLinear, x, c; is_filtered::Bool = false)
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

addmul(a::AbstractLinear, b, c) = addmul!(copy(a), b, c)

add!(a::AbstractLinear, b::AbstractLinear) = modifylinear!(+, a, b)
sub!(a::AbstractLinear, b::AbstractLinear) = modifylinear!(-, a, b)

function copyto!(a::AbstractLinear, b, c = ONE)
# b can be of type AbstractLinear or some term
    a === b ? mul!(a, c) : addmul!(zero!(a), b, c)
end

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
        sub!(convert(Linear, a), b)
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

function deg(a::AbstractLinear)
    if iszero(a)
        error("degree is only defined for non-zero linear combinations")
    else
        deg(first(a).first)
    end
end
