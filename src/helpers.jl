module TestHelpers

using Base: Fix1, Fix2

using StructEqualHash
using ..LinearCombinations
using ..LinearCombinations: sum0
import LinearCombinations: deg, return_type, keeps_filtered, hastrait, linear_filter

# ShowArgs

export ShowArgs

struct ShowArgs{F}
    f::F
end

function (f::ShowArgs)(args...; kw...)
    println("$(f.f): $args $(NamedTuple(kw))")
    f.f(args...; kw...)
end

hastrait(f::ShowArgs, trait::Val, types::Type...) = hastrait(f.f, trait, types...)

keeps_filtered(f::ShowArgs, types::Type...) = keeps_filtered(f.f, types...)

deg(f::ShowArgs) = deg(f.f)

# ShowFilter

export ShowFilter

struct ShowFilter{T} x::T end

@struct_equal_hash ShowFilter

Base.convert(::Type{S}, x) where S <: ShowFilter = S(x)

function linear_filter(sf::ShowFilter)
    b = linear_filter(sf.x)
    println("linear_filter: $(sf.x) $b")
    b
end

# ErrorFilter

export ErrorFilter, FilterException

struct ErrorFilter{T} x::T end

@struct_equal_hash ErrorFilter

deg(x::ErrorFilter) = deg(x.x)

struct FilterException <: Exception end

linear_filter(::ErrorFilter) = throw(FilterException())

Base.:*(x::ErrorFilter, y) = ErrorFilter(x.x*y)
Base.:*(x, y::ErrorFilter) = ErrorFilter(x*y.x)
Base.:*(x::ErrorFilter, y::ErrorFilter) = ErrorFilter(x.x*y.x)

# KeepsFiltered

export KeepsFiltered

struct KeepsFiltered{F}
    f::F
end

struct LinearCallable{W,F}
    f::F
    LinearCallable{W}(f::F) where {W,F} = new{W,F}(f)
    LinearCallable{W}(::Type{T}) where {W,T} = new{W,Type{T}}(T)
end

(f::LinearCallable)(args...; kw...) = f.f(args...; kw...)

@multilinear f::KeepsFiltered LinearCallable{KeepsFiltered}(f.f)

hastrait(f::LinearCallable{KeepsFiltered}, trait::Val, types::Type...) = hastrait(f.f, trait, types...)

keeps_filtered(f::LinearCallable{KeepsFiltered}, ::Type...) = true

return_type(f::LinearCallable{KeepsFiltered}, types::Type...) = return_type(f.f, types...)

# BasisLinear

export BasicLinear

import LinearCombinations: zero, getcoeff, setcoeff!
import Base: length, iterate

struct BasicLinear{T,R} <: AbstractLinear{T,R}
    a::Linear{T,R}
    BasicLinear{T,R}(a::Some{Linear{T,R}}) where {T,R} = new{T,R}(something(a))
end

zero(::Type{BasicLinear{T,R}}) where {T,R} = BasicLinear{T,R}(Some(zero(Linear{T,R})))

function getcoeff(a::BasicLinear{T,R}, x) where {T,R}
    x = convert(T, x)  # to avoid accepting Hashed values
    getcoeff(a.a, x)
end

function setcoeff!(a::BasicLinear{T,R}, c, x) where {T,R}
    x = convert(T, x)  # to avoid accepting Hashed values
    setcoeff!(a.a, c, x)
end

length(a::BasicLinear) = length(a.a)

iterate(a::BasicLinear, state...) = iterate(a.a, state...)

# Graded and @gr_str

using ..LinearCombinations: _termtype, _coefftype

export Graded, GradedString, ungraded, @gr_str

struct Graded{T,D}
    x::T
    n::D
end

Graded(::Type{T}, n::D) where {T,D} = Graded{Type{T},D}(T, n)

const GradedString = Graded{String,Int}

GradedString(s::String) = Graded(s, length(s))

function Base.show(io::IO, gr::Graded)
    if gr isa GradedString && gr.n == length(gr.x)
        print(io, "gr")
        show(io, gr.x)
    else
        show(io, gr.x)
        print(io, '⟨', gr.n, '⟩')
    end
end

@struct_equal_hash Graded

@linear ::Fix2{Type{Graded}}

@linear ungraded

ungraded(gr::Graded) = gr.x
ungraded(::Type{<:Graded{T}}) where T = T
ungraded(::Type{Linear{<:Graded{T},R}}) where {T,R} = Linear{T,R}
ungraded(::Type{Linear1{<:Graded{T},R}}) where {T,R} = Linear1{T,R}
ungraded(::Type{L}) where L <: AbstractLinear = error("linear type $L not supported")

deg(gr::Graded) = gr.n

linear_filter(gr::Graded) = linear_filter(gr.x)

function Base.:*(grs::Vararg{Graded,M}) where M
    n = sum0(map(deg, grs))
    Graded(*(map(ungraded, grs)...), n)
end

struct GradedCallable{GR}
    gr::GR
end

@struct_equal_hash GradedCallable

function (grc::GradedCallable)(args::Graded...; kw...)
    n = sum(deg, args; init = deg(grc.gr))
    Fix2(Graded, n)(grc.gr.x(map(ungraded, args)...; kw...))
end

function return_type(grc::GradedCallable{Graded{GR,D}}, types::Type...) where {GR,D}
    LU = return_type(grc.gr.x, map(ungraded, types)...)
    E = promote_type(D, map(Fix1(return_type, deg), types)...)
    GRU = Graded{_termtype(LU),E}
    if LU <: Linear
        Linear{GRU, _coefftype(LU)}
    elseif LU <: Linear1
        Linear1{GRU, _coefftype(LU)}
    else
        @assert !(LU <: AbstractLinear) "linear type $LU not supported"
        GRU
    end
end

keeps_filtered(grc::GradedCallable, types::Type...) = keeps_filtered(grc.gr.x, map(ungraded, types)...)
hastrait(grc::GradedCallable, trait::Val, types::Type...) = hastrait(grc.gr.x, trait, map(ungraded, types)...)

@multilinear gr::Graded GradedCallable(gr)

macro gr_str(s) GradedString(unescape_string(s)) end

Base.firstindex(gr::GradedString) = firstindex(gr.x)
Base.lastindex(gr::GradedString) = lastindex(gr.x)
Base.length(gr::GradedString) = length(gr.x)

Base.getindex(gr::GradedString, i::Integer) = gr.x[i]
Base.getindex(gr::GradedString, ii::AbstractVector{<:Integer}) = GradedString(gr.x[ii])

Base.:^(gr::GradedString, k::Integer) = GradedString(repeat(gr.x, k))

export ErrorGraded, DegreeException

const ErrorGraded{T} = Graded{T,Missing}

ErrorGraded(x) = Graded(x, missing)

struct DegreeException end

deg(gr::ErrorGraded) = throw(DegreeException())

return_type(::typeof(deg), ::Type{<:ErrorGraded}) = Int
# needed to bypass exception

function (grc::GradedCallable{<:ErrorGraded})(args::Graded...; kw...)
    Fix2(Graded, missing)(grc.gr.x(map(ungraded, args)...; kw...))
end

function return_type(grc::GradedCallable{<:ErrorGraded}, types::Type...)
    LU = return_type(grc.gr.x, map(ungraded, types)...)
    GRU = Graded{_termtype(LU),Missing}
    if LU <: Linear
        Linear{GRU, _coefftype(LU)}
    elseif LU <: Linear1
        Linear1{GRU, _coefftype(LU)}
    else
        @assert !(LU <: AbstractLinear) "linear type $LU not supported"
        GRU
    end
end

end # module TestHelpers
