module TestHelpers

using ..LinearCombinations

# ShowArgs

export ShowArgs

import LinearCombinations: hastrait, keeps_filtered, deg

struct ShowArgs{F}
    f::F
end

function (f::ShowArgs)(args...; kw...)
    println("$(f.f): $args $(NamedTuple(kw))")
    f.f(args...; kw...)
end

hastrait(f::ShowArgs, trait::Val, types...) = hastrait(f.f, trait, types...)

keeps_filtered(f::ShowArgs, types...) = keeps_filtered(f.f, types...)

deg(f::ShowArgs) = deg(f.f)

# BasisLinear

export BasicLinear

import LinearCombinations: zero, getcoeff, setcoeff!, length, iterate

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

end # module TestHelpers
