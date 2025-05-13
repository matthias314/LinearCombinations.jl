```@meta
DocTestSetup = doctestsetup
```

# Linear combinations

## Types

```@docs
AbstractLinear
Linear
DenseLinear
Linear1
```

## Basic methods

```@docs
termtype
coefftype
# getindex(::AbstractLinear, ::Any)
# setindex!(::AbstractLinear, ::Any, ::Any)
in(::Any, ::AbstractLinear)
length(::AbstractLinear)
iterate(::AbstractLinear)
coeffs
terms
zero
zero!
copy
copyto!
sizehint!
convert
```

## Arithmetic

```@docs
add!
sub!
mul!
addmul!
addmul
deg(::AbstractLinear)
```

## Calling linear combinations

Calling objects is extended linearly. Here is an example:
```jldoctest
julia> struct P{T} y::T end

julia> Base.hash(p::P, h::UInt) = hash(p.y, hash(P, h));  # `Linear` uses hashing

julia> @linear p::P; (p::P)(x) = x * p.y

julia> p, q = P('p'), P('q')
(P{Char}('p'), P{Char}('q'))

julia> p('x')
"xp"

julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> p(a)
Linear{String, Int64} with 2 terms:
2*"yp"+"xp"

julia> u = Linear(p => -1, q => 3)
Linear{P{Char}, Int64} with 2 terms:
-P{Char}('p')+3*P{Char}('q')

julia> u('x')
Linear{String, Int64} with 2 terms:
3*"xq"-"xp"

julia> u(a)
Linear{String, Int64} with 4 terms:
3*"xq"-2*"yp"+6*"yq"-"xp"
```

## Broadcasting

Broadcasting is supported for `AbstractLinear` types. Broadcasted versions of `+`, `-`, `*`, `=`
are converted to [`addmul!`](@ref), [`mul!`](@ref) and [`copyto!`](@ref) as much as possible to avoid
(or at least minimize) allocations. For example, for linear combinations `a`, `b`, `c` and `d`,
the statement
```julia
   a .= b .+ 2 .* (c .- 3 .* d)
```
is translated to
```julia
    copyto!(a, b)
    addmul!(a, c, 2)
    addmul!(a, d, 2*(-3))
```
and the statement
```julia
    a .+= b .+ 2 .* (c .- 3 .* d)
```
to
```julia
    addmul!(a, b)
    addmul!(a, c, 2)
    addmul!(a, d, 2*(-3))
```
Broadcasted `.*` is always interpreted as scalar multiplication, with the scalar as the first argument.
The only exception is a statement of the form `a .*= c` (that is, `a .= a .* c`) where the scalar is
the second argument.

By default, only elements of types `AbstractLinear` and `Number` participate in broadcasting.
To allow other scalar or term types, one has to use the macro `@linear_broadcastable`.
```jldoctest
julia> @linear_broadcastable Char

julia> a, b = Linear('x' => 1), Linear('y' => 2)
(Linear{Char, Int64}('x' => 1), Linear{Char, Int64}('y' => 2))

julia> a .+= b .+ 2 .* 'z'
Linear{Char, Int64} with 3 terms:
'x'+2*'y'+2*'z'

julia> a
Linear{Char, Int64} with 3 terms:
'x'+2*'y'+2*'z'
```

```@docs
@linear_broadcastable
```

## `AbstractLinear` interface

```@docs
LinearCombinations.getcoeff
LinearCombinations.setcoeff!
LinearCombinations.modifycoeff!
LinearCombinations.modifylinear!
```
