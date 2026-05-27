#
# AbstractTensor
#

export AbstractTensor

"""
    AbstractTensor{T<:Tuple}

The supertype of all tensor types. The most important subtype is `Tensor`.
There are also [twisted tensors](@ref sec-twisted).

See [`Tensor`](@ref), [`tensor`](@ref),
[`LeftTwistedTensor`](@ref), [`RightTwistedTensor`](@ref).
"""
abstract type AbstractTensor{T<:Tuple} end

(::Type{T})(x...) where T <: AbstractTensor = T(x)

"""
    fieldtypes(::Type{T}) where T <:AbstractTensor -> Tuple

Return the types of the components of `T` as a tuple.

# Example
```jldoctest
julia> fieldtypes(Tensor{Tuple{Char, String}})
(Char, String)
```
"""
Base.fieldtypes(::Type{<:AbstractTensor{T}}) where T <: Tuple = fieldtypes(T)

"""
    Tuple(t::AbstractTensor{T}) -> T <: Tuple

Return the tuple of components of `t`.

Although any `AbstractTensor` has to supports the iteration interface,
it is often more efficient to deal with the underlying `Tuple` of components.
For instance, functions like `map` or `reduce` map return a `Tuple` in this case
instead of a `Vector`.

# Example

```jldoctest
julia> t = Tensor('A','b','c')
'A'⊗'b'⊗'c'

julia> Tuple(t)
('A', 'b', 'c')

julia> map(isuppercase, t)
3-element Vector{Bool}:
 1
 0
 0

julia> map(isuppercase, Tuple(t))
(true, false, false)
```
"""
Base.Tuple(t::AbstractTensor) = error_missing(typeof(t))

length(t::AbstractTensor) = length(Tuple(t))

firstindex(t::AbstractTensor) = 1
lastindex(t::AbstractTensor) = length(t)

iterate(t::AbstractTensor, state...) = iterate(Tuple(t), state...)

@propagate_inbounds getindex(t::AbstractTensor, k) = Tuple(t)[k]

function show(io::IO, ::MIME"text/plain", t::T) where T <: AbstractTensor
    if isempty(t)
        print(io, "()")
    else
        get(io, :intensor, false) && print(io, '(')
        for (i, x) in enumerate(t)
            i == 1 || print(io, tensor_operator(T))
            show_term(IOContext(io, :compact => true, :intensor => true), x)
        end
        get(io, :intensor, false) && print(io, ')')
        nothing
    end
end

copy(t::AbstractTensor) = t

convert(::Type{T}, t::AbstractTensor) where T <: AbstractTensor = T(Tuple(t))

"""
    deg(t::AbstractTensor)

Return the degree of a tensor, which is the sum of the degrees of its components.

See also [`deg`](@ref).
"""
deg(t::AbstractTensor) = sum0(deg, Tuple(t))
# type inference doesn't work without "Tuple"

# factor_types(::Type{<:AbstractTensor{T}}) where T <: Tuple = fieldtypes(T)

#=
deg_return_type_tensor(R, T...) = promote_type(R, map(Fix1(return_type, deg), T)...)

# TODO: needed?
return_type(::typeof(deg), ::Type{T}) where T <: AbstractTensor =
    deg_return_type_tensor(Int, factor_types(T)...)
=#

_revsums(dt) = dt
_revsums(dt, t...) = _revsums((t[end]+dt[1], dt...), t[1:end-1]...)

revsums(::Tuple{}) = ()
revsums(t::Tuple) = _revsums((Zero(),), t[2:end]...)

linear_filter(t::AbstractTensor) = all(linear_filter, Tuple(t))

@linear_broadcastable AbstractTensor

#
# Tensor datatype
#

export Tensor, tensor, ⊗, cat, flatten

"""
    Tensor{T<:Tuple}

    Tensor{T}(xs...) where T
    Tensor(xs...)

The type `Tensor` represents pure tensors.

A general tensor is a linear combination of pure tensors and can conveniently be
created using `tensor`. `$(@__MODULE__)` takes pure tensors as basis elements.

A `Tensor` can be created out of a `Tuple` or out of the individual components.
The second form is not available if the tensor has a tuple as its only component.

`Tensor` implements the
[iteration](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration)
and
[indexing](https://docs.julialang.org/en/v1/manual/interfaces/#Indexing)
interfaces. This makes for example splatting available for tensors, and
the `i`-th component of `t::Tensor` can be accessed as `t[i]`.

Tensors can be nested. Different bracketings lead to different tensors. The functions
`cat`, `flatten`, `swap` and the type `Regroup` are provided to make rearranging tensors more easily.

Note that the type parameter of `Tensor` is always a `Tuple`. For instance, the type of
a `Tensor` with two components of types `T1` and `T2` is `Tensor{Tuple{T1,T2}}`, not
`Tensor{T1,T2}`.

See also [`tensor`](@ref), [`cat`](@ref), [`flatten`](@ref), [`swap`](@ref), [`Regroup`](@ref).

# Examples

```jldoctest
julia> t = Tensor('x', 'y', "z")
'x'⊗'y'⊗"z"

julia> typeof(t)
Tensor{Tuple{Char, Char, String}}

julia> Tuple(t)
('x', 'y', "z")

julia> length(t), t[2], t[end]
(3, 'y', "z")

julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> b = Linear(Tensor('x', 'z') => 1, Tensor('y', 'z') => 2)
Linear{Tensor{Tuple{Char, Char}}, Int64} with 2 terms:
2*'y'⊗'z'+'x'⊗'z'

julia> b == tensor(a, 'z')
true

julia> [uppercase(x) for x in t]
3-element Vector{Any}:
 'X': ASCII/Unicode U+0058 (category Lu: Letter, uppercase)
 'Y': ASCII/Unicode U+0059 (category Lu: Letter, uppercase)
 "Z"

julia> f((x1, xs...)::Tensor) = x1
f (generic function with 1 method)

julia> f(t)
'x': ASCII/Unicode U+0078 (category Ll: Letter, lowercase)

julia> t == Tensor(Tensor('x', 'y'), "z")
false

julia> a = tensor(); a[Tensor()]
1
```
"""
struct Tensor{T<:Tuple} <: AbstractTensor{T}
    a::T
end

@struct_equal_hash Tensor

Base.Tuple(t::Tensor) = t.a

tensor_operator(::Type{<:Tensor}) = "⊗"

function show(io::IO, t::Tensor{T}) where T <: Tuple
    print(io, :Tensor)
    typeof(Tuple(t)) == T || print(io, '{', T, '}')
    print(io, '(')
    if T <: Tuple{Tuple}
        print(io, Tuple(t))
    else
        for (i, x) in enumerate(t)
            i == 1 || print(io, ", ")
            show(io, x)
        end
    end
    print(io, ')')
end

keeps_filtered(::Type{<:Tensor}, ::Type{<:Tuple}) = true

"""
    x ⊗ y -> AbstractLinear{<:Tensor}
    ⊗(xs...) -> AbstractLinear{<:Tensor}
    tensor(xs...) -> AbstractLinear{<:Tensor}

`tensor` is the multilinear extension of `Tensor`. The `⊗` operator is a synomym
for `tensor`. Note that `tensor` always returns a linear combination. The return type
is `DenseLinear` if all arguments are `DenseLinear`; the corresponding basis is the
`TensorBasis` of the bases of the arguments. In all other cases the return type is
`Linear`. This can be overriden via the `addto` keyword argument.

Remember that `⊗` is not associative in Julia, unlike `*`. Writing tensors
with more than two factors in infix notation therefore produces nested tensors.

See also [`Tensor`](@ref), [`@multilinear`](@ref), [`TensorBasis`](@ref).

# Examples

```jldoctest
julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> b = Linear("w" => 3, "z" => -1)
Linear{String, Int64} with 2 terms:
3*"w"-"z"

julia> tensor(a, "w")
Linear{Tensor{Tuple{Char, String}}, Int64} with 2 terms:
'x'⊗"w"+2*'y'⊗"w"

julia> a ⊗ b
Linear{Tensor{Tuple{Char, String}}, Int64} with 4 terms:
-'x'⊗"z"-2*'y'⊗"z"+3*'x'⊗"w"+6*'y'⊗"w"

julia> tensor('x', b, a; coefftype = Float64)
Linear{Tensor{Tuple{Char, String, Char}}, Float64} with 4 terms:
6.0*'x'⊗"w"⊗'y'-2.0*'x'⊗"z"⊗'y'-'x'⊗"z"⊗'x'+3.0*'x'⊗"w"⊗'x'

julia> 'x' ⊗ b ⊗ a
Linear{Tensor{Tuple{Tensor{Tuple{Char, String}}, Char}}, Int64} with 4 terms:
-('x'⊗"z")⊗'x'+3*('x'⊗"w")⊗'x'-2*('x'⊗"z")⊗'y'+6*('x'⊗"w")⊗'y'

julia> tensor('x', b, a) == 'x' ⊗ b ⊗ a
false

julia> c = tensor(); c[Tensor()]
1

julia> d = DenseLinear(a; basis = Basis('w':'z'))
DenseLinear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> d ⊗ d
DenseLinear{Tensor{Tuple{Char, Char}}, Int64} with 4 terms:
'x'⊗'x'+2*'y'⊗'x'+2*'x'⊗'y'+4*'y'⊗'y'

julia> d ⊗ d == a ⊗ a
true

julia> basis(d ⊗ d)
TensorBasis(Basis('w':1:'z'), Basis('w':1:'z'))
```
"""
function tensor end

const ⊗ = tensor
# or define it the other way around?

@multilinear tensor Tensor∘tuple

keeps_filtered(::ComposedFunction{<:Type{<:Tensor}, typeof(tuple)}, ::Type...) = true

function return_type(::ComposedFunction{<:Type{<:Tensor}, typeof(tuple)}, types::Type...)
    @assert !any(T -> T <: AbstractLinear, types)
    Tensor{Tuple{types...}}
end

# transpose of tensors

isrectangular(t::AbstractTensor{<:Tuple{Vararg{AbstractTensor}}}) =
    !isempty(t) && allequal(map(length, Tuple(t)))

function transpose_nosign(t::AbstractTensor{<:Tuple{Vararg{AbstractTensor}}})
    isrectangular(t) || error("all component tensors of the given tensor must have the same length")
    Tensor(map(Tensor∘tuple, map(Tuple, Tuple(t))...))
end

function _transpose_signexp(m, d2, tt)
    d1 = map(deg, tt[end])
    ds1 = revsums(d1)
    m += sum0(map(*, ds1, d2))
    if length(tt) == 1
        m
    else
        _transpose_signexp(m, map(+, d1, d2), tt[1:end-1])
    end
end

transpose_signexp(::AbstractTensor{<:Tuple{AbstractTensor}}) = Zero()

Base.@assume_effects :total function transpose_signexp(t::AbstractTensor{<:Tuple{Vararg{AbstractTensor}}})
    isrectangular(t) || error("all component tensors of the given tensor must have the same length")
    tt = map(Tuple, Tuple(t))
    _transpose_signexp(Zero(), map(deg, tt[end]), tt[1:end-1])
end

import Base: transpose

@linear transpose

"""
    transpose(t::AbstractTensor{T}) where T <: Tuple{Vararg{AbstractTensor}}

Return the transpose of a tensor `t` whose components are tensors of the same length.
In other words, the component `transpose(t)[i][j]` is `t[j][i]`.
If the components `t[i][j]` have non-zero degrees,
a sign is added according to the usual sign rule.
The tensor `t` must have at least one component. If all component tensors are empty,
then the empty tensor `Tensor()` is returned.

This function is linear.

# Examples

## Example without signs

```jldoctest transpose
julia> t = Tensor(Tensor("a", "b", "c"), Tensor("x", "y", "z"))
("a"⊗"b"⊗"c")⊗("x"⊗"y"⊗"z")

julia> transpose(t)
Linear1{Tensor{Tuple{Tensor{Tuple{String, String}}, Tensor{Tuple{String, String}}, Tensor{Tuple{String, String}}}}, Int64} with 1 term:
("a"⊗"x")⊗("b"⊗"y")⊗("c"⊗"z")
```

## Example with signs

As usual, the degree of a `String` is its length.

```jldoctest transpose
julia> $(@__MODULE__).deg(x::String) = length(x)

julia> transpose(t)   # same t as before
Linear1{Tensor{Tuple{Tensor{Tuple{String, String}}, Tensor{Tuple{String, String}}, Tensor{Tuple{String, String}}}}, Int64} with 1 term:
-("a"⊗"x")⊗("b"⊗"y")⊗("c"⊗"z")
```
"""
@linear_kw function transpose(t::AbstractTensor{<:Tuple{Vararg{AbstractTensor}}};
# TODO: sizehint?
        coefftype = missing,
        addto = missing,
        coeff = one(DefaultCoefftype),
        is_filtered::Bool = false)
    coefftype = unval(coefftype)
    tt = transpose_nosign(t)
    if addto !== missing || !has_char2(addto)
        m = transpose_signexp(t)
        coeff = withsign(m, coeff)
    end
    if addto === missing
        if coefftype !== missing
            coeff = convert(coefftype, coeff)
        end
        Linear1(tt => coeff; is_filtered)
    else
        addmul!(addto, tt, coeff; is_filtered)
    end
end

# multiplication of tensors

"""
    *(t1::AbstractTensor , t2::AbstractTensor, ...)

Return the product of the tensors, computed from the products of its components.
Signs are introduced according to the usual sign rule. If all degrees are integers,
then the coefficient type is `DefaultCoefftype`.

This function is linear.

See also: [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Example

```jldoctest
julia> import $(@__MODULE__): deg

julia> deg(x::String) = length(x);

julia> (s, t) = Tensor("ab", "c"), Tensor("x", "yz")
(Tensor("ab", "c"), Tensor("x", "yz"))

julia> s*t
Linear{Tensor{Tuple{String, String}}, Int64} with 1 term:
-"abx"⊗"cyz"

```
"""
function *(ts::AbstractTensor...; kw...)
    f = Tensor(ntuple(Returns(*), length(ts[1])))
    f(ts...; kw...)
end

hastrait(::typeof(*), ::Val, ::Type{<:AbstractTensor}...) = true

one(::Type{<:AbstractTensor{T}}) where T <: Tuple = Tensor(map(one, fieldtypes(T)))

one(::T) where T <: AbstractTensor = one(T)

# coproduct of tensors

"""
    coprod(t::T) where T <: AbstractTensor -> Linear{Tensor{Tuple{T,T}}}

Return the coproduct of a tensor, computed from the coproducts of its components.
Signs are introduced according to the usual sign rule. If all degrees are integers,
then the coefficient type is `DefaultCoefftype`.

This function is linear.

See also: [`coprod`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Example
```jldoctest
julia> import $(@__MODULE__): deg, coprod

julia> deg(x::String) = length(x);

julia> coprod(x::String) = Linear(Tensor(x[1:k], x[k+1:end]) => 1 for k in 1:length(x)-1);

julia> coprod("abc")
Linear{Tensor{Tuple{String, String}}, Int64} with 2 terms:
"a"⊗"bc"+"ab"⊗"c"

julia> t = Tensor("abc", "xyz")
"abc"⊗"xyz"

julia> coprod(t)
Linear{Tensor{Tuple{Tensor{Tuple{String, String}}, Tensor{Tuple{String, String}}}}, Int64} with 4 terms:
("ab"⊗"xy")⊗("c"⊗"z")+("a"⊗"xy")⊗("bc"⊗"z")+("a"⊗"x")⊗("bc"⊗"yz")-("ab"⊗"x")⊗("c"⊗"yz")
```
"""
function coprod(t::AbstractTensor; kw...)
    TensorSlurp(transpose)(map(coprod, Tuple(t))...; kw...)
end

# TODO: other keywords: also sizehint !?
hastrait(::typeof(coprod), ::Val{:coefftype}, ::Type{AbstractTensor}) = true
hastrait(::typeof(coprod), ::Val{:addto_coeff}, ::Type{AbstractTensor}) = true
hastrait(::typeof(coprod), ::Val{:isfiltered}, ::Type{AbstractTensor}) = true

#
# tensor slurping and splatting
#

export TensorSlurp, TensorSplat

"""
    TensorSlurp(f)

`TensorSlurp` turns a linear function acting on `Tensor` terms into a multilinear function.
This is similar to
[slurping](https://docs.julialang.org/en/v1/manual/faq/#...-combines-many-arguments-into-one-argument-in-function-definitions)
in Julia.

The new function always returns a linear combination, even if none of the arguments is a linear combination.
It recognizes all keyword arguments discussed for `@linear`. Unknown keyword arguments are passed on to `f`.

See also [`Tensor`](@ref), [`tensor`](@ref), [`TensorSplat`](@ref), [`@linear`](@ref).

# Examples

We use [`swap`](@ref) as an example of a function acting on tensors.
```jldoctest
julia> const f = TensorSlurp(swap)
TensorSlurp(Regroup{(1, 2), (2, 1)})

julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> b = Linear("w" => 3, "z" => -1)
Linear{String, Int64} with 2 terms:
3*"w"-"z"

julia> c = tensor(a, b)
Linear{Tensor{Tuple{Char, String}}, Int64} with 4 terms:
-'x'⊗"z"-2*'y'⊗"z"+3*'x'⊗"w"+6*'y'⊗"w"

julia> swap(c)
Linear{Tensor{Tuple{String, Char}}, Int64} with 4 terms:
-"z"⊗'x'+6*"w"⊗'y'+3*"w"⊗'x'-2*"z"⊗'y'

julia> f(a, b)
Linear{Tensor{Tuple{String, Char}}, Int64} with 4 terms:
-"z"⊗'x'+6*"w"⊗'y'+3*"w"⊗'x'-2*"z"⊗'y'

julia> f(a, b; addto = swap(c), coeff = -1)
Linear{Tensor{Tuple{String, Char}}, Int64} with 0 terms:
0
```
"""
struct TensorSlurp{F}
    f::F
end

@struct_equal_hash TensorSlurp

show(io::IO, g::TensorSlurp) = (print(io, "TensorSlurp("); show(io, g.f); print(io, ')'))

@multilinear g::TensorSlurp TermComposedFunction(g.f, Tensor∘tuple)

deg(g::TensorSlurp) = deg(g.f)

"""
    TensorSplat(f)

`TensorSplat` turns a multilinear function `f` into a linear function acting on terms of type `Tensor`.
This is similar to
[splatting](https://docs.julialang.org/en/v1/manual/faq/#...-splits-one-argument-into-many-different-arguments-in-function-calls)
in Julia.

When called with an argument of type `Tensor`, the new function returns the the value of `f` on
the components of the tensor (which may or may not be a linear combination).
All keyword arguments are passed on to `f` in this case.

When called with a linear combination as argument, the new function returns a linear combination.
It recognizes all keyword arguments discussed for `@linear`. Unknown keyword arguments are passed on to `f`.

See also [`Tensor`](@ref), [`tensor`](@ref), [`TensorSlurp`](@ref), [`@linear`](@ref).

# Examples

```jldoctest
julia> const f = MultilinearExtension(*)
MultilinearExtension(*)

julia> const g = TensorSplat(f)
TensorSplat(MultilinearExtension(*))

julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> b = Linear("w" => 3, "z" => -1)
Linear{String, Int64} with 2 terms:
3*"w"-"z"

julia> f(a, b)
Linear{String, Int64} with 4 terms:
3*"xw"-2*"yz"+6*"yw"-"xz"

julia> c = tensor(a, b)
Linear{Tensor{Tuple{Char, String}}, Int64} with 4 terms:
-'x'⊗"z"-2*'y'⊗"z"+3*'x'⊗"w"+6*'y'⊗"w"

julia> g(c)
Linear{String, Int64} with 4 terms:
3*"xw"-2*"yz"+6*"yw"-"xz"

julia> g(c; addto = f(a, b), coeff = -1)
Linear{String, Int64} with 0 terms:
0
```
"""
struct TensorSplat{F}
    f::F
end

@struct_equal_hash TensorSplat

show(io::IO, g::TensorSplat) = (print(io, "TensorSplat("); show(io, g.f); print(io, ')'))

(g::TensorSplat)(x::AbstractTensor; kw...) = g.f(Tuple(x)...; kw...)

@linear g::TensorSplat

hastrait(g::TensorSplat, prop::Val, ::Type{<:AbstractTensor{T}}) where T <: Tuple = hastrait(g.f, prop, fieldtypes(T)...)

keeps_filtered(g::TensorSplat, ::Type{<:AbstractTensor{T}}) where T <: Tuple = keeps_filtered(g.f, fieldtypes(T)...)

deg(g::TensorSplat) = deg(g.f)

#
# concatenating and flattening tensors
#

tuple_cat() = tuple()
tuple_cat(x) = Tuple(x)  # needed for conversion of Tensor (and ProductSimplex) to Tuple
tuple_cat(x, y, z...) = tuple_cat(tuple(x..., y...), z...)

@multilinear cat

"""
    $(@__MODULE__).cat(t::AbstractTensor...) -> Tensor

Concatenate the tensors given as arguments. This function is multilinear.

See also [`flatten`](@ref).

# Example

```jldoctest
julia> $(@__MODULE__).cat(Tensor('x'), Tensor('y', Tensor('z', 'w')))
'x'⊗'y'⊗('z'⊗'w')
```
"""
cat(t::AbstractTensor...) = Tensor(tuple_cat(t...))

keeps_filtered(::typeof(cat), ::Type{<:AbstractTensor}...) = true

function return_type(::typeof(cat), types::Type{<:AbstractTensor}...)
    TT = tuple_cat(map(fieldtypes, types)...)
    Tensor{Tuple{TT...}}
end

tuple_flatten(x) = (x,)
tuple_flatten(x::AbstractTensor) = tuple_cat(map(tuple_flatten, Tuple(x))...)

@linear flatten
# no keywords

"""
    flatten(t::AbstractTensor) -> Tensor
    flatten(a::AbstractLinear{<:AbstractTensor}) -> AbstractLinear{Tensor}

Recursively take all tensor components and concatenate the result.
This function is linear.

See also [`cat`](@ref).

# Example

```jldoctest
julia> t = Tensor('x', Tensor('y', Tensor('z', 'w')))
'x'⊗('y'⊗('z'⊗'w'))

julia> flatten(t)
'x'⊗'y'⊗'z'⊗'w'
```
"""
flatten(t::AbstractTensor) = Tensor(tuple_flatten(t))

keeps_filtered(::typeof(flatten), ::Type{<:AbstractTensor}) = true


#
# tensor product of maps
#


struct TensorMap{T<:Tuple,DS<:Tuple} <: AbstractTensor{T}
    ff::T
    degsums::DS
end

function tensormap(ff...)
    TensorMap(ff, revsums(map(deg, ff)))
end

Base.Tuple(f::TensorMap) = f.ff

degsums(f::TensorMap) = f.degsums

function deg(g::TensorMap)
    isempty(g) ? Zero() : deg(g[1])+g.degsums[1]
end

# evaluation of AbstractTensor

@multilinear tf::AbstractTensor

"""
    (tf::AbstractTensor)(tx::AbstractTensor...) -> Tensor

Evaluating an `AbstractTensor` on other `AbstractTensor`s (with the same number of components) is done
componentwise. If the degrees of the components and the maps are not all zero, then
the usual sign is introduced: whenever a map `f` is moved past a component `x`, then
this changes the sign by `(-1)^(deg(f)*deg(x))`.

# Examples

## Examples without degrees

```jldoctest tensorcall
julia> @linear f; f(x) = uppercase(x)
f (generic function with 2 methods)

julia> @linear g; g(x) = lowercase(x)
g (generic function with 2 methods)

julia> const h = Tensor(f, g)
f⊗g

julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> b = Linear('Z' => -1, 'W' => 3)
Linear{Char, Int64} with 2 terms:
-'Z'+3*'W'

julia> h(Tensor('x', 'Z'))
Linear{Tensor{Tuple{Char, Char}}, Int64} with 1 term:
'X'⊗'z'

julia> h(tensor(a, b))
Linear{Tensor{Tuple{Char, Char}}, Int64} with 4 terms:
-'X'⊗'z'+3*'X'⊗'w'+6*'Y'⊗'w'-2*'Y'⊗'z'
```

## Examples with degrees

We again take the length of a `String` as its degree.
```jldoctest tensorcall
julia> import $(@__MODULE__): deg

julia> deg(x::String) = length(x);

julia> struct P{T} y::T end; deg(p::P) = deg(p.y);

julia> @linear p::P; (p::P)(x) = x * p.y

julia> p = P("pp"); q = P("qqq")
P{String}("qqq")

julia> j = Tensor(p, q)
P{String}("pp")⊗P{String}("qqq")

julia> j(Tensor("x", "yy"))
Linear{Tensor{Tuple{String, String}}, Int64} with 1 term:
-"xpp"⊗"yyqqq"

julia> a = Linear("x" => 1, "yy" => 2)
Linear{String, Int64} with 2 terms:
"x"+2*"yy"

julia> b = tensor(a, a)
Linear{Tensor{Tuple{String, String}}, Int64} with 4 terms:
2*"yy"⊗"x"+2*"x"⊗"yy"+4*"yy"⊗"yy"+"x"⊗"x"

julia> j(b)
Linear{Tensor{Tuple{String, String}}, Int64} with 4 terms:
-2*"xpp"⊗"yyqqq"+2*"yypp"⊗"xqqq"-"xpp"⊗"xqqq"+4*"yypp"⊗"yyqqq"
```

## A multilinear example

```jldoctest
julia> @multilinear f; f(x::Char...) = join(x, '&');

julia> @multilinear g; g(x::Char...) = join(x, '@');

julia> f('a', 'p', 'x')
"a&p&x"

julia> Tensor(f, g)(Tensor('a', 'b'), Tensor('p', 'q'), Tensor('x', 'y'))
Linear{Tensor{Tuple{String, String}}, Int64} with 1 term:
"a&p&x"⊗"b@q@y"
```
"""
function (tf::AbstractTensor{<:Tuple{Vararg{Any,N}}})(ttx::Vararg{AbstractTensor{<:Tuple{Vararg{Any,N}}},M};
        coefftype = Sign,
        addto = zero(linear_return_type(tf, unval(coefftype), map(typeof, ttx)...)),
        coeff = ONE,
        is_filtered::Bool = false,
        kw...) where {N,M}

    tfx = map(Tuple(tf), map(Tuple, ttx)...) do f, x...
        TryLinearKw(f)(x...; is_filtered, kw...)
    end

    tensor_if = is_filtered && all(map(Tuple(tf), ttx, tfx) do f, tx, fx
            fx isa AbstractLinear || keeps_filtered(f, map(typeof, Tuple(tx))...)
    end)

    if !has_char2(map(_coefftype, tfx)...; kw...)
        m = transpose_signexp(Tensor(tf, ttx...))
        coeff = withsign(m, coeff)
    else
        m = Zero()
    end

    tensor(tfx...; addto, coeff, is_filtered = tensor_if, kw...)
end

hastrait(::AbstractTensor, ::Val, types::Type...) = true

function return_type(tf::AbstractTensor{<:Tuple{Vararg{Any,N}}}, types::Vararg{Type{<:AbstractTensor{<:Tuple{Vararg{Any,N}}}},M}) where {N,M}
    tt = map(fieldtypes, types)

    rt = ntuple(Val(N)) do i
        return_type(tf[i], map(T -> T[i], tt)...)
    end

    # compute sign type
    S = if N == 0
        Sign
    else
        st = map(Fix1(map, Fix1(return_type, deg)), tt)
        ft = map(Fix1(return_type, deg), fieldtypes(typeof(tf)))
        fst = (ft, st...)
        fst4 = ntuple(Val(N-1)) do i
            fst3 = ntuple(Val(M)) do j
                fst2 = ntuple(j) do k
                    fst1 = map(Fix1(promote_type_product, fst[j+1][i]), fst[k][i+1:end])
                    promote_type(fst1...)
                end
                promote_type(fst2...)
            end
            promote_type(fst3...)
        end
        signtype(promote_type(fst4...))
    end

    U = Tensor{Tuple{map(_termtype, rt)...}}
    R = promote_type(S, map(_coefftype, rt)...)
    Linear{U, R <: Sign ? DefaultCoefftype : R}
end

# differential

function tensor_diff(addto, coeff, x, dx, degx, sizehint)
    isempty(dx) && return addto
    dx1, dx... = dx
    degx1, degx... = degx
    coeff = withsign(degx1, coeff)
    k = length(x)-length(dx)
    tensor(x[1:k-1]..., dx1, x[k+1:end]...; addto, coeff, sizehint)
    tensor_diff(addto, coeff, x, dx, degx, sizehint)
end

"""
    diff(t::T) where T <: AbstractTensor -> Linear{T}

Return the differential of the tensor `t` by differentiating each tensor factor at a time
and adding signs according to the degrees of the components. The coefficient type is usually
`DefaultCoefftype`. However, if the degrees of the tensor components are not integers, then
the coefficient type is chosen such that it can accommodate the signs.

See also [`diff`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Example

As usual, the degree of a string is its length.
```jldoctest
julia> import $(@__MODULE__): deg, diff

julia> deg(x::String) = length(x);

julia> function diff(x::String)
           if isempty(x) || x[1] == 'D'
               zero(Linear1{String,Int})
           else
               Linear1('D'*x => 1)\
           end
       end;

julia> dx = diff("x")
Linear1{String, Int64} with 1 term:
"Dx"

julia> diff(dx)
Linear1{String, Int64} with 0 terms:
0

julia> t = Tensor("a", "bb", "ccc")
"a"⊗"bb"⊗"ccc"

julia> diff(t)
Linear{Tensor{Tuple{String, String, String}}, Int64} with 3 terms:
-"a"⊗"Dbb"⊗"ccc"+"Da"⊗"bb"⊗"ccc"-"a"⊗"bb"⊗"Dccc"
```
"""
@linear_kw function diff(t::T;
        coefftype = Sign,
        addto = zero(linear_return_type(diff, unval(coefftype), T)),
        coeff = ONE,
        is_filtered::Bool = false,
        sizehint::Bool = true) where T <: AbstractTensor

    x = Tuple(t)
    kwc = has_char2(unval(coefftype)) ? (; coefftype) : (;)

    dx = map(x) do y
        Y = typeof(y)
        kwd = has_isfiltered(diff, Y) ? (; is_filtered) : (;)
        if has_coefftype(diff, Y)
            kwd = push_kw(kwd; kwc...)
        end
        diff(y; kwd...)
    end

    if has_char2(unval(coefftype))
        degx = ntuple(Returns(Zero()), length(x))
    else
        degx = (Zero(), map(deg, x[1:end-1])...)
    end

    tensor_diff(addto, coeff, x, dx, degx, sizehint)
end

function return_type(::typeof(diff), ::Type{Tensor{T}}) where T <: Tuple
    DT = map(Fix1(return_type, diff), fieldtypes(T))
    RT = if fieldcount(T) > 1
        map(signtype ∘ Fix1(return_type, deg), fieldtypes(T)[1:end-1])
    else
        ()
    end
    U = map(promote_typejoin, map(_termtype, DT), fieldtypes(T))
    R = promote_type(map(_coefftype, DT)..., RT...)
    Linear{Tensor{Tuple{U...}}, R <: Sign ? DefaultCoefftype : R}
end
