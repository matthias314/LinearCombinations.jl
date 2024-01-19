#
# AbstractTensor
#

export AbstractTensor

using StructEqualHash: typehash

"""
    AbstractTensor{T<:Tuple}

The supertype of all tensor types. Currently the only subtype is `Tensor`.

See [`Tensor`](@ref), [`tensor`](@ref), [`Tuple(t::AbstractTensor)`](@ref).
"""
abstract type AbstractTensor{T<:Tuple} end

(::Type{T})(x...) where T <: AbstractTensor = T(x)

"""
    Tuple(t::AbstractTensor{T}) -> T <: Tuple

Return the tuple of components of `t`.
"""
Base.Tuple(t::AbstractTensor) = error_missing(typeof(t))

==(t1::AbstractTensor, t2::AbstractTensor) = Tuple(t1) == Tuple(t2)

Base.hash(t::AbstractTensor, h::UInt) = hash((Tuple(t),), typehash(AbstractTensor, h))

length(t::AbstractTensor) = length(Tuple(t))

firstindex(t::AbstractTensor) = 1
lastindex(t::AbstractTensor) = length(t)

iterate(t::AbstractTensor, state...) = iterate(Tuple(t), state...)

@propagate_inbounds getindex(t::AbstractTensor, k) = Tuple(t)[k]

function show(io::IO, t::AbstractTensor)
    if isempty(t)
        print(io, "()")
    else
        join(io, (x isa AbstractTensor && !isempty(x) ? "($x)" : x for x in t), '⊗')
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

keeps_filtered(::Type{<:AbstractTensor}, T::Type...) = true

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
created using `tensor`. `LinearCombinations` takes pure tensors as basis elements.

A `Tensor` can be created out of a `Tuple` or out of the individual components.
The second form is not available if the tensor has a tuple as its only component.

`Tensor` implements the
[iteration](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration)
and
[indexing](https://docs.julialang.org/en/v1/manual/interfaces/#Indexing)
interfaces. This makes for example splatting available for tensors, and
the `i`-th component of `t::Tensor` can be accessed as `t[i]`.

Tensors can be nested. Different bracketings lead to different tensors. The functions
`cat`, `flatten`, `swap` and `regroup` are provided to make rearranging tensors more easily.

Note that the type parameter of `Tensor` is always a `Tuple`. For instance, the type of
a `Tensor` with two components of types `T1` and `T2` is `Tensor{Tuple{T1,T2}}`, not
`Tensor{T1,T2}`.

See also [`tensor`](@ref), [`cat`](@ref), [`flatten`](@ref), [`regroup`](@ref), [`swap`](@ref).

# Examples

```jldoctest
julia> t = Tensor('x', 'y', "z")
x⊗y⊗z

julia> typeof(t)
Tensor{Tuple{Char, Char, String}}

julia> Tuple(t)
('x', 'y', "z")

julia> length(t), t[2], t[end]
(3, 'y', "z")

julia> a = Linear('x' => 1, 'y' => 2)
x+2*y

julia> b = Linear(Tensor('x', 'z') => 1, Tensor('y', 'z') => 2)
x⊗z+2*y⊗z

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

Base.Tuple(t::Tensor) = t.a

const Tensor_func = @Function(Tensor)
keeps_filtered(::typeof(Tensor_func), T::Type...) = keeps_filtered(Tensor, T...)

# @multilinear tensor Tensor_func
@multilinear_noesc tensor Tensor
# @multilinear_noesc tensor Tensor{Tuple{TT...}}

"""
    tensor(xs...) -> Linear{Tensor}
    x1 ⊗ x2 ⊗ ... -> Linear{Tensor}

`tensor` is the multilinear extension of `Tensor`. `⊗` is a synomym for `tensor`.
Note that `tensor` always returns a linear combination.

See also [`Tensor`](@ref), [`@multilinear`](@ref)

# Examples

```jldoctest
julia> a, b = Linear('x' => 1, 'y' => 2), Linear("w" => 3, "z" => -1)
(x+2*y, 3*w-z)

julia> tensor(a, "w")
x⊗w+2*y⊗w

julia> tensor(a, b)
-2*y⊗z+3*x⊗w-x⊗z+6*y⊗w

julia> tensor('x', b, a; coefftype = Float64)
-2.0*x⊗z⊗y-x⊗z⊗x+6.0*x⊗w⊗y+3.0*x⊗w⊗x

julia> a = tensor(); a[Tensor()]
1
```
"""
function tensor end

const ⊗ = tensor
# or define it the other way around?

Base.@assume_effects :total function tensor_return_type(T...)
    # T = map(typeof, x)
    RR = map(_coefftype, T)
    UU = map(_termtype, T)
    R = promote_type(Sign, RR...)
    if R == Sign
        R = DefaultCoefftype
    end
    if all(isconcretetype, UU)
        Linear{Tensor{Tuple{UU...}},R}
    else
        Linear{Tensor{T} where T<:Tuple{UU...},R}
    end
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
(a⊗b⊗c)⊗(x⊗y⊗z)

julia> transpose(t)
(a⊗x)⊗(b⊗y)⊗(c⊗z)
```

## Example with signs

As usual, the degree of a `String` is its length.

```jldoctest transpose
julia> LinearCombinations.deg(x::String) = length(x)

julia> transpose(t)   # same t as before
-(a⊗x)⊗(b⊗y)⊗(c⊗z)
```
"""
@linear_kw function transpose(t::AbstractTensor{<:Tuple{Vararg{AbstractTensor}}};
# TODO: sizehint?
        coefftype = missing,
        addto = missing,
        coeff = one(DefaultCoefftype),
        is_filtered = false)
    coefftype = unval(coefftype)
    tt = transpose_nosign(t)
    if !(has_char2(_coefftype(addto)) || (addto === missing && has_char2(coefftype)))
        m = transpose_signexp(t)
        coeff = signed(m, coeff)
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
julia> import LinearCombinations: deg

julia> deg(x::String) = length(x);

julia> (s, t) = Tensor("ab", "c"), Tensor("x", "yz")
(ab⊗c, x⊗yz)

julia> s*t
-abx⊗cyz

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
julia> import LinearCombinations: deg, coprod

julia> deg(x::String) = length(x);

julia> coprod(x::String) = Linear(Tensor(x[1:k], x[k+1:end]) => 1 for k in 1:length(x)-1);

julia> coprod("abc")
a⊗bc+ab⊗c

julia> t = Tensor("abc", "xyz")
abc⊗xyz

julia> coprod(t)
-(ab⊗x)⊗(c⊗yz)+(a⊗xy)⊗(bc⊗z)+(ab⊗xy)⊗(c⊗z)+(a⊗x)⊗(bc⊗yz)
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
TensorSlurp(Regroup{(1, 2),(2, 1)})

julia> a, b = Linear('x' => 1, 'y' => 2), Linear("w" => 3, "z" => -1)
(x+2*y, 3*w-z)

julia> c = tensor(a, b)
-2*y⊗z+3*x⊗w-x⊗z+6*y⊗w

julia> swap(c)
3*w⊗x-z⊗x+6*w⊗y-2*z⊗y

julia> f(a, b)
3*w⊗x+6*w⊗y-z⊗x-2*z⊗y

julia> f(a, b; addto = swap(c), coeff = -1)
0
```
"""
struct TensorSlurp{F}
    f::F
end

@struct_equal_hash TensorSlurp

show(io::IO, g::TensorSlurp) = print(io, "TensorSlurp($(repr(g.f)))")

# @multilinear g::TensorSlurp (x...; kw...) -> g.f(Tensor(x); kw...)
@multilinear g::TensorSlurp LinearComposedFunction(g.f, Tensor_func)
# @multilinear_noesc g::TensorSlurp LinearComposedFunction(g.f, Tensor{Tuple{TT...}})

hastrait(g::TensorSlurp, prop::Val, T::Type...) = hastrait(g.f, prop, Tensor{Tuple{T...}})

keeps_filtered(g::TensorSlurp, T::Type...) = keeps_filtered(g.f, Tensor{Tuple{T...}})

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

julia> a, b = Linear('x' => 1, 'y' => 2), Linear("w" => 3, "z" => -1)
(x+2*y, 3*w-z)

julia> f(a, b)
3*xw-2*yz+6*yw-xz

julia> c = tensor(a, b)
-2*y⊗z+3*x⊗w-x⊗z+6*y⊗w

julia> g(c)
3*xw-2*yz+6*yw-xz

julia> g(c; addto = f(a, b), coeff = -1)
0
```
"""
struct TensorSplat{F}
    f::F
end

@struct_equal_hash TensorSplat

show(io::IO, g::TensorSplat) = print(io, "TensorSplat($(repr(g.f)))")

(g::TensorSplat)(x::AbstractTensor; kw...) = g.f(Tuple(x)...; kw...)

@linear g::TensorSplat

hastrait(g::TensorSplat, prop::Val, ::Type{<:AbstractTensor{T}}) where T <: Tuple = hastrait(g.f, prop, fieldtypes(T)...)

keeps_filtered(g::TensorSplat, ::Type{<:AbstractTensor{T}}) where T <: Tuple = keeps_filtered(g.f, fieldtypes(T)...)

deg(g::TensorSplat) = deg(g.f)

#
# concatenating and flattening tensors
#

_cat() = ()
_cat(x) = (x...,)
# needed for conversion of Tensor (and ProductSimplex) to Tuple
_cat(x, y, z...) = _cat((x..., y...), z...)

@multilinear cat

"""
    cat(t::AbstractTensor...) -> Tensor

Concatenate the tensors given as arguments. This function is multilinear.

See also [`flatten`](@ref).

# Example

```jldoctest
julia> LinearCombinations.cat(Tensor('x'), Tensor('y', Tensor('z', 'w')))
x⊗y⊗(z⊗w)
```
"""
cat(t::AbstractTensor...) = Tensor(_cat(t...))

# TODO: add keeps_filtered?

_flatten(x) = (x,)
_flatten(x::AbstractTensor) = _cat(map(_flatten, Tuple(x))...)

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
x⊗(y⊗(z⊗w))

julia> flatten(t)
x⊗y⊗z⊗w
```
"""
flatten(t::AbstractTensor) = Tensor(_flatten(t))

keeps_filtered(::typeof(flatten), ::Type{<:AbstractTensor}) = true


#
# tensor product of maps
#

export tensormap

struct TensorMap{T<:Tuple,DS<:Tuple} <: AbstractTensor{T}
    ff::T
    degsums::DS
end

function tensormap(ff...)
    TensorMap(ff, revsums(map(deg, ff)))
end

Tuple(f::TensorMap) = f.ff

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

julia> a, b = Linear('x' => 1, 'y' => 2), Linear('Z' => -1, 'W' => 3)
(x+2*y, -Z+3*W)

julia> h(Tensor('x', 'Z'))
X⊗z

julia> h(tensor(a, b))
6*Y⊗w-2*Y⊗z+3*X⊗w-X⊗z
```

## Examples with degrees

We again take the length of a `String` as its degree.
```jldoctest tensorcall
julia> import LinearCombinations: deg

julia> deg(x::String) = length(x);

julia> struct P{T} y::T end; deg(p::P) = deg(p.y);

julia> @linear p::P; (p::P)(x) = x * p.y

julia> p = P("pp"); q = P("qqq")
P{String}("qqq")

julia> j = Tensor(p, q)
P{String}("pp")⊗P{String}("qqq")

julia> j(Tensor("x", "yy"))
-xpp⊗yyqqq

julia> a = Linear("x" => 1, "yy" => 2)
x+2*yy

julia> b = tensor(a, a)
2*x⊗yy+4*yy⊗yy+x⊗x+2*yy⊗x

julia> j(b)
-xpp⊗xqqq-2*xpp⊗yyqqq+2*yypp⊗xqqq+4*yypp⊗yyqqq
```

## A multilinear example

```jldoctest
julia> @multilinear f; f(x::Char...) = join(x, '#');

julia> @multilinear g; g(x::Char...) = join(x, '@');

julia> f('a', 'p', 'x')
"a#p#x"

julia> Tensor(f, g)(Tensor('a', 'b'), Tensor('p', 'q'), Tensor('x', 'y'))
a#p#x⊗b@q@y
```
"""
function (tf::AbstractTensor)(ttx::Vararg{AbstractTensor,N};
        coeff = ONE,
        is_filtered::Bool = false,
        kw...) where N
    n = length(tf)
    all(==(n), map(length, ttx)) || error("all tensor arguments of $tf must have $n components")

    tfx = map(Tuple(tf), map(Tuple, ttx)...) do f, x...
        InnerKw(f; is_filtered, kw...)(x...)
    end

    tensor_if = is_filtered && all(map(Tuple(tf), ttx, tfx) do f, tx, fx
            fx isa AbstractLinear || keeps_filtered(f, map(typeof, Tuple(tx))...)
    end)

    if !has_char2(map(_coefftype, tfx)...; kw...)
        m = transpose_signexp(Tensor(tf, ttx...))
        coeff = signed(m, coeff)
    else
        m = Zero()
    end

    if haskey(kw, :addto) || haskey(kw, :coefftype)
        tensor(tfx...; coeff, is_filtered = tensor_if, kw...)
    else
        R = promote_type(map(_coefftype, tfx)..., sign_type(typeof(m)))
        if R === Sign
            R = DefaultCoefftype
        end
        tensor(tfx...; coefftype = Val(R), coeff, is_filtered = tensor_if, kw...)
    end
end

hastrait(::AbstractTensor, ::Val, types...) = true

#=
# TODO: this poses problems with inference
function return_type(g::TensorMap, ::Type{T}) where T <: Tensor
    types = T.parameters[1].parameters
    length(types) == length(g) || error("wrong number of tensor components")
    TT = ntuple(k -> return_type(g[k], types[k]), length(g))
    U = Tensor{Tuple{map(_termtype, TT)...}}
    S = promote_type(Sign, map(_coefftype, TT)...)
    @show TT U S
    Linear{U,S}
end
=#

# differential

function tensor_diff(addto, coeff, x, dx, degx, sizehint)
    isempty(dx) && return addto
    dx1, dx... = dx
    degx1, degx... = degx
    coeff = signed(degx1, coeff)
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
julia> import LinearCombinations: deg, diff

julia> deg(x::String) = length(x);

julia> function diff(x::String)
           if isempty(x) || x[1] == 'D'
               zero(Linear1{String,Int})
           else
               Linear1('D'*x => 1)\
           end
       end;

julia> dx = diff("x")
Dx

julia> diff(dx)
0

julia> t = Tensor("a", "bb", "ccc")
a⊗bb⊗ccc

julia> diff(t)
Da⊗bb⊗ccc-a⊗Dbb⊗ccc-a⊗bb⊗Dccc
```
"""
@linear_kw function diff(t::T;
        coefftype = missing,
        addto = missing,
        coeff = ONE,
        is_filtered::Bool = false,
        sizehint::Bool = true) where T <: AbstractTensor
    x = Tuple(t)
    isempty(x) && return zero(Linear{T,DefaultCoefftype})

    if addto !== missing
        R = _coefftype(addto)
    elseif coefftype !== missing
        R = unval(coefftype)
    else
        R = missing
    end
    kwc = has_char2(R) ? (; coefftype = R) : (;)

    dx = map(x) do y
        Y = typeof(y)
        kwd = has_isfiltered(diff, Y) ? (; is_filtered) : (;)
        if has_coefftype(diff, Y)
            kwd = push_kw(kwd; kwc...)
        end
        diff(y; kwd...)
    end

    if has_char2(R)
        degx = ntuple(Returns(Zero()), length(x))
    else
        degx = (Zero(), map(deg, x[1:end-1])...)
    end

    if addto === missing
        if R === missing
            R = promote_type(map(_coefftype, dx)..., map(sign_type ∘ typeof, degx)...)
        end
        addto = zero(Linear{T,R})
    end
    tensor_diff(addto, coeff, x, dx, degx, sizehint)
end
