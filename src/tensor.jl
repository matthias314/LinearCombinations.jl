#
# AbstractTensor
#

export AbstractTensor

using StructEqualHash: typehash

abstract type AbstractTensor{T<:Tuple} end

(::Type{T})(x...) where T <: AbstractTensor = T(x)

factors(t::T) where T <: AbstractTensor = error_missing(T)

==(t1::AbstractTensor, t2::AbstractTensor) = factors(t1) == factors(t2)

Base.hash(t::AbstractTensor, h::UInt) = hash((factors(t),), typehash(AbstractTensor, h))

length(t::AbstractTensor) = length(factors(t))

firstindex(t::AbstractTensor) = 1
lastindex(t::AbstractTensor) = length(t)

iterate(t::AbstractTensor, state...) = iterate(factors(t), state...)

@propagate_inbounds getindex(t::AbstractTensor, k) = factors(t)[k]

function show(io::IO, t::AbstractTensor)
    if isempty(t)
        print(io, "()")
    else
        join(io, (x isa AbstractTensor && !isempty(x) ? "($x)" : x for x in t), '⊗')
    end
end

copy(t::AbstractTensor) = t

convert(::Type{T}, t::AbstractTensor) where T <: AbstractTensor = T(factors(t))

deg(t::AbstractTensor) = sum0(deg, factors(t))
# type inference doesn't work without "factors"

# factor_types(::Type{T}) where T <: AbstractTensor = return_type(factors, T).parameters
factor_types(::Type{T}) where T <: AbstractTensor = (T.parameters[1].parameters...,)

#=
deg_return_type_tensor(R, T...) = promote_type(R, map(Fix1(return_type, deg), T)...)

# TODO: needed?
return_type(::typeof(deg), ::Type{T}) where T <: AbstractTensor =
    deg_return_type_tensor(Int, factor_types(T)...)
=#

_degsums(dt) = dt
_degsums(dt, t...) = _degsums((deg(t[end])+dt[1], dt...), t[1:end-1]...)

degsums(::Tuple{}) = ()
degsums(t::Tuple) = _degsums((Zero(),), t[2:end]...)

degsums(t::AbstractTensor) = degsums(factors(t))

linear_filter(t::AbstractTensor) = all(linear_filter, t)

keeps_filtered(::Type{<:AbstractTensor}, T::Type...) = true

@linear_broadcastable AbstractTensor

#
# Tensor datatype
#

export Tensor, factors, tensor, ⊗, cat, flatten

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
factors(t::Tensor) = t.a

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

# multiplication of tensors

mul_rt_tensor(t1::Tuple{}, t2::Tuple{}, T...) = T
mul_rt_tensor(t1, t2, T...) =
    mul_rt_tensor(t1[2:end], t2[2:end], T..., return_type(*, typeof(t1[1]), typeof(t2[1])))

tensor_mul_signexp(m, ::Tuple{}, ::Tuple{}) = m
tensor_mul_signexp(m, f1::Tuple, f2::Tuple) =
    tensor_mul_signexp(m + deg(f2[1])*sum0(deg, f1[2:end]), f1[2:end], f2[2:end])

function *(t1::AbstractTensor, t2::AbstractTensor;
        coefftype = begin
            TT = mul_rt_tensor(factors(t1), factors(t2))
            R = promote_type(Sign, map(_coefftype, TT)...)
            R == Sign ? DefaultCoefftype : R
        end,
        addto = begin
            TT = mul_rt_tensor(factors(t1), factors(t2))
            zero(Linear{Tensor{Tuple{map(_termtype, TT)...}},unval(coefftype)})
        end,
        coeff = ONE,
        is_filtered = false)
    length(t1) == length(t2) || error("argument tensors must have the same length")
    f1 = factors(t1)
    f2 = factors(t2)
    m = tensor_mul_signexp(Zero(), f1, f2)
    tensor(map(*, f1, f2)...; addto, coeff = signed(m, coeff), is_filtered)
    # TODO: does "is_filtered" make sense here?
end

function one(::Type{T}) where T <: AbstractTensor
    TT = factor_types(T)
    Tensor(ntuple(i -> one(TT[i]), length(TT)))
end

one(::T) where T <: AbstractTensor = one(T)

# coproduct of tensors

@linear_kw function coprod(t::T;
# TODO: use some multilinear function instead of "tensor" + reordering
        coefftype = begin
            TT = map(typeof, factors(t))
            R = promote_type(Sign, map(Fix1(linear_extension_coeff_type, coprod), TT)...)
            R == Sign ? DefaultCoefftype : R
        end,
        addto = zero(Linear{Tensor{Tuple{T,T}},unval(coefftype)}),
        coeff = ONE,
        is_filtered = false) where T <: AbstractTensor
    n = length(t)
    for (tt, c) in tensor(map(coprod, factors(t))...)
        f1 = ntuple(i -> tt[i][1], n)
        f2 = ntuple(i -> tt[i][2], n)
        m = tensor_mul_signexp(Zero(), f1, f2)
        addto .+= signed(m, c*coeff) .* Tensor(Tensor(f1), Tensor(f2))
    end
    addto
end

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
@multilinear g::TensorSlurp ComposedFunctionOuterKw(g.f, Tensor_func)
# @multilinear_noesc g::TensorSlurp ComposedFunctionOuterKw(g.f, Tensor{Tuple{TT...}})

hastrait(g::TensorSlurp, prop::Val, T::Type...) = hastrait(g.f, prop, Tensor{Tuple{T...}})

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

(g::TensorSplat)(x::AbstractTensor; kw...) = g.f(factors(x)...; kw...)

@linear g::TensorSplat

hastrait(g::TensorSplat, prop::Val, ::Type{T}) where T <: AbstractTensor = hastrait(g.f, prop, factor_types(T)...)

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
_flatten(x::AbstractTensor) = _cat(map(_flatten, factors(x))...)

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

"""
    tensormap(f...) -> TensorMap

Return the `TensorMap` object representing the tensor products of the given maps.

Evaluating a `TensorMap` on a `Tensor` (with the same number of components) is done
componentwise. If the degrees of the components and the maps are not all zero, then
the usual sign is introduced: whenever a map `f` is moved past a component `x`, then
this changes the sign by `(-1)^(deg(f)*deg(x))`.

# Examples

## Examples without degrees

```jldoctest tensormap
julia> @linear f; f(x) = uppercase(x)
f (generic function with 2 methods)

julia> @linear g; g(x) = lowercase(x)
g (generic function with 2 methods)

julia> const h = tensormap(f, g)
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
```jldoctest tensormap
julia> import LinearCombinations: deg

julia> deg(x::String) = length(x);

julia> struct P{T} y::T end; deg(p::P) = deg(p.y);

julia> @linear p::P; (p::P)(x) = x * p.y

julia> p = P("pp"); q = P("qqq")
P{String}("qqq")

julia> j = tensormap(p, q)
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
"""
function tensormap(ff...)
    TensorMap(ff, degsums(ff))
end

factors(f::TensorMap) = f.ff

degsums(f::TensorMap) = f.degsums

function deg(g::TensorMap)
    isempty(g) ? Zero() : deg(g[1])+g.degsums[1]
end

# evaluation of AbstractTensor

@linear tf::AbstractTensor

@linear_kw function (tf::AbstractTensor)(tx::AbstractTensor;
        coefftype = missing,
        addto = missing,
        coeff = ONE,
        is_filtered::Bool = false,
        sizehint::Bool = true,
        kw...)
    n = length(tf)
    n == length(tx) || error("wrong number of tensor factors")

    kwt = kw
    if addto !== missing
        hc2 = has_char2(_coefftype(addto))
        kwt = push_kw(kwt; addto)
    elseif coefftype !== missing
        hc2 = has_char2(unval(coefftype))
        kwt = push_kw(kwt; coefftype)
    else
        hc2 = false
    end

    if !hc2
        dfs = degsums(tf)
        dx = map(deg, factors(tx))
        m = sum0(splat(*), zip(dfs, dx))
        coeff = signed(m, coeff)
    end

    tfx = ntuple(n) do i
        kwi = kw
        if has_isfiltered(tf[i], typeof(tx[i]))
            kwi = push_kw(kwi; is_filtered = true)
        end
        if has_sizehint(tf[i], typeof(tx[i]))
            kwi = push_kw(kwi; sizehint)
        end
        tf[i](tx[i]; kwi...)
    end

    is_filtered = is_filtered && all(ntuple(n) do i
            tx[i] isa AbstractLinear || keeps_filtered(tf[i], typeof(tx[i]))
        end)

    tensor(tfx...; coeff, is_filtered, kwt...)
end

#=
# TODO: this poses problems with inference
function return_type(g::TensorMap, ::Type{T}) where T <: Tensor
    types = T.parameters[1].parameters
    length(types) == length(g) || error("wrong number of tensor factors")
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

@linear_kw function diff(t::T;
        coefftype = missing,
        addto = missing,
        coeff = ONE,
        is_filtered::Bool = false,
        sizehint::Bool = true) where T <: AbstractTensor
    x = factors(t)
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
