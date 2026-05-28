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
'x'⊗'z'+2*'y'⊗'z'

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

julia> Tensor() |> typeof
Tensor{Tuple{}}
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
for `tensor`. The return value is a `Tensor` if no argument is of type `AbstractLinear`.
It is `DenseLinear` if all arguments are `DenseLinear`; the corresponding basis is the
`TensorBasis` of the bases of the arguments. In all other cases the return type is
`Linear`. This can be overriden via the `addto` keyword argument.

Remember that `⊗` is not associative in Julia, unlike `*`. Writing tensors
with more than two factors in infix notation therefore produces nested tensors.

See also [`Tensor`](@ref), [`@multilinear`](@ref), [`TensorBasis`](@ref).

# Examples

```jldoctest
julia> tensor('x', "w")
'x'⊗"w"

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
-2*'y'⊗"z"+3*'x'⊗"w"+6*'y'⊗"w"-'x'⊗"z"

julia> tensor('x', b, a; coefftype = Float64)
Linear{Tensor{Tuple{Char, String, Char}}, Float64} with 4 terms:
3.0*'x'⊗"w"⊗'x'-'x'⊗"z"⊗'x'+6.0*'x'⊗"w"⊗'y'-2.0*'x'⊗"z"⊗'y'

julia> 'x' ⊗ b ⊗ a
Linear{Tensor{Tuple{Tensor{Tuple{Char, String}}, Char}}, Int64} with 4 terms:
-2*('x'⊗"z")⊗'y'-('x'⊗"z")⊗'x'+3*('x'⊗"w")⊗'x'+6*('x'⊗"w")⊗'y'

julia> tensor('x', b, a) == 'x' ⊗ b ⊗ a
false

julia> tensor() |> typeof
Tensor{Tuple{}}

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

# needed to avoid methods for `DenseLinear` arguments
tensor() = Tensor()
return_type(::typeof(tensor)) = Tensor{Tuple{}}

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
If the components `t[i][j]` may have non-zero degrees, a sign is added according
to the usual sign rule. In this case the return type is `Linear1` instead of `Tensor`.
The tensor `t` must have at least one component. If all component tensors are empty,
then the empty tensor `Tensor()` is returned.

This function is linear.

# Examples

## Example without signs

```jldoctest
julia> t = Tensor(Tensor("a", "b", "c"), Tensor("x", "y", "z"))
("a"⊗"b"⊗"c")⊗("x"⊗"y"⊗"z")

julia> transpose(t)
("a"⊗"x")⊗("b"⊗"y")⊗("c"⊗"z")
```

## Example with degrees

The degree of a `GradedString` (created with the `gr""` string macro) is its length.
```jldoctest
julia> using $(@__MODULE__).TestHelpers: GradedString, @gr_str

julia> t = Tensor(Tensor(gr"a", gr"b", gr"c"), Tensor(gr"x", gr"y", gr"z"))
(gr"a"⊗gr"b"⊗gr"c")⊗(gr"x"⊗gr"y"⊗gr"z")

julia> transpose(t)
Linear1{Tensor{Tuple{Tensor{Tuple{GradedString, GradedString}}, Tensor{Tuple{GradedString, GradedString}}, Tensor{Tuple{GradedString, GradedString}}}}, Int64} with 1 term:
-(gr"a"⊗gr"x")⊗(gr"b"⊗gr"y")⊗(gr"c"⊗gr"z")
```
"""
transpose(t::AbstractTensor{<:Tuple{Vararg{AbstractTensor}}})

@linear_kw function transpose_sign(t::T;
        coefftype = Sign,
        addto = zero(linear_return_type(transpose, unval(coefftype), T)),
        coeff = one(DefaultCoefftype),
        is_filtered::Bool = false) where T
    c = has_char2(addto) ? coeff : withsign(transpose_signexp(t), coeff)
    addmul!(addto, transpose_nosign(t), c; is_filtered)
end

function transpose_signexp_type(T::Type{<:AbstractTensor{<:NTuple{M,AbstractTensor{<:NTuple{N,Any}}}}} where {M,N})
    @foldable
    M = length(fieldtypes(T))
    N = length(fieldtypes(fieldtypes(T)[1]))
    DT = map(Fix1(map, Fix1(return_type, deg)) ∘ fieldtypes, fieldtypes(T))
    promote_type(Zero, (promote_type_product(DT[j+1][i], DT[l][k]) for i in 1:N-1, j in 1:M-1 for k in i+1:N, l in 1:j)...)
end

function transpose_nosign_return_type(T::Type)
    @foldable
    tt = map(tuple, map(fieldtypes, fieldtypes(T))...)
    Tensor{Tuple{map(t -> Tensor{Tuple{t...}}, tt)...}}
end

function transpose_sign_return_type(T::Type)
    @foldable
    R = signtype(transpose_signexp_type(T))
    Linear1{transpose_nosign_return_type(T), R <: Sign ? DefaultCoefftype : R}
end

function return_type(::typeof(transpose), T::Type{<:AbstractTensor{<:NTuple{M,AbstractTensor{<:NTuple{N,Any}}}}} where {M,N})
    @foldable
    if transpose_signexp_type(T) === Zero
        transpose_nosign_return_type(T)
    else
        transpose_sign_return_type(T)
    end
end

keeps_filtered(::typeof(transpose), ::Type{<:AbstractTensor{<:NTuple{M,AbstractTensor{<:NTuple{N,Any}}}}}) where {M,N} = true

function hastrait(::typeof(transpose), trait::Val, T::Type{<:AbstractTensor{<:NTuple{M,AbstractTensor{<:NTuple{N,Any}}}}}) where {M,N}
    transpose_signexp_type(T) !== Zero && hastrait(transpose_sign, trait, T)
end

function transpose(t::AbstractTensor{<:NTuple{M,AbstractTensor{<:NTuple{N,Any}}}}; kw...) where {M,N}
    if transpose_signexp_type(typeof(t)) === Zero
        transpose_nosign(t; kw...)
    else
        transpose_sign(t; kw...)
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

# Example without degrees

```jldoctest
julia> (s, t) = Tensor("ab", "c"), Tensor("x", "yz");

julia> s*t
"abx"⊗"cyz"
```

# Example with degrees

The degree of a `GradedString` (created with the `gr""` string macro) is its length.
```jldoctest
julia> using $(@__MODULE__).TestHelpers: GradedString, @gr_str

julia> (s, t) = Tensor(gr"ab", gr"c"), Tensor(gr"x", gr"yz");

julia> s*t
Linear1{Tensor{Tuple{GradedString, GradedString}}, Int64} with 1 term:
-gr"abx"⊗gr"cyz"
```
"""
function *(ts::AbstractTensor{<:NTuple{N,Any}}...; kw...) where N
    f = Tensor(ntuple(Returns(*), Val(N)))
    f(ts...; kw...)
end

hastrait(::typeof(*), ::Val, ::Type{<:AbstractTensor{<:NTuple{N,Any}}}...) where N = true  # TODO: is "true" OK?

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

The degree of a `GradedString` (created with the `gr""` string macro) is its length.
```jldoctest
julia> using $(@__MODULE__).TestHelpers: GradedString, @gr_str

julia> import $(@__MODULE__): coprod

julia> coprod(x::GradedString) = Linear(Tensor(x[1:k], x[k+1:end]) => 1 for k in 0:length(x));

julia> coprod(gr"xy")
Linear{Tensor{Tuple{GradedString, GradedString}}, Int64} with 3 terms:
gr"x"⊗gr"y"+gr""⊗gr"xy"+gr"xy"⊗gr""

julia> Tensor(gr"x", gr"y") |> coprod
Linear{Tensor{Tuple{Tensor{Tuple{GradedString, GradedString}}, Tensor{Tuple{GradedString, GradedString}}}}, Int64} with 4 terms:
(gr"x"⊗gr"y")⊗(gr""⊗gr"")+(gr"x"⊗gr"")⊗(gr""⊗gr"y")-(gr""⊗gr"y")⊗(gr"x"⊗gr"")+(gr""⊗gr"")⊗(gr"x"⊗gr"y")
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
-2*'y'⊗"z"+3*'x'⊗"w"+6*'y'⊗"w"-'x'⊗"z"

julia> swap(c)
Linear{Tensor{Tuple{String, Char}}, Int64} with 4 terms:
-2*"z"⊗'y'+6*"w"⊗'y'+3*"w"⊗'x'-"z"⊗'x'

julia> f(a, b)
Linear{Tensor{Tuple{String, Char}}, Int64} with 4 terms:
6*"w"⊗'y'-2*"z"⊗'y'+3*"w"⊗'x'-"z"⊗'x'

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

@multilinear g::TensorSlurp LinearComposedFunction(g.f, Tensor∘tuple)

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
-2*'y'⊗"z"+3*'x'⊗"w"+6*'y'⊗"w"-'x'⊗"z"

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
    @foldable
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

# evaluation of AbstractTensor

@multilinear tf::AbstractTensor

"""
    (tf::AbstractTensor)(tx::AbstractTensor...)

Evaluating an `AbstractTensor` on other `AbstractTensor`s (with the same number of components) is done
componentwise. If the degrees of the components or the maps may be non-zero, then
the usual sign is introduced: whenever a map `f` is moved past a component `x`, then
this changes the sign by `(-1)^(deg(f)*deg(x))`.
In this case the return type is `Linear1` instead of `Tensor` if no map returns a linear combination.

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
'X'⊗'z'

julia> h(tensor(a, b))
Linear{Tensor{Tuple{Char, Char}}, Int64} with 4 terms:
-2*'Y'⊗'z'+6*'Y'⊗'w'+3*'X'⊗'w'-'X'⊗'z'

julia> Tensor()(Tensor())
()
```

## Examples with degrees

The degree of a `GradedString` (created with the `gr""` string macro) is its length.
```jldoctest tensorcall
julia> using $(@__MODULE__).TestHelpers: GradedString, @gr_str

julia> using Base: Fix2

julia> j = Tensor(Fix2(*, gr"pp"), Fix2(*, gr"qqq"))
Fix2{typeof(*), GradedString}(*, gr"pp")⊗Fix2{typeof(*), GradedString}(*, gr"qqq")

julia> j(Tensor(gr"x", gr"yy"))
Linear1{Tensor{Tuple{GradedString, GradedString}}, Int64} with 1 term:
-gr"xpp"⊗gr"yyqqq"

julia> a = Linear(gr"x" => 1, gr"yy" => 2)
Linear{GradedString, Int64} with 2 terms:
2*gr"yy"+gr"x"

julia> b = tensor(a, a)
Linear{Tensor{Tuple{GradedString, GradedString}}, Int64} with 4 terms:
gr"x"⊗gr"x"+2*gr"yy"⊗gr"x"+4*gr"yy"⊗gr"yy"+2*gr"x"⊗gr"yy"

julia> j(b)
Linear{Tensor{Tuple{GradedString, GradedString}}, Int64} with 4 terms:
-gr"xpp"⊗gr"xqqq"+2*gr"yypp"⊗gr"xqqq"-2*gr"xpp"⊗gr"yyqqq"+4*gr"yypp"⊗gr"yyqqq"
```

## A multilinear example

```jldoctest
julia> Tensor(*, *)('a'⊗'b', 'p'⊗'q', 'x'⊗'y')
"apx"⊗"bqy"
```
"""
AbstractTensor(::AbstractTensor)  # to work around JuliaDocs/Documenter.jl#558

function (tf::AbstractTensor{<:NTuple{N,Any}})(txs::Vararg{AbstractTensor{<:NTuple{N,Any}},M}; kw...) where {N,M}
    if return_type(tf, map(typeof, txs)...) <: AbstractLinear
        tensor_callable_linear(tf, txs...; kw...)
    else
        tensor_callable_tensor(tf, txs...; kw...)
    end
end

function tensor_callable_tensor(tf::AbstractTensor{<:NTuple{N,Any}}, txs::Vararg{AbstractTensor{<:NTuple{N,Any}},M}) where {N,M}
    tc = transpose(Tensor((tf, txs...)))
    tensor(map(TensorSplat(Eval), Tuple(tc))...)
end

@linear_kw function tensor_callable_linear(tf::AbstractTensor{<:NTuple{N,Any}}, txs::Vararg{AbstractTensor{<:NTuple{N,Any}},M};
        coefftype = Val(Sign),
        addto = zero(linear_return_type(tf, unval(coefftype), map(typeof, txs)...)),
        coeff = ONE,
        is_filtered::Bool = false) where {N,M}
    R = _coefftype(addto)
    inner_kw = has_char2(R) ? (; coefftype = Val(R), is_filtered) : (; is_filtered)
    tc = Tuple(transpose_nosign(Tensor((tf, txs...))))
    ty = map(TensorSplat(TryLinearKw(Eval; inner_kw...)), tc)

    c = has_char2(R) ? coeff : withsign(transpose_signexp(Tensor((tf, txs...))), coeff)

    is_filtered = is_filtered && all(map(Tuple(tf), tc, ty) do f, tx, y
            y isa AbstractLinear || keeps_filtered(f, map(typeof, Tuple(tx)[2:end])...)
    end)

    if any(y -> y isa AbstractLinear, ty)
        tensor(ty...; addto, coeff = c, is_filtered)
    else
        addmul!(addto, Tensor(ty), c; is_filtered)
    end
end

function return_type(tf::AbstractTensor{<:NTuple{N,Any}}, types::Vararg{Type{<:AbstractTensor{<:NTuple{N,Any}}}}) where N
    @foldable
    TC = return_type(transpose, Tensor{Tuple{typeof(tf), types...}})
    TY = map(Tuple(tf), fieldtypes(_termtype(TC))) do f, T
        return_type(f, fieldtypes(T)[2:end]...)
    end
    LU = return_type(tensor, TY...)
    if TC <: Tensor
        LU
    elseif LU <: Tensor
        Linear1{LU,coefftype(TC)}
    else
        change_coefftype(LU, promote_type(_coefftype(TC), _coefftype(LU)))
    end
end

function hastrait(tf::AbstractTensor{<:NTuple{N,Any}}, trait::Val, types::Vararg{Type{<:AbstractTensor{<:NTuple{N,Any}}}}) where N
    return_type(tf, types...) <: AbstractLinear && hastrait(tensor_callable_linear, trait, typeof(tf), types...)
end

function keeps_filtered(tf::AbstractTensor{<:NTuple{N,Any}}, types::Vararg{Type{<:AbstractTensor{<:NTuple{N,Any}}}}) where N
    TT = transpose_nosign_return_type(Tensor{Tuple{types...}})
    all(map((f, T) -> keeps_filtered(f, fieldtypes(T)...), Tuple(tf), fieldtypes(TT)))
end

# differential

function tensor_diff(addto, coeff, x, dx, degx, sizehint, is_filtered)
    isempty(dx) && return addto
    dx1, dx... = dx
    degx1, degx... = degx
    coeff = withsign(degx1, coeff)
    k = length(x)-length(dx)
    if dx1 isa AbstractLinear
        tensor(x[1:k-1]..., dx1, x[k+1:end]...; addto, coeff, sizehint, is_filtered)
    else # in this case `diff` is probably not a differential ...
        addmul!(addto, Tensor((x[1:k-1]..., dx1, x[k+1:end]...)), coeff; is_filtered)
    end
    tensor_diff(addto, coeff, x, dx, degx, sizehint, is_filtered)
end

"""
    diff(t::T) where T <: AbstractTensor -> Linear{T}

Return the differential of the tensor `t` by differentiating each tensor factor at a time
and adding signs according to the degrees of the components. The coefficient type is usually
`DefaultCoefftype`. However, if the degrees of the tensor components are not integers, then
the coefficient type is chosen such that it can accommodate the signs.

See also [`diff`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Example

The degree of a `GradedString` (created with the `gr""` string macro) is its length.
```jldoctest
julia> using $(@__MODULE__).TestHelpers: GradedString, @gr_str

julia> import $(@__MODULE__): diff

julia> diff(x::GradedString) = Linear1(gr"δ"*x => Int(x[1] != 'δ'));

julia> gr"x" |> diff
Linear1{GradedString, Int64} with 1 term:
gr"δx"

julia> gr"x" |> diff |> diff
Linear1{GradedString, Int64} with 0 terms:
0

julia> Tensor(gr"x", gr"yy", gr"zzz") |> diff
Linear{Tensor{Tuple{GradedString, GradedString, GradedString}}, Int64} with 3 terms:
gr"δx"⊗gr"yy"⊗gr"zzz"-gr"x"⊗gr"δyy"⊗gr"zzz"-gr"x"⊗gr"yy"⊗gr"δzzz"
```
"""
@linear_kw function diff(t::T;
        coefftype = Sign,
        addto = zero(linear_return_type(diff, unval(coefftype), T)),
        coeff = ONE,
        is_filtered::Bool = false,
        sizehint::Bool = true) where T <: AbstractTensor

    x = Tuple(t)
    kwc = has_char2(addto) ? (; coefftype) : (;)

    dx = map(x) do y
        Y = typeof(y)
        kwd = has_isfiltered(diff, Y) ? (; is_filtered) : (;)
        if has_coefftype(diff, Y)
            kwd = push_kw(kwd; kwc...)
        end
        diff(y; kwd...)
    end

    is_filtered = is_filtered && all(map(Tuple(t), dx) do x, y
        y isa AbstractLinear || keeps_filtered(diff, typeof(x))
    end)

    if has_char2(addto)
        degx = ntuple(Returns(Zero()), length(x))
    else
        degx = (Zero(), map(deg, x[1:end-1])...)
    end

    tensor_diff(addto, coeff, x, dx, degx, sizehint, is_filtered)
end

function return_type(::typeof(diff), T::Type{<:Tensor})
    @foldable
    DT = map(Fix1(return_type, diff), fieldtypes(T))
    RT = if fieldcount(T) > 1
        map(signtype ∘ Fix1(return_type, deg), fieldtypes(T)[1:end-1])
    else
        (Sign,)  # ensures that `promote_type` for `R` has at least one argument
    end
    U = map(promote_typejoin, map(_termtype, DT), fieldtypes(T))
    R = promote_type(map(_coefftype, DT)..., RT...)
    Linear{Tensor{Tuple{U...}}, R <: Sign ? DefaultCoefftype : R}
end
