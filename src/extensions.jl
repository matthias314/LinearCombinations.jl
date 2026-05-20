#
# linear extension
#

is_term_or_linear1(::Type{T}) where T = !(T <: AbstractLinear) || T <: Linear1

function return_type(f::F, types::Type...) where F
    TT = map(_termtype, types)
    if TT == types
        ReturnType.return_type(f, types...)
    else
        LU = return_type(f, TT...)
        U = _termtype(LU)
        R = promote_type(map(_coefftype, types)..., _coefftype(LU))
        if all(is_term_or_linear1, types) && is_term_or_linear1(LU)
            Linear1{U, R == Sign ? DefaultCoefftype : R}
        else
            Linear{U, R == Sign ? DefaultCoefftype : R}
        end
    end
end

function linear_return_type(f::F, ::Type{R}, types::Type...) where {F,R}
    L = return_type(f, types...)
    @assert L <: AbstractLinear
    S = promote_type(coefftype(L), R)
    change_coefftype(L, S)
end

# macros for linear extension

export @linear, @linear_kw, keeps_filtered

using MacroTools

Base.@nospecializeinfer hastrait(f, trait::Val, @nospecialize(types::Type...)) = false

"""
    $(@__MODULE__).has_coefftype(f, types...) -> Bool

Return `true` if the method for `f` with signature given by `types` is known
to support the keyword argument `coefftype`. The macro `@linear_kw` is used to make
this keyword known to the $(@__MODULE__) package.

See also
[`@linear_kw`](@ref),
[`$(@__MODULE__).has_addto_coeff`](@ref),
[`$(@__MODULE__).has_isfiltered`](@ref),
[`$(@__MODULE__).has_sizehint`](@ref).
"""
has_coefftype(f, types::Type...) = hastrait(f, Val(:coefftype), types...)

"""
    $(@__MODULE__).has_addto_coeff(f, types...) -> Bool

Return `true` if the method for `f` with signature given by `types` is known
to support the keyword arguments `addto` and `coeff`. The macro `@linear_kw` is used to make
these keywords known to the $(@__MODULE__) package.

See also
[`@linear_kw`](@ref),
[`$(@__MODULE__).has_coefftype`](@ref),
[`$(@__MODULE__).has_isfiltered`](@ref),
[`$(@__MODULE__).has_sizehint`](@ref).
"""
has_addto_coeff(f, types::Type...) = hastrait(f, Val(:addto_coeff), types...)

"""
    $(@__MODULE__).has_isfiltered(f, types...) -> Bool

Return `true` if the method for `f` with signature given by `types` is known
to support the keyword argument `is_filtered::Bool`. The macro `@linear_kw` is used to make
this keyword known to the $(@__MODULE__) package.

The keyword argument `is_filtered = true` for a linear or multilinear function `f` indicates
this potentially expensive test can be skipped when evaluating `f`.

See also
[`@linear_kw`](@ref),
[`$(@__MODULE__).has_coefftype`](@ref),
[`$(@__MODULE__).has_addto_coeff`](@ref),
[`$(@__MODULE__).has_sizehint`](@ref),
[`keeps_filtered`](@ref).
"""
has_isfiltered(f, types::Type...) = hastrait(f, Val(:is_filtered), types...)

"""
    $(@__MODULE__).has_sizehint(f, types...) -> Bool

Return `true` if the method for `f` with signature given by `types` is known
to support the keyword argument `sizehint`. The macro `@linear_kw` is used to make
this keyword known to the $(@__MODULE__) package.

See also
[`@linear_kw`](@ref),
[`$(@__MODULE__).has_coefftype`](@ref),
[`$(@__MODULE__).has_addto_coeff`](@ref),
[`$(@__MODULE__).has_isfiltered`](@ref),
"""
has_sizehint(f, types::Type...) = hastrait(f, Val(:sizehint), types...)

"""
    keeps_filtered(f, types...) -> Bool

Return `true` if the following is satisfied, and `false` otherwise: Whenever the function `f` is
called with arguments of types `types` and returns a single term `y`, then `linear_filter(y) == true` holds.

By default, `keeps_filtered` returns `false` for all arguments. This can be changed to avoid unneccesary
(and possibly expensive) calls to `linear_filter`. Note that if `f` returns a linear combination when called
with term arguments, then all terms appearing in this linear combination satisfy the condition above anyway.
The setting for `keeps_filtered` doesn't matter in this case.

See also [`$(@__MODULE__).linear_filter`](@ref).
"""
keeps_filtered(f, ::Type...) = false
keeps_filtered(::typeof(identity), ::Type) = true

function addtraits!(ex, def::Dict, traits)
    def[:name] = :($(@__MODULE__).hastrait)
    tunion = Expr(:curly, :Union, (Expr(:curly, :Val, QuoteNode(t)) for t in traits)...)
    def[:args][2] = Expr(:(::), tunion)
    push!(ex.args, esc(combinedef(def)))
    ex
end

"""
    @linear_kw function def

`@linear_kw` scans a function definition for the keywords `coefftype`, `addto`, `coeff`
and `sizehint` and makes them known to the `$(@__MODULE__)` package. This allows to
write performant code. Not all keywords have to present. However, `addto` and `coeff`
only have an effect if used together.

See also
[`$(@__MODULE__).has_coefftype`](@ref),
[`$(@__MODULE__).has_addto_coeff`](@ref),
[`$(@__MODULE__).has_isfiltered`](@ref),
[`$(@__MODULE__).has_sizehint`](@ref),
[`$(@__MODULE__).unval`](@ref).

# Example

Consider the following two functions:
```jldoctest addto-coeff; output = false
f(x::Char) = Linear(uppercase(x) => 1, x => -1)

@linear f

using $(@__MODULE__): unval   # unwraps a Val argument

@linear_kw function g(x::Char;
        coefftype = Int,
        addto = zero(Linear{Char,unval(coefftype)}),
        coeff = 1)
    addmul!(addto, uppercase(x), coeff)
    addmul!(addto, x, -coeff)
    addto
end

@linear g

# output

g (generic function with 2 methods)
```
The linear extensions are functionally equivalent,  but `g` will be much faster than `f`.
```jldoctest addto-coeff
julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> f(a; coefftype = Float64, coeff = 2)
Linear{Char, Float64} with 4 terms:
4.0*'Y'-2.0*'x'-4.0*'y'+2.0*'X'

julia> g(a; coefftype = Float64, coeff = 2)
Linear{Char, Float64} with 4 terms:
4.0*'Y'-2.0*'x'-4.0*'y'+2.0*'X'
```
Test whether keywords have been registered:
```jldoctest addto-coeff
julia> using $(@__MODULE__): has_coefftype, has_addto_coeff, has_sizehint

julia> has_coefftype(g, Char), has_addto_coeff(g, Char), has_sizehint(g, Char)
(true, true, false)
```
"""
macro linear_kw(ex)
    # skip macro calls
    ex1 = ex
    while Meta.isexpr(ex1, :macrocall)
        ex1 = ex1.args[end]
    end

    def = splitdef(ex1)
    f = def[:name]
    FT = isexpr(f, :(::)) ? f.args[end] : :(typeof($f))
    kwnames = map(kw -> splitarg(kw)[1], def[:kwargs])
    args = map(def[:args]) do ex
        name, type, slurp, default = splitarg(ex)
        default === nothing || error("default argument values not supported")
        combinearg(nothing, :(Type{<:$type}), slurp, nothing)
    end
    def[:args] = prepend!(args::Vector, (:(::$FT), :(::Val)))   # "::Vector" for JET analysis
    def[:kwargs] = []

    traits = Symbol[]
    notraits = Symbol[]
    for t in (:coefftype, :is_filtered, :sizehint)
        push!(t in kwnames ? traits : notraits, t)
    end
    push!(:addto in kwnames && :coeff in kwnames ? traits : notraits, :addto_coeff)

    ex2 = Expr(:block, :(f = Core.@__doc__ $(esc(ex))))
    def[:body] = :true
    isempty(traits) || addtraits!(ex2, def, traits)
    def[:body] = :false
    isempty(notraits) || addtraits!(ex2, def, notraits)
    push!(ex2.args, :f)
    ex2
end

"""
    @linear f
    @linear ::F
    @linear f::F

This macro defines a linear extension of the function `f` (or a callable object of type `F`).
More specifically, it defines a new method `f(a::AbstractLinear{T,R}; kw...) where {T,R}` that returns
the linear combination obtained by summing up `c*f(x)` for all term-coefficient pairs `x => c`
appearing in `a`.

The new method recognizes the following keyword arguments:

* `coefftype`:
    This optional keyword argument influences the coefficient type of the linear combination returned
    by `f(a)` if the keyword argument `addto` is not present. If `coefftype` is also not specified
    and `f(x::T)` is a term (as opposed to a linear combination), then `coefftype` is set to `R`.
    If `f(x::T) <: AbstractLinear`, say with coefficient type `S`, then `promote_type(R, S)`
    is chosen as the new coefficient type. If the `addto` keyword is present, then `coefftype` is ignored.

    Because of the way Julia handles keyword arguments, the form `f(a; coefftype = Int)` is not type-stable.
    Type stability can be achieved by saying `f(a; coefftype = Val(Int))`.

* `addto::AbstractLinear`:
    If given, the sum of all terms `c*f(x)` is added to `addto`, and the result is returned.
    This avoids allocating a new linear combination each time `f` is called with an `AbstractLinear` argument.
    The default value for `addto` is `Linear{U,coefftype}`. Here `U` is the return type of `f(x::T)`
    if this return type is not a subtype of `AbstractLinear` and the term type of the return values otherwise.

* `coeff`:
    This optional keyword argument allows to efficiently compute scalar multiples of `f(a)`. More precisely,
    `f(a; coeff = c)` returns `c*f(a)`, and `f(a; addto = b, coeff = c)` adds `c*f(a)` to `b` and returns
    this new value.

* `sizehint::Bool = true`:
    The new method for `f` may call `sizehint!` for `addto` to pre-allocate room for the new terms.
    This keyword argument permits to turn pre-allocation off.

All other keyword arguments are passed on to `f(x)`. With the macro `@linear_kw` one can make `f(a)` pass
the special keyword arguments listed above on to `f(x)`, too.

See also [`@multilinear`](@ref), [`sizehint!`](@ref), [`@linear_kw`](@ref), [`keeps_filtered`](@ref).

# Examples

## Linear extension of a function returning a term

```jldoctest linear
julia> f(x) = uppercase(x); @linear f
f (generic function with 2 methods)

julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> f(a)
Linear{Char, Int64} with 2 terms:
2*'Y'+'X'

julia> f(a; coefftype = Float64)
Linear{Char, Float64} with 2 terms:
2.0*'Y'+'X'

julia> b = Linear('z' => 3); f(a; addto = b, coeff = -1); b
Linear{Char, Int64} with 3 terms:
-2*'Y'-'X'+3*'z'
```

## Linear extension of a function returning a linear combination

```jldoctest linear
julia> g(x) = Linear(x*x => 1.0, string(x) => -1.0); @linear g
g (generic function with 2 methods)

julia> g("x"), g("")
(Linear{String, Float64}("xx" => 1.0, "x" => -1.0), Linear{String, Float64}())

julia> g(a)   # same a as before
Linear{String, Float64} with 4 terms:
"xx"-"x"+2.0*"yy"-2.0*"y"

julia> g(a; coefftype = Val(Int), coeff = 3.0)
Linear{String, Int64} with 4 terms:
3*"xx"-3*"x"+6*"yy"-6*"y"
```

## Linear extension of a callable object

```jldoctest linear
julia> struct P y::String end

julia> (p::P)(x) = p.y*x*p.y; @linear ::P  # or `@linear p::P`

julia> p = P("w"); p(a)   # same a as before
Linear{String, Int64} with 2 terms:
"wxw"+2*"wyw"
```
"""
macro linear(f)
    F = Meta.isexpr(f, :(::), 1) ? Expr(:(::), :f, esc(f.args[1])) : esc(f)
    FT = Meta.isexpr(f, :(::)) ? F : :(::typeof($(esc(f))))
    quote
        function $F end

        $(@__MODULE__).hastrait($FT, ::Val{trait}, ::Type{<:Linear}) where trait = trait != :is_filtered  # TODO: use @linear_kw

        function $F(a::L;
                coefftype = Sign,
                addto = zero(linear_return_type($F, unval(coefftype), L)),
                coeff = ONE,
                sizehint::Bool = true,
                kw...) where {T,R,L<:AbstractLinear{T,R}}
            if iszero(coeff)
                ;
            elseif return_type($F, T) <: AbstractLinear
                has_ac = has_addto_coeff($F, T)
                fkw = kw
                if has_isfiltered($F, T)
                    fkw = push_kw(fkw; is_filtered = true)
                end
                if has_sizehint($F, T)
                    fkw = push_kw(fkw; sizehint)
                end
                for (x, c) in a
                    if has_ac
                        $F(x; addto, coeff = coeff*c, fkw...)
                    else
                        addmul!(addto, $F(x; fkw...), coeff*c)
                    end
                end
            else
                sizehint && sizehint!(addto, length(addto)+length(a))
                for (x, c) in a
                    addmul!(addto, $F(x; kw...), coeff*c; is_filtered = keeps_filtered($F, T))
                end
            end
            addto
        end
    end
end

#
# has_char2 including linear types, addto and coefftype
#

has_char2(::Type{L}) where {T,R,L<:AbstractLinear{T,R}} = has_char2(R)
has_char2(::Type{Union{}}) = error("not defined")   # for JET analysis

has_char2(::Val{R}) where R <: Type = has_char2(R)

Base.@assume_effects :total function has_char2(types::Type...; kw...)
    R = _coefftype(get(kw, :addto, missing))
    R !== missing && return has_char2(R)
    R = get(kw, :coefftype, missing)
    R !== missing && return has_char2(R)
    any(has_char2, types)
end

#
# new type for linear extension
#

export LinearExtension

"""
    LinearExtension{F}

This type is the linear extension of the given type `F`.

# Examples

```jldoctest
julia> const g = LinearExtension(uppercase)
LinearExtension(uppercase)

julia> g('x')
'X': ASCII/Unicode U+0058 (category Lu: Letter, uppercase)

julia> a = Linear('x' => 1, 'y' => 2); g(a; coeff = 3)
Linear{Char, Int64} with 2 terms:
6*'Y'+3*'X'
```
"""
struct LinearExtension{F}  # <: Function
    f::F
    name::String
end

LinearExtension(f::F, name = "LinearExtension($(repr(f)))") where F = LinearExtension{F}(f, name)

keeps_filtered(g::LinearExtension, T::Type) = keeps_filtered(g.f, T)

show(io::IO, g::LinearExtension) = print(io, g.name)

@propagate_inbounds (g::LinearExtension)(x; kw...) = g.f(x; kw...)

@linear g::LinearExtension

hastrait(g::LinearExtension, trait::Val, types::Type...) = hastrait(g.f, trait, types...)

deg(g::LinearExtension) = deg(g.f)

# linear extension of function evaluation

(a::AbstractLinear)(x...; kw...) = MultilinearExtension(Eval)(a, x...; kw...)


#
# multilinear extensions
#

export @multilinear

using Base.Cartesian
using Base.Cartesian: inlineanonymous

macro ncallkw(N::Int, f, kw, args...)
#=
    pre = args[1:end-1]
    ex = args[end]
    vars = (inlineanonymous(ex, i) for i = 1:N)
    param = Expr(:parameters, Expr(:(...), kw))
    Expr(:escape, Expr(:call, f, param, pre..., vars...))
=#
    esc(quote
        if isempty($kw)
            Base.Cartesian.@ncall($N, $f, $(args...))
        else
            Base.Cartesian.@ncall($N, Core.kwcall, $kw, $f, $(args...))
        end
    end)
end

_length(x) = 1
_length(a::AbstractLinear) = length(a)

@generated function multilin(f::F, addto, a::Vararg{Any,N};
        coeff = ONE,
        is_filtered::Bool = false,
        sizehint::Bool = true,
        kw...) where {F,N}
    N = length(a)
    TS = map(_termtype, a)
    quote
        is_filtered || all(linear_filter, a) || return addto
        has_ac = has_addto_coeff(f, $TS...)
        fkw = kw
        if has_isfiltered(f, $TS...)
            fkw = push_kw(fkw; is_filtered = true)
        end
        if has_sizehint(f, $TS...)
            fkw = push_kw(fkw; sizehint)
        elseif sizehint # && !(return_type(f, $TS...) <: AbstractLinear)
            l = prod(_length, a; init = 1)
            sizehint!(addto, length(addto)+l)
        end
        @nexprs(1, i -> cc_{$N+i} = coeff)  # initialize cc_{N+1}
        @nloops($N, xc, i -> a[i] isa AbstractLinear ? a[i] : ((a[i], ONE),), i -> begin
            x_i, c_i = xc_i
            cc_i = c_i*cc_{i+1}
        end, begin
            if has_ac # || return_type(f, $TS...) <: AbstractLinear
                # has_ac || println("$f: ", $TS)
                @ncallkw($N, f, (addto, coeff = cc_1, fkw...), x)
            else
                addmul!(addto, @ncallkw($N, f, fkw, x), cc_1; is_filtered = keeps_filtered(f, $TS...))
            end
        end)
        addto
    end
end

"""
    @multilinear f [f0]
    @multilinear ::F [f0]
    @multilinear f::F [f0]

This macro defines a multilinear extension of the function `f` (or the callable object of type `F`). This is analogous to `@linear f`.
The new methods accepts both terms and linear combinations as arguments. It linearly expands all arguments that are
linear combinations and then calls `f` for each combination of terms. If `f0` is specified, then `f0` is called
instead to evaluate terms. In other words, `f` is the multilinear extension of `f0` in this case.

The new method defined by `@multilinear` accepts all keyword arguments discussed for `@linear`. Unknown
keyword arguments are passed on to the call for term evaluation. The macro `@linear_kw` works as for
linear functions.

The new method always returns a linear combination (of type `Linear` unless this is overriden by the `addto`
keyword). The term type is inferred from the return type of `f` (or `f0`) with terms as arguments. The coefficient type
is computed by promoting the coefficient types of all `AbstractLinear` arguments. In case `f` (or `f0`) returns
a linear combination for term arguments, that coefficient type is also taken into account.

In order to catch all possible combinations of terms and linear combinations, `@multilinear f` and
`@multilinear f f0` define a single new method `f(x...; kw...)` that matches **all** argument types.
(This is different from `@linear`.) Hence, if `f0` is not given, then the methods for `f` that evaluate
terms must have a non-generic signature. If instead the signature also is `f(x::Any...)`, then this
method is overwritten, resulting in an error when `f` is called.

If the two-argument version of `@multilinear` is used, then typically there is no other method for `f`.
Hence `f` returns a linear combination for all arguments in this case. If all arguments are terms and also `f0`
returns a term, then the coefficient type is `$(@__MODULE__).DefaultCoefftype`. For the one-argument version there
must be at least one other method as discussed above. So `f` may not return a linear combination for all arguments.

See also [`@linear`](@ref), [`@linear_kw`](@ref), [`$(@__MODULE__).DefaultCoefftype`](@ref).

# Examples

## Bilinear extension of a function returning a term

```jldoctest multilinear
julia> f(x::Char, y::String) = x*y; @multilinear f

julia> a, b = Linear('x' => 1, 'y' => 2), Linear("z" => 1.0, "w" => -1.0)
(Linear{Char, Int64}('x' => 1, 'y' => 2), Linear{String, Float64}("w" => -1.0, "z" => 1.0))

julia> f(a, "z")
Linear{String, Int64} with 2 terms:
2*"yz"+"xz"

julia> f('x', b)
Linear{String, Float64} with 2 terms:
-"xw"+"xz"

julia> f(a, b)
Linear{String, Float64} with 4 terms:
-"xw"+2.0*"yz"-2.0*"yw"+"xz"
```

## Bilinear extension of a function returning a linear combination

```jldoctest multilinear
julia> f(x::Char, y::String) = Linear(x*y => BigInt(1), y*x => BigInt(-1)); @multilinear f

julia> f(a, b)   # same a and b as before
Linear{String, BigFloat} with 8 terms:
-2.0*"zy"-"xw"-"zx"+2.0*"yz"+"wx"-2.0*"yw"+"xz"+2.0*"wy"

julia> typeof(ans)
Linear{String, BigFloat}
```

## Multilinear extension of a function

```jldoctest multilinear
julia> g(xs::Union{Char,String}...) = *(xs...); @multilinear g

julia> g(a)   # same a and b as before
Linear{String, Int64} with 2 terms:
"x"+2*"y"

julia> g(a, b)
Linear{String, Float64} with 4 terms:
-"xw"+2.0*"yz"-2.0*"yw"+"xz"

julia> g(a, b, a)
Linear{String, Float64} with 8 terms:
-"xwx"+"xzx"+4.0*"yzy"+2.0*"xzy"+2.0*"yzx"-2.0*"ywx"-2.0*"xwy"-4.0*"ywy"
```

## Multilinear extension using the two-argument version of `@multilinear`

```jldoctest multilinear
julia> @multilinear(h, *)

julia> h(a, b; coeff = 2)   # same a and b as before
Linear{String, Float64} with 4 terms:
-2.0*"xw"+4.0*"yz"-4.0*"yw"+2.0*"xz"
```
"""
macro multilinear(f, f0 = f)
    F = Meta.isexpr(f, :(::), 1) ? Expr(:(::), :f, esc(f.args[1])) : esc(f)
    FT = Meta.isexpr(f, :(::)) ? F : :(f::typeof($F))
    F0 = f0 == f ? F : esc(f0)

    if f0 == f
        traits = quote end
    else
        traits = quote
            $(@__MODULE__).hastrait($FT, ::Val, types::Type...) = true
            $(@__MODULE__).keeps_filtered($FT, types::Type...) = keeps_filtered($F0, types...)
        end
    end

    rt_ex = if f != f0
        quote
            function $(@__MODULE__).return_type($FT, types::Type...)
                if any(T -> T <: AbstractLinear, types)
                    invoke(return_type, Tuple{Any,Vararg{Type}}, $F0, types...)
                else
                    LU = return_type($F0, types...)
                    LU <: AbstractLinear ? LU : Linear1{LU,DefaultCoefftype}
                end
            end
        end
    else
        :()
    end

    # TODO: does @propagate_inbounds make sense?
    quote
        function $F end

        $rt_ex

        $traits

        @propagate_inbounds function $F(xs...;
                coefftype = Sign,
                addto = zero(linear_return_type($F, unval(coefftype), map(typeof, xs)...)),
                kw...)
            multilin($F0, addto, xs...; kw...)
        end
    end
end

# new type for multilinear extension

export MultilinearExtension

struct MultilinearExtension{F}
    f::F
    name::String
end

"""
    MultilinearExtension(f)
    MultilinearExtension(f, name)

An element of this type is a multilinear extension of `f`. One can additionally specify the name displayed for it.

# Example

```jldoctest
julia> a, b = Linear('x' => 1, 'y' => 2), Linear("z" => 1.0, "w" => -1.0)
(Linear{Char, Int64}('x' => 1, 'y' => 2), Linear{String, Float64}("w" => -1.0, "z" => 1.0))

julia> const concat = MultilinearExtension(*, "concat")
concat

julia> concat(a, b)
Linear{String, Float64} with 4 terms:
-"xw"+2.0*"yz"-2.0*"yw"+"xz"
```
"""
MultilinearExtension(f::F, name = "MultilinearExtension($(repr(f)))") where F = MultilinearExtension{F}(f, name)

keeps_filtered(g::MultilinearExtension, T::Type) = keeps_filtered(g.f, T)

show(io::IO, g::MultilinearExtension) = print(io, g.name)

@multilinear g::MultilinearExtension g.f

deg(g::MultilinearExtension) = deg(g.f)

#
# composition of linear functions
#

abstract type AbstractComposedFunction end

keeps_filtered(f::AbstractComposedFunction, types::Type...) = keeps_filtered(f.outer, return_type(f.inner, types...))

deg(f::AbstractComposedFunction) = deg(f.outer) + deg(f.inner)

return_type(f::AbstractComposedFunction, types::Type...) = return_type(f.outer, return_type(f.inner, types...))

struct LinearComposedFunction{O,I} <: AbstractComposedFunction
    outer::O
    inner::I
end

function (f::LinearComposedFunction)(xs::Vararg{Any,M}; is_filtered = false, kw...) where M
    TryLinearKw(f.inner)(xs...; is_filtered) |>
        TryLinearKw(f.outer; is_filtered = is_filtered && keeps_filtered(f.inner, map(typeof, xs)...), kw...)
end

hastrait(f::LinearComposedFunction, trait::Val, types::Type...) = hastrait(f.outer, trait, return_type(f.inner, types...))
hastrait(f::LinearComposedFunction, trait::Val{:is_filtered}, types::Type...) = hastrait(f.inner, trait, types...)

struct TermComposedFunction{O,I} <: AbstractComposedFunction
    outer::O
    inner::I
end

@multilinear f::TermComposedFunction LinearComposedFunction(f.outer, f.inner)

#
# TryLinearKw
#

struct TryLinearKw{F,KW}
    f::F
    kw::KW
end

TryLinearKw(f; kw...) = TryLinearKw{Typeof(f), typeof(kw)}(f, kw)

@struct_equal_hash TryLinearKw

deg(f::TryLinearKw) = deg(f.f)

function (f::TryLinearKw)(x...; kw...)
    kw = (; f.kw..., kw...)
    TT = map(typeof, x)
    kw = has_coefftype(f.f, TT...) ? kw : Base.delete(kw, :coefftype)
    kw = has_isfiltered(f.f, TT...) ? kw : Base.delete(kw, :is_filtered)
    kw = has_sizehint(f.f, TT...) ? kw : Base.delete(kw, :sizehint)
    f.f(x...; kw...)
end

hastrait(::TryLinearKw, ::Union{Val{:coefftype}, Val{:is_filtered}, Val{:sizehint}}, ::Type...) = true
hastrait(f::TryLinearKw, trait::Val, types::Type...) = hastrait(f.f, trait, types...)

keeps_filtered(f::TryLinearKw, types::Type...) = keeps_filtered(f.f, types...)

return_type(f::TryLinearKw, types::Type...) = return_type(f.f, types...)

#
# bilinear and multilinear extension of multiplication
#

"""
    $(@__MODULE__).mul(x1::Any, x2::Any, ...)

Return the product of the arguments. This is the multilinear extension of `*`.

The new name avoids type piracy since `mul` accepts all argument types.
"""
const mul = MultilinearExtension(*)

function isone(a::AbstractLinear{T}) where T
    length(a) == 1 || return false
    x, c = first(a)
    isone(c) && isone(x)
end

one(::Type{L}) where {T,R,L<:AbstractLinear{T,R}} = L(one(T) => one(R))
one(::T) where T <: AbstractLinear = one(T)

*(x::AbstractLinear{T}, y::T; kw...) where T = mul(x, y; kw...)
*(x::T, y::AbstractLinear{T}; kw...) where T = mul(x, y; kw...)
*(x::AbstractLinear...; kw...) = mul(x...; kw...)

function ^(a::AbstractLinear, n::Integer)
    if n > 0
        # TODO: use square and multiply?
        b = a
        for k in 2:n
            b *= a
        end
        b
    elseif n == 0
        one(a)
    else
        error("negative powers are not supported for type ", typeof(a))
    end
end

function promote_rule(::Type{Linear{T,R}}, ::Type{S}) where {T,R,S}
    W = promote_type(R,S)
    Linear{T,W}
end

promote_rule(::Type{Linear{T,R}}, ::Type{T}) where {T,R} = Linear{T,R}

function promote_rule(::Type{Linear{T,R}}, ::Type{Linear{U,S}}) where {T,R,U,S}
    V = promote_type(T,U)
    W = promote_type(R,S)
    Linear{V,W}
end

#
# coproduct
#

export coprod

"""
    coprod(x)

The coproduct (or comultiplication) of `x`, which is assumed to be an element of a coalgebra.

The module $(@__MODULE__) only defines the linear extension of `coprod`, but no methods for terms,
except for tensors.

See also [`coprod(t::AbstractTensor)`](@ref).

# Example

```jldoctest
julia> import LinearCombinations: coprod

julia> coprod(s::String) = Linear(Tensor(s[1:k], s[k+1:end]) => 1 for k in 0:length(s));

julia> s = "ab";

julia> coprod(s)
Linear{Tensor{Tuple{String, String}}, Int64} with 3 terms:
""⊗"ab"+"a"⊗"b"+"ab"⊗""

julia> p = s |> coprod |> Tensor(coprod, identity) |> flatten
Linear{Tensor{Tuple{String, String, String}}, Int64} with 6 terms:
""⊗"ab"⊗""+""⊗"a"⊗"b"+"a"⊗"b"⊗""+""⊗""⊗"ab"+"a"⊗""⊗"b"+"ab"⊗""⊗""

julia> q = s |> coprod |> Tensor(identity, coprod) |> flatten
Linear{Tensor{Tuple{String, String, String}}, Int64} with 6 terms:
""⊗"ab"⊗""+""⊗"a"⊗"b"+"a"⊗"b"⊗""+""⊗""⊗"ab"+"a"⊗""⊗"b"+"ab"⊗""⊗""

julia> p == q  # coproduct is coassociative
true
```
"""
function coprod end

@linear coprod

#
# differential
#

export diff

"""
    diff(x)

The differential of `x`.

The module $(@__MODULE__) only defines the linear extension of `diff`, but no methods for terms.
The only exception is `diff` for tensors.

See also [`diff(t::AbstractTensor)`](@ref).
"""
function diff end

@linear diff

deg(::typeof(diff)) = -1
