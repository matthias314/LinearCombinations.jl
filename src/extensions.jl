#
# linear extension
#

# macros for linear extension

export @linear, @linear_kw, keeps_filtered

using MacroTools

Base.@nospecializeinfer hastrait(f, trait::Val, @nospecialize(types...)) = false

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
has_coefftype(f, types...) = hastrait(f, Val(:coefftype), types...)

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
has_addto_coeff(f, types...) = hastrait(f, Val(:addto_coeff), types...)

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
has_isfiltered(f, types...) = hastrait(f, Val(:is_filtered), types...)

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
has_sizehint(f, types...) = hastrait(f, Val(:sizehint), types...)

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
    push!(ex.args, combinedef(def))
end

"""
    @linear_kw function def

`@linear_kw` scans a function definition for the keywords `coefftype`, `addto`, `coeff`
and `sizehint!` and makes them known to the `$(@__MODULE__)` package. This allows to
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

using LinearCombinations: unval   # unwraps a Val argument

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
x+2*y

julia> f(a; coefftype = Float64, coeff = 2)
4.0*Y-2.0*x-4.0*y+2.0*X

julia> g(a; coefftype = Float64, coeff = 2)
4.0*Y-2.0*x-4.0*y+2.0*X
```
Test whether keywords have been registered:
```jldoctest addto-coeff
julia> using LinearCombinations: has_coefftype, has_addto_coeff, has_sizehint

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
    if isexpr(f, :(::))
        FT = f.args[2]
        traitex = Expr(:block)
    else
        FT = :(typeof($f))
        traitex = Expr(:block, :(function $f end))
    end
    kwnames = map(kw -> splitarg(kw)[1], def[:kwargs])
    args = map(def[:args]) do ex
        name, type, slurp, default = splitarg(ex)
        default === nothing || error("default argument values not supported")
        combinearg(nothing, :(Type{<:$type}), slurp, nothing)
    end
    def[:args] = prepend!(args, (:(::$FT), :(::Val)))
    def[:kwargs] = []
    def[:body] = :true
    traits = Symbol[]
    for t in (:coefftype, :is_filtered, :sizehint)
        t in kwnames && push!(traits, t)
    end
    :addto in kwnames && :coeff in kwnames && push!(traits, :addto_coeff)
    isempty(traits) || addtraits!(traitex, def, traits)
    esc(:($traitex; Base.@__doc__ $ex))
end

linear_extension_coeff_type(f, types...) = _coefftype(return_type(f, types...))

linear_extension_term_type(f, ::Type{T}) where T = _termtype(return_type(f, T))

function linear_extension_type(f, ::Type{L}, ::Type{R}) where {L<:AbstractLinear,R}
    LU = return_type(f, _termtype(L))
    U = _termtype(LU)
    L <: Linear1 && (LU <: Linear1 || !(LU <: AbstractLinear)) ? Linear1{U,R} : Linear{U,R}
end

"""
    @linear f

This macro defines a linear extension of the function (or callable object) `f`.
More specifically, it defines a new method `f(a::AbstractLinear{T,R}; kw...) where {T,R}` that returns
the linear combination obtained by summing up `c*f(x)` for all term-coefficient pairs `x => c`
appearing in `a`.

The new method recognizes the following keyword arguments:

* `coefftype`:
    This optional keyword argument specifies the coefficient type of the linear combination returned
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
x+2*y

julia> f(a)
2*Y+X

julia> f(a; coefftype = Float64)
2.0*Y+X

julia> b = Linear('z' => 3); f(a; addto = b, coeff = -1); b
-2*Y-X+3*z
```

## Linear extension of a function returning a linear combination

```jldoctest linear
julia> g(x) = Linear(x*x => 1.0, string(x) => -1.0); @linear g
g (generic function with 2 methods)

julia> g("x"), g("")
(xx-x, 0)

julia> g(a)   # same a as before
xx-x+2.0*yy-2.0*y

julia> g(a; coefftype = Val(Int), coeff = 3.0)
3*xx-3*x+6*yy-6*y
```

## Linear extension of a callable object

```jldoctest linear
julia> struct P y::String end

julia> (p::P)(x) = p.y*x*p.y; @linear p::P

julia> p = P("w"); p(a)   # same a as before
wxw+2*wyw
```
"""
macro linear(f)
    F = esc(f)
    quote
        function $F(a::L;
                coefftype = promote_type(R, linear_extension_coeff_type($F, T)),
                # addto = zero(Linear{linear_extension_term_type($F, T), coefftype}),
                addto = zero(linear_extension_type($F, L, unval(coefftype))),
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
# linear extension of ComposedFunctionOuterKw
#

@linear f::ComposedFunctionOuterKw

hastrait(f::ComposedFunctionOuterKw, trait::Val, types...) = hastrait(f.outer, trait, types...)

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
6*Y+3*X
```
"""
struct LinearExtension{F}  # <: Function
    f::F
    name::String
end

LinearExtension(f::F, name = "LinearExtension($(repr(f)))") where F = LinearExtension{F}(f, name)

keeps_filtered(g::LinearExtension, T::Type) = keeps_filtered(g.f, T)

# function show(io::IO, ::MIME"text/plain", g::LinearExtension)
function show(io::IO, g::LinearExtension)
    print(io, g.name)
    # show(io, MIME("text/plain"), g.f)
end

@propagate_inbounds (g::LinearExtension)(x; kw...) = g.f(x; kw...)

@linear g::LinearExtension

hastrait(g::LinearExtension, trait::Val, types...) = hastrait(g.f, trait, types...)

deg(g::LinearExtension) = deg(g.f)

# linear extension of function evaluation

(a::AbstractLinear)(x...; kw...) = multilin(Eval, a, x...; kw...)


#
# multilinear extensions
#

export @multilinear

multilin_return_type(f::F, x::A) where {F,A} = return_type(f, map(_termtype, x)...)

function multilin_coeff_type(f::F, x::A) where {F,A<:Tuple}
    R = promote_type(_coefftype(multilin_return_type(f, x)), map(_coefftype, x)...)
    R == Sign ? DefaultCoefftype : R
end

multilin_term_type(f::F, x::A) where {F,A<:Tuple} = _termtype(multilin_return_type(f, x))

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

@generated function multilin(f::F, a...;
        coefftype = multilin_coeff_type(f, a),
        addto = zero(Linear{multilin_term_type(f, a), unval(coefftype)}),
            # TODO: we want coefftype::Type{R} and use "R" here, see julia #49367
        coeff = ONE,
        is_filtered = false,
        sizehint = true,
        kw...) where F
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
    @multilinear f
    @multilinear f f0

This macro defines a multilinear extension of the function `f` (or `f0`). This is analogous to `@linear f`.
The new methods accepts both terms and linear combinations as arguments. It linearly expands all arguments that are
linear combinations and then calls `f` for each combination of terms. If `f0` is specified, then `f0` is called
instead to evaluate terms.

The new method always returns a linear combination (of type `Linear` unless this is overriden by the `addto`
keyword). The term type is inferred from the return type of `f` (or `f0`) with terms as arguments. The coefficient type
is computed by promoting the coefficient types of all `AbstractLinear` arguments. In case `f` (or `f0`) returns
a linear combination for term arguments, that coefficient type is also taken into account.

In order to catch all possible combinations of terms and linear combinations, `@multilinear f` and
`@multilinear f f0` define a single new method `f(x...; kw...)` that matches **all** argument types.
(This is different from `@linear`.) Hence, if `f0` is not given, then the methods for `f` that evaluate
terms must have a non-generic signature. If instead the signature also is `f(x::Any...)`, then this
method is overwritten, resulting in an error when `f` is called.

The new method defined by `@multilinear` accepts all keyword arguments discussed for `@linear`. Unknown
keyword arguments are passed on to the call for term evaluation. The macro `@linear_kw` works as for
linear functions.

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
(x+2*y, -w+z)

julia> f(a, "z")
2*yz+xz

julia> f('x', b)
-xw+xz

julia> f(a, b)
-xw+2.0*yz-2.0*yw+xz
```

## Bilinear extension of a function returning a linear combination

```jldoctest multilinear
julia> f(x::Char, y::String) = Linear(x*y => BigInt(1), y*x => BigInt(-1)); @multilinear f

julia> f(a, b)   # same a and b as before
-2.0*zy-xw-zx+2.0*yz+wx-2.0*yw+xz+2.0*wy

julia> typeof(ans)
Linear{String, BigFloat}
```

## Multilinear extension of a function

```jldoctest multilinear
julia> g(xs::Union{Char,String}...) = *(xs...); @multilinear g

julia> g(a)   # same a and b as before
x+2*y

julia> g(a, b)
-xw+2.0*yz-2.0*yw+xz

julia> g(a, b, a)
-xwx+xzx+4.0*yzy+2.0*xzy+2.0*yzx-2.0*ywx-2.0*xwy-4.0*ywy
```

## Multilinear extension using the two-argument version of `@multilinear`

```jldoctest multilinear
julia> @multilinear(h, *)

julia> h(a, b; coeff = 2)   # same a and b as before
-2.0*xw+4.0*yz-4.0*yw+2.0*xz
```
"""
macro multilinear(f, f0 = f)
    F = esc(f)
    F0 = esc(f0)
    if f0 == f
        traits = quote end
    else
        FT = isexpr(f, :(::)) ? esc(f.args[2]) : :(typeof($F))
        traits = quote
            $(@__MODULE__).hastrait(::$FT, ::Val, types...) = true
        end
    end
    # TODO: does @propagate_inbounds make sense?
    quote
        @propagate_inbounds $F(x...; kw...) = multilin($F0, x...; kw...)
        $traits
    end
end

macro multilinear_noesc(f, f0 = f)
    F = esc(f)
    F0 = f0  # esc(f0)
    if f0 == f
        traits = quote end
    else
        FT = isexpr(f, :(::)) ? esc(f.args[2]) : :(typeof($F))
        traits = quote
            $(@__MODULE__).hastrait(::$FT, ::Val, types...) = true
        end
    end
    quote
        # TODO: does @propagate_inbounds make sense?
        @propagate_inbounds @generated function $F($(esc(:a))...;
                coefftype = multilin_coeff_type($F0, a),
                addto = zero(Linear{multilin_term_type($F0, a), unval(coefftype)}),
                    # TODO: we want coefftype::Type{R} and use "R" here, see julia #49367
                coeff = ONE,
                is_filtered = false,
                sizehint = true,
                $(esc(:kw))...)
            N = length(a)
            TT = map(_termtype, a)
            quote
                # TT = $TT
                is_filtered || all($linear_filter, a) || return addto
                has_ac = $has_addto_coeff($$F0, $TT...)
                fkw = kw
                if $has_isfiltered($$F0, $TT...)
                    fkw = push_kw(fkw; is_filtered)
                end
                if $has_sizehint($$F0, $TT...)
                    fkw = push_kw(fkw; sizehint)
                elseif sizehint # && !(return_type(f, $TT...) <: AbstractLinear)
                    l = prod($_length, a; init = 1)
                    sizehint!(addto, length(addto)+l)
                end
                Base.Cartesian.@nexprs(1, i -> cc_{$N+i} = coeff)  # initialize cc_{N+1}
                Base.Cartesian.@nloops($N, xc, i -> a[i] isa $AbstractLinear ? a[i] : ((a[i], $ONE),), i -> begin
                    x_i, c_i = xc_i
                    cc_i = c_i*cc_{i+1}
                end, begin
                    if has_ac   # || return_type(f, $TT...) <: AbstractLinear   # for testing
                        $$(@__MODULE__).@ncallkw($N, $$F0, (addto, coeff = cc_1, fkw...), x)
                    else
                        $addmul!(addto, $$(@__MODULE__).@ncallkw($N, $$F0, fkw, x), cc_1; is_filtered = $keeps_filtered($$F0, $TT...))
                    end
                end)
                addto
            end
        end

        $traits
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
(x+2*y, -w+z)

julia> const concat = MultilinearExtension(*, "concat")
concat

julia> concat(a, b)
-xw+2.0*yz-2.0*yw+xz
```
"""
MultilinearExtension(f::F, name = "MultilinearExtension($(repr(f)))") where F = MultilinearExtension{F}(f, name)

keeps_filtered(g::MultilinearExtension, T::Type) = keeps_filtered(g.f, T)

show(io::IO, g::MultilinearExtension) = print(io, g.name)

@multilinear g::MultilinearExtension g.f

# hastrait(g::MultilinearExtension, trait::Val, types...) = hastrait(g.f, trait, types...)

deg(g::MultilinearExtension) = deg(g.f)

#
# bilinear and multilinear extension of multiplication
#

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

The coproduct (or comultiplication) of `x`. The argument `x` is assumed to be an element of a coalgebra.

The module $(@__MODULE__) only defines the linear extension of `coprod`, but no methods for terms.
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
"""
function diff end

@linear diff

deg(::typeof(diff)) = -1
