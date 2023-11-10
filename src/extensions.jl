#
# linear extension
#

# macros for linear extension

export @linear, @linear_kw, keeps_filtered

using MacroTools

Base.@nospecializeinfer hastrait(f, trait::Val, @nospecialize(types...)) = false

has_coefftype(f, types...) = hastrait(f, Val(:coefftype), types...)
has_addto_coeff(f, types...) = hastrait(f, Val(:addto_coeff), types...)
has_isfiltered(f, types...) = hastrait(f, Val(:isfiltered), types...)
has_sizehint(f, types...) = hastrait(f, Val(:sizehint), types...)

keeps_filtered(f, ::Type...) = false
keeps_filtered(::typeof(identity), ::Type) = true

function addtraits!(ex, def::Dict, traits)
    def[:name] = :($(@__MODULE__).hastrait)
    tunion = Expr(:curly, :Union, (Expr(:curly, :Val, QuoteNode(t)) for t in traits)...)
    def[:args][2] = Expr(:(::), tunion)
    push!(ex.args, combinedef(def))
end

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
    esc(:($traitex; $ex))
end

linear_extension_coeff_type(f, types...) = _coefftype(return_type(f, types...))

linear_extension_term_type(f, ::Type{T}) where T = _termtype(return_type(f, T))

function linear_extension_type(f, ::Type{L}, ::Type{R}) where {L<:AbstractLinear,R}
    LU = return_type(f, _termtype(L))
    U = _termtype(LU)
    L <: Linear1 && (LU <: Linear1 || !(LU <: AbstractLinear)) ? Linear1{U,R} : Linear{U,R}
end

macro linear(f)
    F = esc(f)
    quote
        function $F(a::L;
                coefftype = promote_type(R, linear_extension_coeff_type($F, T)),
                # addto = zero(Linear{linear_extension_term_type($F, T), coefftype}),
                addto = zero(linear_extension_type($F, L, coefftype)),
                coeff = ONE,
                sizehint = true,
                kw...) where {T,R,L<:AbstractLinear{T,R}}
            if iszero(coeff)
                ;
            elseif return_type($F, T) <: AbstractLinear
                has_ac = has_addto_coeff($F, T)
                # has_ac || println($F, ": ", T)
                new_kw = kw
                if has_isfiltered($F, T)
                    new_kw = (; is_filtered = true, new_kw...)
                end
                if has_sizehint($F, T)
                    new_kw = (; sizehint, new_kw...)
                end
                for (x, c) in a
                    if has_ac
                        $F(x; addto, coeff = coeff*c, new_kw...)
                    else
                        addmul!(addto, $F(x; new_kw...), coeff*c)
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
    promote_type(_coefftype(multilin_return_type(f, x)), map(_coefftype, x)...)
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
        addto = zero(Linear{multilin_term_type(f, a), coefftype}),
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
        new_kw = kw
        if has_isfiltered(f, $TS...)
            new_kw = (; is_filtered = true, new_kw...)
        end
        if has_sizehint(f, $TS...)
            new_kw = (; sizehint, new_kw...)
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
                @ncallkw($N, f, (addto, coeff = cc_1, new_kw...), x)
            else
                addmul!(addto, @ncallkw($N, f, new_kw, x), cc_1; is_filtered = keeps_filtered(f, $TS...))
            end
        end)
        addto
    end
end

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
                addto = zero(Linear{multilin_term_type($F0, a), coefftype}),
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
                new_kw = kw
                if $has_isfiltered($$F0, $TT...)
                    new_kw = (; is_filtered = true, new_kw...)
                end
                if $has_sizehint($$F0, $TT...)
                    new_kw = (; sizehint, new_kw...)
                elseif sizehint # && !(return_type(f, $TT...) <: AbstractLinear)
                    l = prod($_length, a; init = 1)
                    sizehint!(addto, length(addto)+l)
                end
                Base.Cartesian.@nexprs(1, i -> cc_{$N+i} = coeff)  # initialize cc_{N+1}
                Base.Cartesian.@nloops($N, xc, i -> a[i] isa $Linear ? a[i] : ((a[i], $ONE),), i -> begin
                    x_i, c_i = xc_i
                    cc_i = c_i*cc_{i+1}
                end, begin
                    if has_ac   # || return_type(f, $TT...) <: AbstractLinear   # for testing
                        $$(@__MODULE__).@ncallkw($N, $$F0, (addto, coeff = cc_1, new_kw...), x)
                    else
                        $addmul!(addto, $$(@__MODULE__).@ncallkw($N, $$F0, new_kw, x), cc_1; is_filtered = $keeps_filtered($$F0, $TT...))
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

@linear coprod

#
# differential
#

export diff

@linear diff

deg(::typeof(diff)) = -1
