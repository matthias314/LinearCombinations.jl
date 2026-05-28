using LinearCombinations, Test
using LinearCombinations.TestHelpers
using LinearCombinations: Zero, Sign, ONE, withsign, DefaultCoefftype, unval, return_type, keeps_filtered, diff

using Base: Fix1
using InteractiveUtils: @code_typed
using MacroTools: inexpr
using StructEqualHash

module LinearTest

    using StructEqualHash
    using ..LinearCombinations, ..TestHelpers
    using ..LinearCombinations: return_type, unval, ONE, Eval

    @linear f
    f(x) = x * x

    @linear g
    @linear_kw function g(x::T;
            coefftype = Float64,
            addto = zero(Linear{T, unval(coefftype)}),
            coeff = ONE,
            is_filtered = false) where T
        addmul!(addto, x, 2*coeff; is_filtered)
    end

    LinearCombinations.return_type(::typeof(g), ::Type{T}) where T <: Union{Char,String,ErrorFilter} = Linear{T,Float64}
    LinearCombinations.return_type(::typeof(Eval), ::Type{typeof(g)}, ::Type{T}) where T <: Union{Char,String,ErrorFilter} = Linear{T,Float64}
    # necessary to avoid exceptions
end

module MultilinearTest

    using ..LinearCombinations, ..TestHelpers
    import ..LinearCombinations: keeps_filtered

    @multilinear(f, *)

    @multilinear g f
    keeps_filtered(::typeof(g), ::Type{Val{B1}}, ::Type{Val{B2}}) where {B1,B2} = B1 && B2

    h(x::Char, y::Char) = x*y
    keeps_filtered(::typeof(h), ::Type{Val{B1}}, ::Type{Val{B2}}) where {B1,B2} = B1 && B2

    struct A end
    @multilinear ::A
    (::A)(x::Char, y::Char) = x*y

    struct B end
    @multilinear b::B
    (::B)(x::Char, y::Char) = x*y

    struct C end
    @multilinear ::C f
    keeps_filtered(::C, ::Type{Val{B1}}, ::Type{Val{B2}}) where {B1,B2} = B1 && B2

    struct D{T} f::T end
    @multilinear d::D d.f
    keeps_filtered(::D, ::Type{Val{B1}}, ::Type{Val{B2}}) where {B1,B2} = B1 && B2

    struct E end
    (::E)(x::Char, y::Char) = x*y
    keeps_filtered(::E, ::Type{Val{B1}}, ::Type{Val{B2}}) where {B1,B2} = B1 && B2
end

module LinearKwTest
    using ..LinearCombinations

    @linear_kw f1(x::Some{T}; coefftype = Int, is_filtered = false) where T <: AbstractChar = Linear(something(x) => one(coefftype); is_filtered)
    @linear_kw f2(x::Some{T}; addto = zero(Linear{T,Int}), coeff = 1, sizehint = false) where T <: AbstractChar = addmul!(addto, something(x), coeff)

    @linear_kw g1(xs::Char...; coefftype = Int, is_filtered = false) = Linear(string(xs...) => one(coefftype); is_filtered)
    @linear_kw g2(xs::Vararg; coefftype = Int, is_filtered = false) = Linear(string(xs...) => one(coefftype); is_filtered)
    @linear_kw g3(xs::Vararg{Char}; coefftype = Int, is_filtered = false) = Linear(string(xs...) => one(coefftype); is_filtered)
    @linear_kw g4(xs::Vararg{Char,N}; coefftype = Int, is_filtered = false) where N = Linear(string(xs...) => one(coefftype); is_filtered)
end

@testset "Sign" begin
    s0::Sign = 1
    s1::Sign = -1
    @test isone(s0) && !isone(s1)
    @test !iszero(s0) && !iszero(s1)
    @test s0 === one(Sign) === one(s1)
    @test +s0 == -s1 == s0
    @test +s1 == -s0 == s1
    @test s0*s0 == s1*s1 == s0
    @test s0*s1 == s1*s0 == s1
    for x in (s0, s1, Int8(1), Int(2), Float32(0.25), BigInt(-1))
        @test s0*x === x*s0 === x
        @test s1*x == x*s1 == -x
        @test (s0 == x) == (x == s0) == isone(x)
        @test (s1 == x) == (x == s1) == isone(-x)
    end
    @test hash(s0) == hash(1)
    @test hash(s1) == hash(-1)
    @test hash(s0, UInt(7)) == hash(1, UInt(7))
    @test hash(s1, UInt(7)) == hash(-1, UInt(7))

    for x in (Int16(1), 1, 1.0, BigFloat(1))
        @test convert(Sign, x) == s0
        @test convert(Sign, -x) == s1
        @test_throws Exception convert(Sign, 2*x)
    end

    for R in (Int32, Float16, BigInt)
        @test convert(R, s0) == one(R)
        @test convert(R, s1) == -one(R)
        @test promote_type(Sign, R) === R
    end
end

@testset "Zero" begin
    z = Zero()
    @test iszero(z) && iseven(z) && !isodd(z)
    @test z === zero(Zero) === zero(z)
    @test -z === z
    for x in (z, Int8(-4), 10, 3.5, BigInt(2))
        @test z+x === x+z === x-z === x
        @test z-x == -x
        @test z*x === x*z === z
    end

    for R in (Int32, Float16, BigInt)
        @test convert(R, z) == zero(R)
        @test promote_type(Zero, R) === R
    end
end

@testset "basic operations" begin
    a = Linear('x' => 1, 'y' => -2, 'z' => 0)
    b = Linear('x' => 1.0, 'y' => -2.0, 'z' => 0.0)
    c = Linear("x" => 1.0, "y" => -2.0, "z" => 0.0)

    @test @inferred(termtype(a)) == Char
    @test @inferred(coefftype(b)) == Float64
    @test @inferred(eltype(c)) == Pair{String,Float64}

    @test @inferred(length(a)) == 2
    @test @inferred(a['x']) == 1
    @test @inferred(a['w']) == 0
    c["x"] = 3.5
    @test c["x"] == 3.5
    @test !iszero(a)
    @test a == b != c

    aa = @inferred copy(a)
    @test aa == a && aa !== a
    @test typeof(aa) == typeof(a)

    @test @inferred(hash(a)) isa UInt
    @test hash(a) == hash(aa) == hash(b) != hash(c)

    @test @inferred(Set(coeffs(b))) == Set([1.0, -2.0])

    @test iszero(@inferred(zero(a)))
    @test iszero(@inferred(zero(typeof(a))))

    @test sizehint!(a, 2*length(a)) == a

    aa = copy(a)
    @test iszero(@inferred(zero!(aa)))

    a = Linear('a'+k => k for k in 1:8)
    @test a == Linear(a)
    @test a == Linear(x => c for (x, c) in a)
end

@testset "addmul, addmul!, add!, sub!" begin
    a = Linear('x' => 1, 'y' => -2)
    @test @inferred(addmul(a, 'w', 0)) == a
    @test @inferred(addmul(a, 'w', 1)) == a + 'w'
    @test @inferred(addmul(a, 'w', 2)) == a + Linear('w' => 2)
    @test @inferred(addmul(a, 'w', -1)) == a - 'w'
    @test @inferred(addmul(a, 'w', -2)) == a - Linear('w' => 2)

    @test +a == a && +a !== a  # see julia#58295

    aa = copy(a)
    @test @inferred(addmul!(aa, 'w', 0)) === aa
    @test aa == a
    aa = copy(a)
    @test @inferred(addmul!(aa, 'w', 1)) === aa
    @test aa == a + 'w'
    aa = copy(a)
    @test @inferred(addmul!(aa, 'w', 2)) === aa
    @test aa == a + Linear('w' => 2)
    aa = copy(a)
    @test @inferred(addmul!(aa, 'w', -1)) === aa
    @test aa == a - 'w'
    aa = copy(a)
    @test @inferred(addmul!(aa, 'w', -2)) === aa
    @test aa == a - Linear('w' => 2)

    b = Linear('x' => -1.0, 'z' => 3.0)
    aa = addmul(a, b, 2.0)
    @test termtype(aa) == termtype(a)
    @test coefftype(aa) == promote_type(coefftype(a), coefftype(b), Float64)
    @test @inferred(addmul(a, b, 0)) == a
    @test @inferred(addmul(a, b, 1)) == a + b
    @test @inferred(addmul(a, b, 2)) == a + 2*b
    @test @inferred(addmul(a, b, -1)) == a - b
    @test @inferred(addmul(a, b, -2)) == a - 2*b

    aa = copy(a)
    @test @inferred(addmul!(aa, b, 0)) === aa == a
    aa = copy(a)
    @test @inferred(addmul!(aa, b, 1)) === aa == a + b
    aa = copy(a)
    @test @inferred(addmul!(aa, b, 2)) === aa == a + 2*b
    aa = copy(a)
    @test @inferred(addmul!(aa, b, -1)) === aa == a - b
    aa = copy(a)
    @test @inferred(addmul!(aa, b, -2)) === aa == a - 2*b

    aa = copy(a)
    @test @inferred(add!(aa, b)) === aa == a + b
    aa = copy(a)
    @test @inferred(sub!(aa, b)) === aa == a - b
end

@testset "conversion" begin
    a = Linear('a'+k => k for k in 1:8)
    @test @inferred convert(typeof(a), a) === a
    @test @inferred convert(Linear{Char,Float64}, 'x') == Linear('x' => 1.0)

    a = Linear(Int8(k) => k for k in 1:8)
    @test @inferred convert(Linear{BigInt,Float64}, a) == Linear(BigInt(k) => Float64(k) for k in 1:8)
    @test @inferred convert(Linear{Int8,Float64}, a) == Linear(Int8(k) => Float64(k) for k in 1:8)
end

@testset "add and sub" begin
    for R in (Int8, Int, BigInt, Float64, BigFloat), S in (Int8, Int, BigInt, Float64, BigFloat)
        a = Linear{Char,R}('x' => 2, 'y' => -1, 'z' => 1)
        b = Linear{Char,S}('x' => 2, 'y' => -1, 'z' => 1)
        c = @inferred(a+b)
        @test coefftype(c) == promote_type(R, S)
        @test promote_type(typeof(a), typeof(b)) == typeof(c)
        @test c == 2*a
        c = @inferred(+a)
        @test typeof(c) == typeof(a)
    end

    for R in (Int8, Int, BigInt, Float64, BigFloat), S in (Int8, Int, BigInt, Float64, BigFloat)
        a = Linear{Char,R}('x' => 2, 'y' => -1, 'z' => 1)
        b = Linear{Char,S}('x' => 2, 'y' => -1, 'z' => 1)
        c = @inferred(a-b)
        @test coefftype(c) == promote_type(R, S)
        @test promote_type(typeof(a), typeof(b)) == typeof(c)
        @test iszero(c)
        c = @inferred(-a)
        @test typeof(c) == typeof(a)
    end

    for R in (Int8, Int, BigInt, Float64, BigFloat), op in (+, -)
        y = 'y'
        a = Linear{Char,R}('x' => 2, y => -1, 'z' => 1)
        b = Linear{Char,R}(y => 1)
        c = @inferred op(a, y)
        @test coefftype(c) == R
        @test c == op(a, b)
        @test promote_type(typeof(a), typeof(y)) == typeof(c)
        c = @inferred op(y, a)
        @test coefftype(c) == R
        @test c == op(b, a)
        @test promote_type(typeof(y), typeof(a)) == typeof(c)
    end
end

@testset "scalar mul" begin
    for R in (Int8, Int, BigInt, Float64, BigFloat), S in (Int8, Int, BigInt, Float64, BigFloat)
        a = Linear{Char,R}('x' => 2, 'y' => -1, 'z' => 1)
        c = 3
        b = @inferred S(c)*a
        @inferred a*S(c)
        @test b == c*a == a*S(c)
        @test promote_type(typeof(a), S) == typeof(b)

        @test iszero(zero(S)*a)
    end
end

@testset "Linear mul" begin
    for R in (Int8, Int, BigInt, Float64, BigFloat), S in (Int8, Int, BigInt, Float64, BigFloat)
        a = Linear{String,R}("x" => 2, "y" => -1)
        b = Linear{String,S}("u" => 1, "v" => -3)
        c = Linear("xu" => 2, "xv" => -6, "yu" => -1, "yv" => 3)
        ab = @inferred a*b
        @test ab == c
        @test coefftype(ab) == promote_type(R, S)
    end

    for R in (Int, BigFloat)
        a = Linear{String,R}("x" => 2, "y" => -1)
        c = @inferred(a^8)
        @test typeof(c) == typeof(a)
        b = a*a
        b = b*b
        @test c == b*b
    end
end

@testset "broadcasting" begin
    for R in (Int16, Float32)
        a = Linear{Char,R}('x' => 1)
        b = Linear{Char,R}('y' => 2)
        c = Linear{Char,R}('z' => 3)
        @test a == @inferred .+ a
        @test -a == @inferred .- a
        aa = @inferred a .+ 'x'
        @test  aa == a + 'x'
        aa = @inferred a .- 'x'
        @test  aa == a - 'x'
        aa = @inferred a .+ 'y'
        @test  aa == a + 'y'
        aa = @inferred a .- 'y'
        @test  aa == a - 'y'
        @test a + b == @inferred a .+ b
        @test a - b == @inferred a .- b
        @test 3*a == @inferred 3 .* a
        @test -24*a == @inferred 2 .* (3 .* (4 .* (.- a)))
        @test 2*a + 15*b - 2*c == @inferred 5 .* (2 .* a .+ 3 .* b) .+ 2 .* (-1 .* c .- 4 .* a)
        @test 18*a + 15*b + 2*c == @inferred 5 .* (2 .* a .+ 3 .* b) .- 2 .* (-1 .* c .- 4 .* a)
        aa = copy(a)
        aa .= b .- c
        @test aa == b - c
        aa .= 3 .* b
        @test aa == 3*b
        aa = copy(a)
        aa .+= b
        @test aa == a + b
        aa = copy(a)
        aa .-= b
        @test aa == a - b
        aa = copy(a)
        aa .+= 2 .* b
        @test aa == a + 2*b
        aa = copy(a)
        aa .-= 2 .* b .- 3 .* c
        @test aa == a - 2*b + 3*c
    end
end

@testset "@linear_kw" begin
    import .LinearKwTest as L
    using ..LinearCombinations: has_coefftype, has_addto_coeff, has_isfiltered, has_sizehint

    function returns_const(ct::Pair{Core.CodeInfo}, val)
        code = first(ct).code
        length(code) == 1 || return false
        code[1] isa Core.ReturnNode && code[1].val == val
    end

    @test returns_const(@code_typed(has_coefftype(L.f1, Some{Char})), true)
    @test returns_const(@code_typed(has_coefftype(L.f1, Some{String})), false)
    @test returns_const(@code_typed(has_addto_coeff(L.f1, Some{Char})), false)
    @test returns_const(@code_typed(has_isfiltered(L.f1, Some{Char})), true)
    @test returns_const(@code_typed(has_sizehint(L.f1, Some{Char})), false)

    @test returns_const(@code_typed(has_coefftype(L.f2, Some{Char})), false)
    @test returns_const(@code_typed(has_addto_coeff(L.f2, Some{Char})), true)
    @test returns_const(@code_typed(has_addto_coeff(L.f2, Some{String})), false)
    @test returns_const(@code_typed(has_isfiltered(L.f2, Some{Char})), false)
    @test returns_const(@code_typed(has_sizehint(L.f2, Some{Char})), true)

    for f in [L.g1, L.g2, L.g3, L.g4]
        @test returns_const(@code_typed(has_coefftype(f, Char)), true)
        if f == L.g2
            @test returns_const(@code_typed(has_coefftype(f, String)), true)
        else
            @test returns_const(@code_typed(has_coefftype(f, String)), false)
        end
        @test returns_const(@code_typed(has_addto_coeff(f, Char)), false)
        @test returns_const(@code_typed(has_isfiltered(f, Char)), true)
        @test returns_const(@code_typed(has_sizehint(f, Char)), false)

        @test returns_const(@code_typed(has_coefftype(f, Char, Char)), true)
        if f == L.g2
            @test returns_const(@code_typed(has_coefftype(f, String, Char)), true)
        else
            @test returns_const(@code_typed(has_coefftype(f, String, Char)), false)
        end
        @test returns_const(@code_typed(has_addto_coeff(f, Char, Char)), false)
        @test returns_const(@code_typed(has_isfiltered(f, Char, Char)), true)
        @test returns_const(@code_typed(has_sizehint(f, Char, Char)), false)
    end

    @test returns_const(@code_typed(has_coefftype(Tensor(L.f1, L.f1), Tensor{Tuple{Some{Char},Some{Char}}})), :true)

    ex = :( @inline @noinline f(x::Char; coefftype = Int) = Linear(x => one(coefftype)) )
    newex = LinearCombinations.linear_kw(LineNumberNode(@__LINE__, @__FILE__), ex)
    @test inexpr(newex, Symbol('@', :inline))
    @test inexpr(newex, Symbol('@', :noinline))
end

@testset "@linear" begin
    using .LinearTest: f, g

    for R in (Int8, Int, Float64, BigFloat)
        a = Linear{Char,R}('x' => 1, 'y' => 2, 'z' => 3)
        b = Linear("xx" => 1, "yy" => 2, "zz" => 3)
        c = @inferred f(a)
        @test typeof(c) == Linear{String,R}
        @test c == b

        c = @inferred f(a; coefftype = Val(Int16))
        @test typeof(c) == Linear{String,promote_type(Int16,R)}
        @test c == b

        c = @inferred f(a; coeff = -2)
        @test typeof(c) == Linear{String,R}
        @test c == -2*b

        d = zero(Linear{String,BigInt})
        c = @inferred f(a; addto = d)
        @test c === d == b

        b = 2*a
        S = promote_type(R, Float64)
        c = @inferred g(a)
        @test typeof(c) == Linear{Char,S}
        @test c == b

        c = @inferred g(a; coeff = -2)
        @test typeof(c) == Linear{Char,S}
        @test c == -2*b

        d = zero(Linear{Char,BigInt})
        c = @inferred g(a; addto = d)
        @test c === d == b

        h = Fix1(*, 'h')
        c = @inferred h(a)
        @test c == 'h' * a
    end
end

@testset "LinearExtension" begin
    using .LinearTest: f, g

    h = LinearExtension(x -> x*x)
    j = LinearExtension(g)
    for R in (Int8, Int, Float64, BigFloat)
        a = Linear{Char,R}('x' => 1, 'y' => 2, 'z' => 3)
        b = Linear("xx" => 1, "yy" => 2, "zz" => 3)
        c = @inferred h(a)
        @test typeof(c) == Linear{String,R}
        @test c == b

        c = @inferred h(a; coeff = -2)
        @test typeof(c) == Linear{String,R}
        @test c == -2*b

        d = zero(Linear{String,BigInt})
        c = @inferred h(a; addto = d)
        @test c === d == b

        b = 2*a
        S = promote_type(R, Float64)
        c = @inferred j(a)
        @test typeof(c) == Linear{Char,S}
        @test c == b

        c = @inferred j(a; coeff = -2)
        @test typeof(c) == Linear{Char,S}
        @test c == -2*b

        d = zero(Linear{Char,BigInt})
        c = @inferred j(a; addto = d)
        @test c === d == b
    end
end

@testset "@multilinear" begin
    import .MultilinearTest as M

    for f in [M.f, M.A(), M.B()]
        @test String == @inferred return_type(f, Char, Char)
        @test Linear{String,DefaultCoefftype} == @inferred return_type(f, Linear{Char,DefaultCoefftype}, Char)
        @test Linear{String,BigFloat} == @inferred return_type(f, Linear{Char,BigInt}, Linear1{Char,Float32})
        @test Linear1{String,Int8} == @inferred return_type(f, Char, Linear1{Char,Int8})
    end

    for f in [M.g, M.C(), M.D(M.f), MultilinearExtension(M.h), MultilinearExtension(M.E())]
        # @test Linear1{String,DefaultCoefftype} == @inferred return_type(f, Char, Char)
        @test String == @inferred return_type(f, Char, Char)
        @test Linear{String,DefaultCoefftype} == @inferred return_type(f, Linear{Char,DefaultCoefftype}, Char)
        @test Linear{String,BigFloat} == @inferred return_type(f, Linear{Char,BigInt}, Linear1{Char,Float32})
        @test Linear1{String,Int8} == @inferred return_type(f, Char, Linear1{Char,Int8})
        @test false == @inferred keeps_filtered(f, Val{true}, Val{false})
        @test true == @inferred keeps_filtered(f, Val{true}, Val{true})
    end
end

@testset "Linear callable" begin
    struct P
        s::String
    end

    @struct_equal_hash P

    (p::P)(x) = p.s * x
    (p::P)(x, y; addto = zero(Linear{String,Int16}), coeff = ONE) = addmul!(addto, p.s * x * y, coeff*Int16(5))

    for R in (Int8, BigInt, BigFloat), S in (Int8, Int32, Float64)
        x, y, u, v = "x", "y", "u", "v"
        a = Linear(P(x) => R(1), P(y) => R(2))
        b = Linear{String,S}(u => -1, v => 3)

        au = @inferred a(u)
        @test au isa Linear{String,R}
        @test au == Linear("xu" => 1, "yu" => 2)

        au = @inferred a(u; coeff = 3.0)
        @test au isa Linear{String,R}
        @test au == 3*Linear("xu" => 1, "yu" => 2)

        c = zero(Linear{String,R})
        au = @inferred a(u; addto = c, coeff = 3.0)
        @test au === c == 3*Linear("xu" => 1, "yu" => 2)

        ab = @inferred a(b)
        @test ab isa Linear{String,promote_type(R,S)}
        @test ab == Linear("xu" => -1, "xv" => 3, "yu" => -2, "yv" => 6)

        ab = @inferred a(b; coeff = 5)
        @test ab isa Linear{String,promote_type(R,S)}
        @test ab == 5*Linear("xu" => -1, "xv" => 3, "yu" => -2, "yv" => 6)

        c = copy(ab)
        ab = @inferred a(b; addto = c, coeff = 5)
        @test ab === c == 2*5*Linear("xu" => -1, "xv" => 3, "yu" => -2, "yv" => 6)

        a = Linear(P(x) => R(2))
        b = Linear(u => S(3))
        c = Linear(v => Int64(-1))
        auv = @inferred a(u, v)
        @test typeof(auv) == Linear{String,promote_type(Int16,R)}
        @test auv == Linear("xuv" => 5*2)

        abv = @inferred a(b, v)
        @test typeof(abv) == Linear{String,promote_type(Int16,R,S)}
        @test abv == Linear("xuv" => 5*2*3)

        abc = @inferred a(b, c)
        @test typeof(abc) == Linear{String,promote_type(Int64,R,S)}
        @test abc == Linear("xuv" => -5*2*3)
    end
end

@testset "Tensor and tensor" begin
    @inferred(Tensor()) isa Tensor{Tuple{}}

    tt = ('x', "y", [1,2])
    t = @inferred Tensor(tt)
    @test t == @inferred Tensor(tt...)
    @test @inferred(Tuple(t)) == tt
    @test Tuple(x for x in t) == tt
    @test fieldtypes(typeof(t)) == fieldtypes(typeof(Tuple(t)))

    @test @inferred(hash(t)) isa UInt
    t2 = Tensor('x', "y", [1,2])
    @test t2 !== t && t2 == t && hash(t2) == hash(t)
    t3 = Tensor('x', "y", [1,3])
    @test t3 != t && hash(t3) != hash(t)

    tt = ('x', Linear("y" => 1), [1,2])
    a = @inferred tensor(tt...)
    @test a == Linear(t => ONE)
    b = zero(Linear{Tensor{Tuple{Char, String, Vector{Int64}}}, Float64})
    a = @inferred tensor(tt...; addto = b)
    @test a === b == Linear(t => 1.0)

    b = zero(Linear{Tensor{Tuple{Char, String, Vector{Int64}}}, Float64})
    a = @inferred tensor(tt...; addto = b, coeff = -2)
    @test a === b == Linear(t => -2.0)
    @test iszero(tensor(tt...; addto = a, coeff = 2))

    @test @inferred(tensor()) == Tensor()

    for R in (Int8, Int, BigInt, Float64, BigFloat), S in (Int8, Int, BigInt, Float64, BigFloat)
        a = Linear{Char,R}('x' => 1, 'y' => -2)
        b = Linear{String,S}("u" => -1, "v" => 3)
        c = @inferred tensor(a, b)
        @test termtype(c) == Tensor{Tuple{Char,String}}
        @test coefftype(c) == promote_type(R, S)
        @test c == Linear(Tensor('x', "u") => -1, Tensor('x', "v") => 3,
            Tensor('y', "u") => 2, Tensor('y', "v") => -6)

        c = zero(Linear{Tensor{Tuple{Char,String}},Int32})
        @inferred tensor(a, b; addto = c)
        cc = copy(c)
        @test tensor(a, b; addto = cc, coeff = -3) == -2*c
    end

    for n in 1:8
        a = @inferred tensor(['a'+k for k in 1:n]...)
        @test a isa Tensor{NTuple{n,Char}}
        a = @inferred tensor([Linear1('a'+k => 2) for k in 1:n]...)
        @test a isa Linear1{Tensor{NTuple{n,Char}},Int}
        a = @inferred tensor([Linear('a'+k => 2.0) for k in 1:n]...)
        @test a isa Linear{Tensor{NTuple{n,Char}},Float64}
    end

    for t in (Tensor(), Tensor('x'), Tensor('x', 'y', 'z'))
        @test @inferred(deg(t)) === Zero()
    end

    a = Linear('x' => 1, 'y' => -1)
    for n in 1:8
        t = Tuple(a for _ in 1:n)
        c = @inferred tensor(t...)
        @test length(c) == 2^n
    end

    B = Basis('x':'z')
    b = DenseLinear('x' => 1, 'y' => -1; basis = B)
    c = @inferred(tensor(b, b; coeff = 2))
    @test c isa DenseLinear && c == tensor(a, a; coeff = 2)
    c = @inferred tensor(b, tensor(b, b))
    @test c isa DenseLinear && c == tensor(a, tensor(a, a))
    c = @inferred tensor(tensor(b, b), tensor(b, b))
    @test c isa DenseLinear && c == tensor(tensor(a, a), tensor(a, a))
    c = @inferred tensor(b, tensor())
    @test_broken c isa DenseLinear && c == tensor(a, tensor())
end

@testset "Tensor deg" begin
    degs = (Zero(), 1, BigInt(2))
    for k1 in degs, k2 in degs, k3 in degs
        t = Tensor(Graded('x', k1), Graded('y', k2), Graded('z', k3))
        @test @inferred(deg(t)) == k1+k2+k3
    end
end

@testset "TensorSlurp, TensorSplat" begin
    import .MultilinearTest as M

    a = Linear('x' => 1, 'y' => 2)
    b = Linear("z" => -1, "w" => 3)

    @test 'x'*"z" == @inferred TensorSplat(*)(Tensor('x', "z"))
    @test a*b == @inferred TensorSplat(*)(tensor(a, b))
    @test false == @inferred keeps_filtered(TensorSplat(M.h), Tensor{Tuple{Val{true}, Val{false}}})
    @test true == @inferred keeps_filtered(TensorSplat(M.h), Tensor{Tuple{Val{true}, Val{true}}})

    f(t::Tensor) = *(t...)
    LinearCombinations.keeps_filtered(::typeof(f), ::Type{Tensor{Tuple{Val{B1},Val{B2}}}}) where {B1,B2} = B1 && B2
    @test 'x'*"z" == @inferred TensorSlurp(f)('x', "z")
    @test a*b == @inferred TensorSlurp(f)(a, b)
    @test false == @inferred keeps_filtered(TensorSlurp(f), Val{true}, Val{false})
    @test true == @inferred keeps_filtered(TensorSlurp(f), Val{true}, Val{true})
end

@testset "tensor callable" begin
    h = Tensor()
    t = Tensor()
    @test t == @inferred h(t)
    a = Linear(t => 2)
    b = @inferred h(a)
    @test b == a
    @test h(a; coeff = 3) == 3*a

    degs = (Zero(), 1, 2)

    for m1 in degs, m2 in degs, n1 in degs, n2 in degs
        h = Tensor(Graded(identity, m1), Graded(identity, m2))
        t = Tensor(Graded('x', n1), Graded('y', n2))
        b = @inferred h(t)
        se = m2*n1
        u = Tensor(Graded('x', m1+n1), Graded('y', m2+n2))
        if se isa Zero
            @test b == u
        else
            @test b == Linear1(u => withsign(se, 1))
            @test b isa Linear1
            @test coefftype(b) == DefaultCoefftype
        end

        a = Linear(t => 2.0)
        b = @inferred h(a)
        @test b == Linear1(u => withsign(se, 2.0))
        @test b isa Linear
        @test coefftype(b) == Float64
    end

    for m1 in degs, m2 in degs, n1 in degs, n2 in degs, o1 in degs, o2 in degs
        h = Tensor(Graded(*, m1), Graded(*, m2))
        t = Tensor(Graded('x', n1), Graded('y', n2))
        t2 = Tensor(Graded('p', o1), Graded('q', o2))
        b = @inferred h(t, t2)
        se = m2*n1 + o1*(m2+n2)
        u = Tensor(Graded("xp", m1+n1+o1), Graded("yq", m2+n2+o2))
        if se isa Zero
            @test b == u
        else
            @test b == Linear1(u => withsign(se, 1))
            @test b isa Linear1
            @test coefftype(b) == DefaultCoefftype
        end

        a1 = Linear(t => 2.0)
        a2 = Linear1(t2 => Int8(3))
        b = @inferred h(a1, a2)
        @test b == Linear1(u => withsign(se, 6.0))
    end
end

@testset "regroup" begin
    @test_throws "incompatible" regroup(:((1,2)), :((1,2,3)))
    @test_throws "incompatible" regroup(:((1,2)), :((1,3)))
    @test_throws "incompatible" regroup(:((1,2)), :((1,)))
    @test_throws "malformed" regroup(:((1,1)), :((1,2)))
    @test_throws "malformed" regroup(:((1,2)), :((1,1)))
    @test_throws Exception regroup(1, :(1,))

    rg1 = regroup(:((1,(2,3))), :(3,(1,2)))
    rg2 = regroup(:(('a',('b','c'))), :('c',('a','b')))
    @test rg1 == rg2

    rg3 = regroup"(1,(2,3)) -> (3,(1,2))"
    @test rg1 == rg3

    rg1i = regroup_inv(:((1,(2,3))), :(3,(1,2)))
    rg2i = regroup(:((1,(2,3))), :(3,(1,2))), regroup(:(3,(1,2)), :((1,(2,3))))
    rg3i = regroup_inv"(1,(2,3)) -> (3,(1,2))"
    @test rg1i == rg2i == rg3i
end

@testset "regroup tensor" begin
    rg, rg_inv = regroup_inv(:((1,(2,3))), :(3,(1,2)))
    t = Tensor('x', Tensor("y", [1,2]))
    a = Linear(t => 1)
    @inferred rg(t)
    @test rg_inv(rg(t)) == t
    @inferred rg(a)
    @test @inferred(deg(rg)) === Zero()

    @test rg_inv(rg(a)) == a
    @test rg(rg(rg(t))) == t
    @test rg(rg(rg(a))) == a

    a = tensor('x', "y")
    @inferred swap(a)
    @test swap(swap(a)) == a

    rg = regroup(:(), :())
    b = tensor()
    @inferred rg(b)
    @test rg(b) == b

    rg, rg_inv = regroup_inv(:((1,)), :(((1,),)))
    b = tensor('x')
    @test rg(b) == tensor(b)
    @test rg_inv(tensor(tensor(b))) == tensor(b)
end

@testset "regroup tensor kw args" begin
    a = @inferred tensor(Linear('x' => 1), "y"; coefftype = Val(Int))
    b = @inferred swap(a; coeff = 2)
    @test b == 2*swap(a)

    b = zero(Linear{Tensor{Tuple{String,Char}},Int})
    @test b === swap(a; addto = b)
    @test b == swap(a)

    a = Linear1(Tensor('x', "y") => 1)
    b = zero(Linear1{Tensor{Tuple{String,Char}},Float64})
    swap(a; addto = b)
    c = @inferred swap(a; coefftype = Val(Float64))
    @test b == c
    @test typeof(b) == typeof(c)
end

@testset "regroup tensor sign" begin
    degs = (Zero(), 1, 2)

    for k1 in degs, k2 in degs
        t = Tensor(Graded('x', k1), Graded('y', k2))
        a = @inferred swap(t)
        se = k1*k2
        u = Tensor(t[2], t[1])
        if se isa Zero
            @test a == u
        else
            @test a == Linear1(u => withsign(se, 1))
            @test a isa Linear1
            @test coefftype(a) == DefaultCoefftype
        end
    end

    rg = regroup(:(1,2,3), :(3,2,1))
    for k1 in degs, k2 in degs, k3 in degs
        t = Tensor(Graded('x', k1), Graded('y', k2), Graded('z', k3))
        a = @inferred rg(t)
        se = k3*(k1+k2) + k2*k1
        u = Tensor(t[3], t[2], t[1])
        if se isa Zero
            @test a == u
        else
            @test a == Linear1(u => withsign(se, 1))
            @test a isa Linear1
            @test coefftype(a) == DefaultCoefftype
        end
    end
end

@linear_kw LinearCombinations.diff(x::ErrorFilter{GradedString}; is_filtered = false) = Linear1(x => 1; is_filtered)
@linear_kw LinearCombinations.diff(x::ErrorFilter{Char}) = x

LinearCombinations.return_type(::typeof(diff), ::Type{ErrorFilter{GradedString}}) = Linear1{ErrorFilter{GradedString},Int}
LinearCombinations.return_type(::typeof(diff), ::Type{ErrorFilter{Char}}) = ErrorFilter{Char}
# necessary because `diff` throws an exception without `is_filtered = true`

LinearCombinations.keeps_filtered(::typeof(diff), ::Type{ErrorFilter{Char}}) = true

@testset "filtered" begin
    @test_throws FilterException Linear(ErrorFilter('x') => 1)
    @test Linear(ErrorFilter('x') => 1; is_filtered = true) isa Linear
    @test_throws FilterException Linear1(ErrorFilter('x') => 1)
    @test Linear1(ErrorFilter('x') => 1; is_filtered = true) isa Linear1

    x = ErrorFilter('x')
    y = ErrorFilter(gr"y")

    # linear
    using ..LinearTest: f, g
    @test f(x) |> Returns(true)
    @test_throws FilterException g(x)
    @test g(x; is_filtered = true) |> Returns(true)
    for L in (Linear, Linear1)
        a = L(x => 1; is_filtered = true)
        @test_throws FilterException f(a)
        @test KeepsFiltered(f)(a) |> Returns(true)
        @test g(a) |> Returns(true)
    end

    # multilinear
    using ..MultilinearTest: f as ff
    @test (ff(x, x); true)
    for L in (Linear, Linear1)
        a = L(x => 1; is_filtered = true)
        for (p, q) in [('x', a), (a, 'x'), (a, a)]
            @test_throws FilterException ff(p, q)
            @test KeepsFiltered(ff)(p, q; is_filtered = true) |> Returns(true)
        end
    end

    # linear callable
    a = Linear(g => 1)
    @test_throws FilterException a(ErrorFilter('x'))
    @test a(ErrorFilter('x'); is_filtered = true) |> Returns(true)

    # composition
    @test x |> LinearComposedFunction(f, f) |> Returns(true)
    @test x |> LinearComposedFunction(KeepsFiltered(f), f) |> Returns(true)
    @test x |> LinearComposedFunction(f, KeepsFiltered(f)) |> Returns(true)
    @test x |> LinearComposedFunction(KeepsFiltered(f), KeepsFiltered(f)) |> Returns(true)

    a = Linear(x => 1; is_filtered = true)
    @test_throws FilterException a |> LinearComposedFunction(f, f)
    @test_throws FilterException a |> LinearComposedFunction(KeepsFiltered(f), f)
    @test_throws FilterException a |> LinearComposedFunction(f, KeepsFiltered(f))
    @test a |> LinearComposedFunction(KeepsFiltered(f), KeepsFiltered(f)) |> Returns(true)

    @test_throws FilterException x |> LinearComposedFunction(f, g)
    @test_throws FilterException x |> LinearComposedFunction(KeepsFiltered(f), g)
    @test LinearComposedFunction(KeepsFiltered(f), g)(x; is_filtered = true) |> Returns(true)
    @test LinearComposedFunction(g, KeepsFiltered(f))(x; is_filtered = true) |> Returns(true)
    @test LinearComposedFunction(g, g)(x; is_filtered = true) |> Returns(true)

    @test LinearComposedFunction(f, ff)(x, x) |> Returns(true)
    @test LinearComposedFunction(KeepsFiltered(f), ff)(x, x) |> Returns(true)
    @test LinearComposedFunction(f, KeepsFiltered(ff))(x, x) |> Returns(true)
    @test LinearComposedFunction(KeepsFiltered(f), KeepsFiltered(ff))(x, x) |> Returns(true)

    for (p, q) in [('x', a), (a, 'x'), (a, a)]
        @test_throws FilterException LinearComposedFunction(f, ff)(p, q; is_filtered = true)
        @test_throws FilterException LinearComposedFunction(KeepsFiltered(f), ff)(p, q; is_filtered = true)
        @test_throws FilterException LinearComposedFunction(f, KeepsFiltered(ff))(p, q; is_filtered = true)
        @test LinearComposedFunction(KeepsFiltered(f), KeepsFiltered(ff))(p, q; is_filtered = true) |> Returns(true)
    end

    # transpose
    t = Tensor(Tensor(x, x), Tensor(x, x))
    u = Tensor(Tensor(y, y), Tensor(y, y))
    @test (transpose(t); true)
    @test_throws FilterException transpose(u)
    @test (transpose(u; is_filtered = true); true)
    @test (Linear(t => 1; is_filtered = true) |> transpose; true)
    @test (Linear(u => 1; is_filtered = true) |> transpose; true)

    # tensor callable
    t = Tensor(x, x)
    tf = Tensor(Fix1(*, x), Fix1(*, x))
    tf1 = Tensor(KeepsFiltered(Fix1(*, x)), KeepsFiltered(Fix1(*, x)))
    @test (tf(t); true)
    u = Tensor(y, y)
    tg = Tensor(Fix1(*, y), Fix1(*, y))
    tg1 = Tensor(KeepsFiltered(Fix1(*, y)), KeepsFiltered(Fix1(*, y)))
    @test_throws FilterException tg(u)
    @test (tg1(u); true)
    a = Linear(t => 1; is_filtered = true)
    @test_throws FilterException tf(a)
    @test (KeepsFiltered(tf)(a); true)
    @test (tf1(a); true)
    b = Linear(u => 1; is_filtered = true)
    @test_throws FilterException tg(b)
    @test (tg1(b); true)

    # regroup
    @test (Tensor(x, x) |> swap; true)
    @test_throws FilterException swap(Tensor(y, y))
    @test (swap(Tensor(y, y); is_filtered = true); true)
    @test (Linear(Tensor(x, x) => 1; is_filtered = true) |> swap; true)
    @test (Linear(Tensor(y, y) => 1; is_filtered = true) |> swap; true)

    # tensor diff
    @test diff(x) |> Returns(true)
    @test Linear(x => 1; is_filtered = true) |> diff |> Returns(true)
    t = Tensor(x, x)
    @test_throws FilterException diff(t)
    @test diff(t; is_filtered = true) |> Returns(true)

    @test_throws FilterException diff(y)
    @test diff(y; is_filtered = true) |> Returns(true)
    @test Linear(y => 1; is_filtered = true) |> diff |> Returns(true)
    t = Tensor(y, y)
    @test_throws FilterException diff(t)
    @test diff(t; is_filtered = true) |> Returns(true)
end

using Modulo2: ZZ2

struct Char2Exception end

@linear error_unless_char2
@linear_kw function error_unless_char2(x::T; coefftype = Int) where T <: Union{Char,String}
    has_char2(unval(coefftype)) || throw(Char2Exception())
    Linear1{T,unval(coefftype)}(x => 1)
end

LinearCombinations.return_type(::typeof(error_unless_char2), ::Type{T}) where T <: Union{Char,String} = Linear1{T,Int}
# needed to bypass error if `coefftype` is not given

@linear linear_char2
linear_char2(x) = Linear1(x => ZZ2(1))

LinearCombinations.diff(x::ErrorGraded) = Linear1(x => 1)

@testset failfast=true "char 2" begin
    # linear coefftype
    b = Linear('x' => Int8(1))
    @test_throws Char2Exception error_unless_char2('x')
    @test_throws Char2Exception error_unless_char2(b)
    a = Linear('x' => ZZ2(1))
    @test a |> error_unless_char2 |> coefftype |> ==(ZZ2)

    # multilinear coefftype
    @test_throws Char2Exception LinearComposedFunction(error_unless_char2, *)('x', 'y')
    @test_throws Char2Exception LinearComposedFunction(error_unless_char2, *)('x', b)
    @test_throws Char2Exception LinearComposedFunction(error_unless_char2, *)(b, b)
    @test LinearComposedFunction(error_unless_char2, *)(a, a) |> coefftype |> ==(ZZ2)
    @test LinearComposedFunction(error_unless_char2, *)(a, 'y') |> coefftype |> ==(ZZ2)
    @test LinearComposedFunction(error_unless_char2, *)(b, a) |> coefftype |> ==(ZZ2)

    # LinearComposedFunction coefftype
    @test_throws Char2Exception 'x' |> LinearComposedFunction(identity, error_unless_char2)
    @test 'x' |> LinearComposedFunction(linear_char2, error_unless_char2) |> coefftype |> ==(ZZ2)
    @test_throws Char2Exception 'x' |> LinearComposedFunction(error_unless_char2, identity)
    @test 'x' |> LinearComposedFunction(error_unless_char2, linear_char2) |> coefftype |> ==(ZZ2)

    # tensor callable coefftype
    @test_throws Char2Exception Tensor('x', 'y', 'z') |> Tensor(identity, identity, error_unless_char2)
    @test Tensor('x', 'y', 'z') |> Tensor(linear_char2, identity, error_unless_char2) |> coefftype |> ==(ZZ2)

    # transpose sign
    t = Tensor(Tensor(ErrorGraded('x'), ErrorGraded('x')), Tensor(ErrorGraded('x'), ErrorGraded('x')))
    @test_throws DegreeException transpose(t)
    @test Linear1(t => ZZ2(1)) |> transpose |> coefftype |> ==(ZZ2)

    # tensor callable sign
    t = Tensor(ErrorGraded('x'), ErrorGraded('x'))
    tf = Tensor(ErrorGraded(identity), ErrorGraded(identity))
    @test_throws DegreeException tf(t)
    @test Linear1(t => ZZ2(1)) |> tf |> coefftype |> ==(ZZ2)

    # regroup sign
    t = Tensor(ErrorGraded('x'), ErrorGraded('x'))
    @test_throws DegreeException swap(t)
    @test Linear1(t => ZZ2(1)) |> swap |> coefftype |> ==(ZZ2)

    # diff sign
    t = Tensor(ErrorGraded('x'), ErrorGraded('x'))
    @test_throws DegreeException diff(t)
    @test Linear1(t => ZZ2(1)) |> diff |> coefftype |> ==(ZZ2)
end

function LinearCombinations.diff(grs::GradedString;
        coefftype = Int,
        addto = zero(Linear{GradedString,unval(coefftype)}),
        coeff = 1,
        is_filtered = false)
    if grs != gr"" && grs[1] != 'd'
        addmul!(addto, gr"d" * grs, coeff)
    end
    addto
end

@testset "diff" begin
    x = gr"xx"
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)

    x = Linear(gr"x" => 3.5, gr"y" => -1.0, gr"z" => 5.2)
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)
end

@testset "diff Tensor" begin
    x = Tensor(gr"x", gr"yy", gr"zzz")
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)

    a = Linear(gr"x" => 2, gr"xx" => -1)
    b = Linear(gr"yy" => 1, gr"yyy" => -3)
    c = Linear(gr"z" => 5, gr"zzz" => -4)

    x = @inferred tensor(a, b, c; coefftype = Val(Float64))
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)

    x = @inferred tensor(a, b, c; coefftype = Val(Float64))  # Int16 doesn't work!
    dx = @inferred diff(x; coeff = -2)
    @test dx isa Linear{Tensor{NTuple{3,GradedString}},Float64}
    @test dx == -2*diff(x)

    for n in 0:8
        a = tensor((GradedString(string(x)) for x in 'a':'a'+n-1)...)
        b = @inferred diff(a)
        @test iszero(diff(diff(b)))
    end
end
