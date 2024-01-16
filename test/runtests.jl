using LinearCombinations, Test

using LinearCombinations: Sign, Zero, ONE, unval, DefaultCoefftype

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
    @test typeof(aa) == typeof(a)
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

@testset "lin ext mul" begin
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

f(x) = x * x
@linear f

function g(x::Char;
        coefftype = Float64,
        addto = zero(Linear{Char,unval(coefftype)}),
        coeff = ONE,
        is_filtered = false)
    addmul!(addto, uppercase(x), 2*coeff)
end

@linear g

@testset "@linear" begin
    for R in (Int8, Int, Float64, BigFloat)
        a = Linear{Char,R}('x' => 1, 'y' => 2, 'z' => 3)
        b = Linear("xx" => 1, "yy" => 2, "zz" => 3)
        c = @inferred f(a)
        @test typeof(c) == Linear{String,R}
        @test c == b

        c = @inferred f(a; coefftype = Val(Int16))
        @test typeof(c) == Linear{String,Int16}
        @test c == b

        c = @inferred f(a; coeff = -2)
        @test typeof(c) == Linear{String,R}
        @test c == -2*b

        d = zero(Linear{String,BigInt})
        c = @inferred f(a; addto = d)
        @test c === d == b
	
	b = 2 * Linear('X' => 1, 'Y' => 2, 'Z' => 3)
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
    end
end

@testset "LinearExtension" begin
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
	
	b = 2 * Linear('X' => 1, 'Y' => 2, 'Z' => 3)
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
end

@testset "Tensor and tensor" begin
    @inferred(Tensor()) isa Tensor{Tuple{}}

    tt = ('x', "y", [1,2])
    t = @inferred Tensor(tt)
    @test t == @inferred Tensor(tt...)
    @test @inferred(Tuple(t)) == tt
    @test Tuple(x for x in t) == tt
    

    @test @inferred(hash(t)) isa UInt
    t2 = Tensor('x', "y", [1,2])
    @test t2 !== t && t2 == t && hash(t2) == hash(t)
    t3 = Tensor('x', "y", [1,3])
    @test t3 != t && hash(t3) != hash(t)

    a = @inferred tensor(tt...)
    @test a == Linear(t => ONE)
    b = zero(Linear{Tensor{Tuple{Char, String, Vector{Int64}}}, Float64})
    a = @inferred tensor(tt...; addto = b)
    @test a === b == Linear(t => 1.0)

    b = zero(Linear{Tensor{Tuple{Char, String, Vector{Int64}}}, Float64})
    a = @inferred tensor(tt...; addto = b, coeff = -2)
    @test a === b == Linear(t => -2.0)
    @test iszero(tensor(tt...; addto = a, coeff = 2))
    
    @test @inferred(tensor(; coeff = ONE)) isa Linear{Tensor{Tuple{}},DefaultCoefftype}
    
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
        @test a isa Linear{Tensor{NTuple{n,Char}},DefaultCoefftype}
        a = @inferred tensor([Linear('a'+k => 2.0) for k in 1:n]...)
        @test a isa Linear{Tensor{NTuple{n,Char}},Float64}
    end
    
    for t in (Tensor(), Tensor('x'), Tensor('x', 'y', 'z'))
        @test @inferred(deg(t)) === Zero()
    end

    a = Linear('x' => 1, 'y' => -1)
    for n in 0:8
        t = Tuple(a for _ in 1:n)
        c = @inferred tensor(t...)
        @test length(c) == 2^n
    end
end

import LinearCombinations: deg
deg(x::String) = length(x)

@testset "Tensor deg String" begin    
    for k1 in 1:3, k3 in 1:3
        t = Tensor("x"^k1, 'y', "z"^k3)
	@test @inferred(deg(t)) == k1+k3
    end
end

deg(x::Char) = BigInt(1)

@testset "Tensor deg Char String" begin    
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        t = Tensor("x"^k1, 'y', "z"^k3)
	@test @inferred(deg(t)) == k1+1+k3
        t = Tensor("x"^k1, "y"^k2, "z"^k3)
	@test @inferred(deg(t)) == k1+k2+k3
    end
end

@testset "TensorSlurp, TensorSplat" begin
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
end

@testset "regroup tensor" begin
    rg, rg_inv = regroup_inv(:((1,(2,3))), :(3,(1,2)))
    t = Tensor('x', Tensor("y", [1,2]))
    a = Linear(t => 1)
    @inferred rg(t)
    @test rg_inv(rg(t)) == a
    @inferred rg(a)
    @test @inferred(deg(rg)) === Zero()
    
    @test rg_inv(rg(a)) == a
    @test rg(rg(rg(t))) == a
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
    a = @inferred tensor('x', "y"; coefftype = Val(Int))
    b = @inferred swap(a; coeff = 2)
    @test b == 2*swap(a)
    
    b = zero(Linear{Tensor{Tuple{String,Char}},Int})
    @test b === swap(a; addto = b)
    @test b == swap(a)
    
    t = Tensor('x',"y")
    b = zero(Linear1{Tensor{Tuple{String,Char}},Float64})
    swap(t; addto = b)
    c = @inferred swap(t; coefftype = Val(Float64))
    @test b == c
    @test typeof(b) == typeof(c)
end

@testset "regroup tensor sign" begin 
    @test swap(tensor("x", "y")) == -tensor("y", "x")
    @test swap(tensor("x", "yy")) == tensor("yy", "x")
    @test swap(tensor("xx", "y")) == tensor("y", "xx")
    @test swap(tensor("xx", "yy")) == tensor("yy", "xx")
    
    rg = regroup(:(1,2,3), :(3,2,1))
    @test rg(tensor("x", "y", "z")) == -tensor("z", "y", "x")
    @test rg(tensor("x", "yy", "z")) == -tensor("z", "yy", "x")
    @test rg(tensor("xx", "yy", "z")) == tensor("z", "yy", "xx")
end

f0(x) = uppercase(x)
g0(x) = Linear(f0(x) => 2.0)

@testset "tensormap deg 0 0" begin
    h = tensormap()
    t = Tensor()
    a = @inferred tensor(; coefftype = Val(Int32))
    @test h(t) == Linear(t => 1)
    @test h(a; coeff = 3) == 3*a
    
    h = tensormap(f0, f0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Int}

        a = tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Int}
    end

    h = tensormap(f0, g0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(g0, f0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = Tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(g0, g0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 4)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = Tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 4)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(f0, f0, f0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        t = Tensor(x, y, z)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^0)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Int}
	
        a = Tensor(x, y, z)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^0)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Int}
    end

    h = tensormap(f0, f0, g0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        c = 3
        t = Tensor(x, y, z)
        b = @inferred(h(t; coeff = c))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^0 * 2 * c)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Float64}
	
        a = Tensor(x, y, z)
        b = @inferred(h(a; coeff = c))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^0 * 2 * c)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Float64}
    end

    h = tensormap(g0, g0, f0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        a = zero(Linear{Tensor{NTuple{3, String}}, BigFloat})
        t = Tensor(x, y, z)
        b = @inferred(h(t; addto = a))
        @test b === a == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^0 * 4)
    end

    h = tensormap(g0, g0, g0)
    @test @inferred(deg(h)) === Zero()
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        c = -5
        a = zero(Linear{Tensor{NTuple{3, String}}, BigInt})
        t = Tensor(x, y, z)
        b = @inferred(h(t; addto = a, coeff = c))
        @test b === a == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^0 * 8 * c)
    end
end

f1(x) = uppercase(x)
deg(::typeof(f1)) = 1

g1(x) = Linear(f1(x) => 2.0)
deg(::typeof(g1)) = 1

@testset "tensormap deg 1 0" begin
    h = tensormap()
    t = Tensor()
    a = @inferred tensor(; coefftype = Val(Int32))
    @test h(t) == Linear(t => 1)
    @test h(a; coeff = 3) == 3*a
    
    h = tensormap(f1, f1)
    @test @inferred(deg(h)) == deg(f1)+deg(f1)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Int}

        a = tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Int}
    end

    h = tensormap(f1, g0)
    @test @inferred(deg(h)) == deg(f1)+deg(g0)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(g0, f1)
    @test @inferred(deg(h)) == deg(g0)+deg(f1)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = Tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(g0, g0)
    @test @inferred(deg(h)) == deg(g0)+deg(g0)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 4)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = Tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^0 * 4)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(f1, f1, f1)
    @test @inferred(deg(h)) == deg(f1)+deg(f1)+deg(f1)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        t = Tensor(x, y, z)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Int}
	
        a = Tensor(x, y, z)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Int}
    end

    h = tensormap(f1, f1, g0)
    @test @inferred(deg(h)) == deg(f1)+deg(f1)+deg(g0)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        c = 3
        t = Tensor(x, y, z)
        b = @inferred(h(t; coeff = c))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k1 * 2 * c)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Float64}
	
        a = Tensor(x, y, z)
        b = @inferred(h(a; coeff = c))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k1 * 2 * c)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Float64}
    end

    h = tensormap(g0, g0, f1)
    @test @inferred(deg(h)) == deg(g0)+deg(g0)+deg(f1)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        a = zero(Linear{Tensor{NTuple{3, String}}, BigFloat})
        t = Tensor(x, y, z)
        b = @inferred(h(t; addto = a))
        @test b === a == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^(k1+k2) * 4)
    end

    h = tensormap(g0, g0, g0)
    @test @inferred(deg(h)) == deg(g0)+deg(g0)+deg(g0)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        c = -5
        a = zero(Linear{Tensor{NTuple{3, String}}, BigInt})
        t = Tensor(x, y, z)
        b = @inferred(h(t; addto = a, coeff = c))
        @test b === a == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^0 * 8 * c)
    end
end

@testset "tensormap deg 1 1" begin
    h = tensormap()
    t = Tensor()
    a = @inferred tensor(; coefftype = Val(Int32))
    @test h(t) == Linear(t => 1)
    @test h(a; coeff = 3) == 3*a
    
    h = tensormap(f1, f1)
    @test @inferred(deg(h)) == deg(f1)+deg(f1)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Int}

        a = tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Int}
    end

    h = tensormap(f1, g1)
    @test @inferred(deg(h)) == deg(f1)+deg(g1)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(g1, f1)
    @test @inferred(deg(h)) == deg(g1)+deg(f1)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = Tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 2)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(g1, g1)
    @test @inferred(deg(h)) == deg(g1)+deg(g1)
    for k1 in 1:3, k2 in 1:3
        x = "x"^k1
        y = "y"^k2
        t = Tensor(x, y)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 4)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
	
        a = Tensor(x, y)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2) => (-1)^k1 * 4)
        @test typeof(b) == Linear{Tensor{NTuple{2, String}}, Float64}
    end

    h = tensormap(f1, f1, f1)
    @test @inferred(deg(h)) == deg(f1)+deg(f1)+deg(f1)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        t = Tensor(x, y, z)
        b = @inferred(h(t))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Int}
	
        a = Tensor(x, y, z)
        b = @inferred(h(a))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Int}
    end

    h = tensormap(f1, f1, g1)
    @test @inferred(deg(h)) == deg(f1)+deg(f1)+deg(g1)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        c = 3
        t = Tensor(x, y, z)
        b = @inferred(h(t; coeff = c))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2 * 2 * c)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Float64}
	
        a = Tensor(x, y, z)
        b = @inferred(h(a; coeff = c))
        @test b == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2 * 2 * c)
        @test typeof(b) == Linear{Tensor{NTuple{3, String}}, Float64}
    end

    h = tensormap(g1, g1, f1)
    @test @inferred(deg(h)) == deg(g1)+deg(g1)+deg(f1)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        a = zero(Linear{Tensor{NTuple{3, String}}, BigFloat})
        t = Tensor(x, y, z)
        b = @inferred(h(t; addto = a))
        @test b === a == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2 * 4)
    end

    h = tensormap(g1, g1, g1)
    @test @inferred(deg(h)) == deg(g1)+deg(g1)+deg(g1)
    for k1 in 1:3, k2 in 1:3, k3 in 1:3
        x = "x"^k1
        y = "y"^k2
        z = "z"^k3
        c = -5
        a = zero(Linear{Tensor{NTuple{3, String}}, BigInt})
        t = Tensor(x, y, z)
        b = @inferred(h(t; addto = a, coeff = c))
        @test b === a == Linear(Tensor("X"^k1, "Y"^k2, "Z"^k3) => (-1)^k2 * 8 * c)
    end
end

struct P
    s::String
end

Base.hash(p::P, h::UInt) = hash(p.s, h)

(p::P)(x) = p.s * x
(p::P)(x, y; addto = zero(Linear{String,Int16}), coeff = ONE) = addmul!(addto, p.s * x * y, coeff*Int16(5))

@testset "lin ext eval" begin
    for R in (Int8, BigInt, BigFloat), S in (Int8, Int32, Float64)
	x, y, u, v = "x", "y", "u", "v"
        a = Linear{P,R}(P(x) => 1, P(y) => 2)
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

import LinearCombinations: diff

function diff(x::String;
        coefftype = Int,
        addto = zero(Linear{String,unval(coefftype)}),
        coeff = 1,
        is_filtered = false) 
    if x[1] != 'd'
        addmul!(addto, 'd' * x, coeff)
    end
    addto
end

@testset "diff String" begin
    x = "xx"
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)

    x = Linear("x" => 3.5, "y" => -1.0, "z" => 5.2)
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)
end

@testset "diff Tensor" begin
    x = Tensor("x", "yy", "zzz")
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)

    a = Linear("x" => 2, "xx" => -1)
    b = Linear("yy" => 1, "yyy" => -3)
    c = Linear("z" => 5, "zzz" => -4)
    
    x = @inferred tensor(a, b, c; coefftype = Val(Float64))
    dx = @inferred diff(x)
    ddx = @inferred diff(dx)
    @test iszero(ddx)

    x = @inferred tensor(a, b, c; coefftype = Val(Float64))  # Int16 doesn't work!
    dx = @inferred diff(x; coeff = -2)
    @test dx isa Linear{Tensor{NTuple{3,String}},Float64}
    @test dx == -2*diff(x)
    
    for n in 0:8
        a = @inferred tensor((string(x) for x in 'a':'a'+n-1)...; coefftype = Val(Int))
        b = @inferred diff(a)
        @test iszero(diff(diff(b)))
    end
end
