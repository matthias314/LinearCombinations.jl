# LinearCombinations.jl

This Julia package allows to work with formal linear combinations and linear maps.
Multilinear maps and tensors are also supported.
The terms appearing in a linear combination can be of any type,
and coefficients can be in any commutative ring with unit.
The overall aim of the package is to provide functions that are efficient and easy to use.

The package comes with an extensive [documentation](https://matthias314.github.io/LinearCombinations.jl/stable/).

## Examples

For simplicity we use `Char` and `String` as term types and integers, rationals or
floating-point numbers as coefficients.

We start with some linear combinations. They are of type [`Linear`](https://matthias314.github.io/LinearCombinations.jl/stable/linear/#LinearCombinations.Linear), which can store
any number of term-coefficient pairs. Term type and coefficient type are automatically
determined to be `Char` and `Int` in the following example.
```julia
julia> a = Linear('x' => 1, 'y' => 2)
Linear{Char, Int64} with 2 terms:
'x'+2*'y'

julia> b = Linear('z' => 3, 'w' => -1)
Linear{Char, Int64} with 2 terms:
-'w'+3*'z'

julia> c = a + 2*b - 'v'
Linear{Char, Int64} with 5 terms:
-2*'w'+'x'+2*'y'-'v'+6*'z'

julia> c['y'], c['u']
(2, 0)
```
Linear combinations with non-concrete types are also possible.
```julia
julia> p = Linear{AbstractVector,Real}([1,2,3] => 5, 4:6 => 1//2)
Linear{AbstractVector, Real} with 2 terms:
1//2*4:6+5*[1, 2, 3]

julia> [4,5,6] - 2*p   # [4,5,6] is equal to 4:6
Linear{AbstractVector, Real} with 1 term:
-10*[1, 2, 3]
```
Next we define a linear function mapping terms to terms. This is done with the help of the macro [`@linear`](https://matthias314.github.io/LinearCombinations.jl/stable/extensions/#LinearCombinations.@linear).
```julia
julia> @linear f; f(x::Char) = uppercase(x)
f (generic function with 2 methods)

julia> f(a)
Linear{Char, Int64} with 2 terms:
2*'Y'+'X'
```
Another linear function, this time mapping terms to linear combinations.
```julia
julia> @linear g; g(x::Char) = Linear(f(x) => 1, x => -1)
g (generic function with 2 methods)

julia> g(a)
Linear{Char, Int64} with 4 terms:
2*'Y'-'x'-2*'y'+'X'
```
Multiplication is bilinear by default.
Recall that multiplying `Char` or `String` values in Julia means concatenation.
```julia
julia> a * 'w'
Linear{String, Int64} with 2 terms:
"xw"+2*"yw"

julia> a * b
Linear{String, Int64} with 4 terms:
-"xw"+6*"yz"-2*"yw"+3*"xz"
```
The next example is a user-defined bilinear function. Bilinearity is achieved by the macro [`@multilinear`](https://matthias314.github.io/LinearCombinations.jl/stable/extensions/#LinearCombinations.@multilinear).
```julia
julia> @multilinear h; h(x, y) = x*y*x
h (generic function with 2 methods)

julia> h(a, b)
Linear{String, Int64} with 8 terms:
-"xwx"+3*"xzx"+12*"yzy"+6*"xzy"+6*"yzx"-2*"ywx"-2*"xwy"-4*"ywy"
```
Here is a user-defined multilinear function with a variable number of arguments.
```julia
julia> @multilinear j; j(x::Char...) = *(x...)
j (generic function with 2 methods)

julia> j(a)
Linear{String, Int64} with 2 terms:
"x"+2*"y"

julia> j(a, b)
Linear{String, Int64} with 4 terms:
-"xw"+6*"yz"-2*"yw"+3*"xz"

julia> j(a, b, a)
Linear{String, Int64} with 8 terms:
-"xwx"+3*"xzx"+12*"yzy"+6*"xzy"+6*"yzx"-2*"ywx"-2*"xwy"-4*"ywy"
```
In the following example we define a [tensor](https://matthias314.github.io/LinearCombinations.jl/stable/tensor/#LinearCombinations.Tensor) and swap the two components of each summand.
```julia
julia> t = tensor(a, b)
Linear{Tensor{Tuple{Char, Char}}, Int64} with 4 terms:
-'x'⊗'w'-2*'y'⊗'w'+3*'x'⊗'z'+6*'y'⊗'z'

julia> swap(t)
Linear{Tensor{Tuple{Char, Char}}, Int64} with 4 terms:
-'w'⊗'x'+3*'z'⊗'x'+6*'z'⊗'y'-2*'w'⊗'y'
```
We finally take the tensor product of the functions `f` and `g` and apply it to `t`.
```julia
julia> k = tensor(f, g)
Linear{Tensor{Tuple{typeof(f), typeof(g)}}, Int64} with 1 term:
f⊗g

julia> k(Tensor('x', 'z'))
Linear{Tensor{Tuple{Char, Char}}, Int64} with 2 terms:
-'X'⊗'z'+'X'⊗'Z'

julia> k(t)
Linear{Tensor{Tuple{Char, Char}}, Int64} with 8 terms:
-3*'X'⊗'z'+2*'Y'⊗'w'+3*'X'⊗'Z'+'X'⊗'w'-'X'⊗'W'-2*'Y'⊗'W'-6*'Y'⊗'z'+6*'Y'⊗'Z'
```
