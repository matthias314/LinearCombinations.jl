# LinearCombinations.jl

This Julia package allows to work with formal linear combinations and linear maps.
Multilinear maps and tensors are also supported.
The terms appearing in a linear combination can be of any type,
and coefficients can be in any commutative ring with unit.
The overall aim of the package is to provide functions that are efficient and easy to use.

The package comes with an extensive [documentation](https://matthias314.github.io/LinearCombinations.jl/dev/).

## Examples

For simplicity we use `Char` and `String` as term types and integers, rationals or
floating-point numbers as coefficients.

We start with some linear combinations. They are of type [`Linear`](https://matthias314.github.io/LinearCombinations.jl/dev/linear/#LinearCombinations.Linear), which can store
any number of term-coefficient pairs. Term type and coefficient type are automatically
determined to be `Char` and `Int` in the following example.
```julia
julia> a = Linear('x' => 1, 'y' => 2)
x+2*y

julia> typeof(a)
Linear{Char, Int64}

julia> b = Linear('z' => 3, 'w' => -1)
-w+3*z

julia> c = a + 2*b - 'v'
-2*w+x+2*y-v+6*z

julia> c['y'], c['u']
(2, 0)
```
Linear combinations with non-concrete types are also possible.
```julia
julia> p = Linear{AbstractVector,Real}([1,2,3] => 5, 4:6 => 1//2)
1//2*4:6+5*[1, 2, 3]

julia> [4,5,6] - 2*p   # [4,5,6] is equal to 4:6
-10*[1, 2, 3]
```
Next we define a linear function mapping terms to terms. This is done with the help of the macro [`@linear`](https://matthias314.github.io/LinearCombinations.jl/dev/extensions/#LinearCombinations.@linear).
```julia
julia> @linear f; f(x::Char) = uppercase(x)
f (generic function with 2 methods)

julia> f(a)
2*Y+X
```
Another linear function, this time mapping terms to linear combinations.
```julia
julia> @linear g; g(x::Char) = Linear(f(x) => 1, x => -1)
g (generic function with 2 methods)

julia> g(a)
2*Y-x-2*y+X
```
Multiplication is bilinear by default.
Recall that multiplying `Char` or `String` values in Julia means concatenation.
```julia
julia> a * 'w'
xw+2*yw

julia> a * b
-xw+6*yz-2*yw+3*xz
```
The next example is a user-defined bilinear function. Bilinearity is achieved by the macro [`@multilinear`](https://matthias314.github.io/LinearCombinations.jl/dev/extensions/#LinearCombinations.@multilinear).
```julia
julia> @multilinear h; h(x, y) = x*y*x
h (generic function with 2 methods)

julia> h(a, b)
-xwx+3*xzx+12*yzy+6*xzy+6*yzx-2*ywx-2*xwy-4*ywy
```
Here is a user-defined multilinear function with a variable number of arguments.
```julia
julia> @multilinear j; j(x::Char...) = *(x...)
j (generic function with 2 methods)

julia> j(a)
x+2*y

julia> j(a, b)
-xw+6*yz-2*yw+3*xz

julia> j(a, b, a)
-xwx+3*xzx+12*yzy+6*xzy+6*yzx-2*ywx-2*xwy-4*ywy
```
In the following example we define a [tensor](https://matthias314.github.io/LinearCombinations.jl/dev/tensor/#LinearCombinations.Tensor) and swap the two components of each summand.
```julia
julia> t = tensor(a, b)
3*x⊗z-x⊗w-2*y⊗w+6*y⊗z

julia> swap(t)
-2*w⊗y+6*z⊗y-w⊗x+3*z⊗x
```
We finally take the tensor product of the functions `f` and `g` and apply it to `t`.
```julia
julia> k = tensor(f, g)
f⊗g

julia> k(Tensor('x', 'z'))
-X⊗z+X⊗Z

julia> k(t)
-3*X⊗z-X⊗W+2*Y⊗w+6*Y⊗Z+X⊗w-6*Y⊗z+3*X⊗Z-2*Y⊗W
```
