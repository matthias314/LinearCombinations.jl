using Documenter
# using DocumenterMarkdown

include("setup.jl")

const doctestsetup = quote
    using LinearCombinations
    if LinearCombinations.termcoeff('X' => 1) != ('X' => 1)
        LinearCombinations.termcoeff(xc::Pair{Char}) = xc
    end
end

DocMeta.setdocmeta!(LinearCombinations, :DocTestSetup, doctestsetup; recursive = true)

makedocs(sitename = "LinearCombinations.jl",
    modules = [LinearCombinations],
    pages = [
        "index.md",
        "linear.md",
        "basis.md",
        "tensor.md",
        "extensions.md",
        "basics.md",
        "internals.md",
    ],
    format = Documenter.HTML(),
    doctest = isempty(ARGS) ? true : Symbol(ARGS[1]),  # "false", "only" and "fix" are useful
    warnonly = true)
