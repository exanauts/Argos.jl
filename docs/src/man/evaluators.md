# Evaluators

Argos wraps [ExaPF.jl](https://github.com/exanauts/ExaPF.jl/) to gather the objective and the constraints
associated to a particular OPF problem inside an `Evaluator`.
The two principal evaluators are the `ReducedSpaceEvaluator`
and the `FullSpaceEvaluator`.


## Overview of the AbstractNLPEvaluator

An [`Argos.AbstractNLPEvaluator`](@ref) implements the callbacks
associated to a given optimization problem:
associated with an underlying `ExaPF.AbstractFormulation`:
```math
\min_{u \in \mathbb{R}^n} \;              f(u)
\quad \text{subject to}\quad
\left\{
\begin{aligned}
 & g(u) = 0 \\
                        & h(u) \leq 0.
\end{aligned}
\right.
```
with $f: \mathbb{R}^n \to \mathbb{R}$ the objective function,
$g: \mathbb{R}^n \to \mathbb{R}^{m_E}$ non-linear equality constraints and
$h: \mathbb{R}^n \to \mathbb{R}^{m_I}$ non-linear inequality constraints.

By design, the implementation
of [`Argos.AbstractNLPEvaluator`](@ref) shares a similar spirit with the implementations
introduced in other packages, such as

- MathOptInterface.jl's [AbstractNLPEvaluator](https://jump.dev/MathOptInterface.jl/stable/apireference/#MathOptInterface.AbstractNLPEvaluator)
- NLPModels.jl's [AbstractNLPModel](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/api/#AbstractNLPModel-functions)

Internally, the evaluator caches all the information needed to evaluate
the callbacks (e.g. the polar representation of the problem, with voltage
magnitudes and angles). This cache allows to reduce the number of memory allocations to
a minimum.
Once a new variable $u$ passed to the evaluator
a function `Argos.update!` is being called to update the cache,
according to the model specified in the underlying `ExaPF.AbstractFormulation`.
Denoting by `nlp` an instance of AbstractNLPEvaluator, the cache is
updated via
```julia-repl
julia> Argos.update!(nlp, u)
```

Once the internal structure updated, we are ready to call the different
callbacks, in every order. For instance, computing the objective, the
gradient and the constraints amounts to
```julia-repl
# Objective
julia> obj = Argos.objective(nlp, u)
# Gradient
julia> g = zeros(n_variables(nlp))
julia> Argos.gradient!(nlp, g, u)
# Constraints
julia> cons = zeros(n_constraints(nlp))
julia> Argos.constraint!(nlp, cons, u)

```


## A journey to the reduced space with the ReducedSpaceEvaluator

When we aim at optimizing the problem directly in the powerflow
manifold, the [`Argos.ReducedSpaceEvaluator`](@ref) is our workhorse.
We recall that the powerflow manifold is defined implicitly by the
powerflow equations:
```math
    g(x(u), u) = 0.
```
By design, the [`Argos.ReducedSpaceEvaluator`](@ref) works in the reduced
space $(x(u), u)$. Hence, the reduced optimization problem writes out
```math
\begin{aligned}
\min_{u \in \mathbb{R}^n} \; & f(x(u), u) \\
\text{subject to} \quad      & h(x(u), u) \leq 0.
\end{aligned}
```
This formulation comes with two advantages:

- if the dimension of the state is large, the reduced problem has
  a lower dimension.
- the powerflow equality constraints $g(x, u) = 0$ disappear in the reduced problem.


### Playing with the ReducedSpaceEvaluator

#### Constructor
To create a [`Argos.ReducedSpaceEvaluator`](@ref), we just need a polar formulation
`polar::PolarForm`:
```julia-repl
julia> nlp = Argos.ReducedSpaceEvaluator(polar)

```
or we could alternatively instantiate the evaluator directly from
a MATPOWER (or PSSE) instance:
```julia-repl
julia> datafile = "case9.m"
julia> nlp = Argos.ReducedSpaceEvaluator(datafile)
A ReducedSpaceEvaluator object
    * device: KernelAbstractions.CPU()
    * #vars: 5
    * #cons: 10
    * constraints:
        - voltage_magnitude_constraints
        - active_power_constraints
        - reactive_power_constraints
    * linear solver: ExaPF.LinearSolvers.DirectSolver()

```

Let's describe the output of the last command.

* `device: KernelAbstractions.CPU()`: the evaluator is instantiated on the CPU ;
* `#vars: 5`: it has 5 optimization variables ;
* `#cons: 10`: and 10 inequality constraints ;
* `constraints`: by default, `nlp` comes with three inequality constraints: `voltage_magnitude_constraints` (specifying the bounds $x_L \leq x(u) \leq x_U$ on the state $x$), `active_power_constraints` and `reactive_power_constraints` (bounding the active and reactive power of the generators) ;
* `linear solver`: `ExaPF.LinearSolvers.DirectSolver`: to solve the linear systems, the evaluator uses a direct linear algebra solver.

Of course, these settings are only specified by default. The user is free
to choose the parameters she wants. For instance,

* We could remove all constraints by passing an empty array of constraints
  to the evaluator:
  ```julia-repl
  julia> constraints = Function[]
  julia> nlp = Argos.ReducedSpaceEvaluator(datafile; constraints=constraints)
  ```
* We could load the evaluator on the GPU simply by changing the `device` option:
  ```julia-repl
  julia> nlp = Argos.ReducedSpaceEvaluator(datafile; device=CUDADevice())
  ```



#### Caching

To juggle between the mathematical description (characterized
by a state $x$ and a control $u$) and the physical description (characterized
by the voltage and power injection at each bus), the evaluator `nlp`
stores internally a cache `nlp.buffer`, with type `ExaPF.AbstractNetworkBuffer`.
```julia-repl
julia> buffer = get(nlp, ExaPF.PhysicalState())
```

#### Evaluation of the callbacks

Now that we have a `nlp` evaluator available, we can embed it in any
optimization routine. For instance, suppose we have a new control `uk`
available. First, we need to find the corresponding state `xk`,
such that ``g(x_k, u_k) = 0``.
In the evaluator's API, this sums up to:
```julia-repl
Argos.update!(nlp, uk)

```
The function `update!` will
- Feed the physical description `nlp.buffer` with the values stored in the new control `uk`.
- Solve the powerflow equations corresponding to the formulation specified in `form`. This operation
  updates the cache `nlp.buffer` inplace.

Once the function `update!` called (and only after that), we can evaluate
all the different callbacks, independently of one other.

* Objective
  ```julia-repl
  julia> cost = Argos.objective(nlp, uk)
  ```
* Objective's gradient
  ```julia-repl
  julia> g = zeros(n_variables(nlp))
  julia> Argos.gradient!(nlp, g, uk)
  ```
* Constraints
  ```julia-repl
  # Evaluate constraints
  julia> cons = zeros(n_constraints(nlp))
  julia> Argos.constraint!(nlp, cons, uk)
  ```
* Constraints' Jacobian
  ```julia-repl
  ## Evaluate Jacobian
  julia> Argos.jacobian!(nlp, jac, uk)
  ```
* Constraints' Jacobian-vector product:
  ```julia-repl
  ## Evaluate Jacobian-vector product
  julia> v = zeros(n_variables(nlp))
  julia> jv = zeros(n_constraints(nlp))
  julia> Argos.jprod!(nlp, jv, uk, v)
  ```
* Constraints' transpose Jacobian-vector product
  ```julia-repl
  ## Evaluate transpose Jacobian-vector product
  julia> v = zeros(n_constraints(nlp))
  julia> jv = zeros(n_variables(nlp))
  julia> Argos.jtprod!(nlp, jv, uk, v)
  ```
* Hessian-vector product:
  ```julia-repl
  ## Evaluate transpose Jacobian-vector product
  julia> v = zeros(n_variables(nlp))
  julia> hv = zeros(n_variables(nlp))
  julia> Argos.hessprod!(nlp, hv, uk, v)
  ```
* Hessian:
  ```julia-repl
  ## Evaluate transpose Jacobian-vector product
  julia> H = zeros(n_variables(nlp), n_variables(nlp))
  julia> Argos.hessprod!(nlp, H, uk)
  ```

!!! note
    Once the powerflow equations solved in a `update!` call, the solution ``x_k`` is stored implicitly in `nlp.buffer`. These values will be used as a starting point for the next resolution of powerflow equations.


