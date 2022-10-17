```@meta
CurrentModule = Argos
```

# Evaluator API

## Description

```@docs
AbstractNLPEvaluator
```


## API Reference

### Optimization
```@docs
optimize!
```

### Attributes
```@docs
Variables
Constraints
n_variables
n_constraints
constraints_type

```

## Callbacks


```@docs
update!
objective
gradient!
constraint!
jacobian!
jacobian_coo!
jprod!
jtprod!
ojtprod!
hessian!
hessian_coo!
hessprod!
hessian_lagrangian_prod!
hessian_lagrangian_penalty_prod!

```


### Utilities

```@docs
reset!
```

