

const INSTANCES_DIR = joinpath(dirname(pathof(ExaPF)), "..", "data")

PROBLEMS = Dict(
    "case57" => joinpath(INSTANCES_DIR, "case57.m"),
    "case118" => joinpath(INSTANCES_DIR, "case118.m"),
    "case300" => joinpath(INSTANCES_DIR, "case300.m"),
    "case1354" => joinpath(INSTANCES_DIR, "case1354.m"),
    "case2869" => joinpath(INSTANCES_DIR, "case2869_pegase.m"),
    "case9241" => joinpath(INSTANCES_DIR, "case9241pegase.m"),
)
