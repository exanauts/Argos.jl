
const CONTINGENCIES_DIR = joinpath(dirname(@__FILE__), "contingencies")

function filter_islanding(network, contingencies)
    nbus, nlines = PS.get(network, PS.NumberOfBuses()), PS.get(network, PS.NumberOfLines())

    graph = Graph()
    add_vertices!(graph, nbus)

    f, t = network.lines.from_buses, network.lines.to_buses
    for (i, j) in zip(f, t)
        add_edge!(graph, i, j)
    end
    @assert is_connected(graph)
    screened = Int[]
    for (k, c) in enumerate(contingencies)
        i, j = f[c], t[c]
        rem_edge!(graph, i, j)
        if is_connected(graph)
            push!(screened, c)
        end
        add_edge!(graph, i, j)
    end
    return screened
end

function filter_infeasible(network, contingencies)
    is_feasible = Int[]
    for line in contingencies
        println("Screen contingency $(line) [Total: $(length(contingencies))]")
        # Drop line from power network
        post = PS.PowerNetwork(network)
        post.lines.Yff[line] = 0.0im
        post.lines.Yft[line] = 0.0im
        post.lines.Ytf[line] = 0.0im
        post.lines.Ytt[line] = 0.0im

        model = ExaPF.PolarForm(post, ExaPF.CPU())
        nlp = Argos.FullSpaceEvaluator(model)

        ips = MadNLP.MadNLPSolver(
            Argos.OPFModel(nlp);
            linear_solver=Ma27Solver,
            max_iter=500, tol=1e-5,
            print_level=MadNLP.ERROR,
        )
        MadNLP.solve!(ips)

        if ips.status == MadNLP.SOLVE_SUCCEEDED
            push!(is_feasible, line)
        end
    end
    return is_feasible
end

function screen_contingencies(model::PolarForm)
    nlines = PS.get(model, PS.NumberOfLines())

    all_lines = 1:nlines
    cont1 = filter_islanding(model.network, all_lines)
    cont2 = filter_infeasible(model.network, cont1)
    ncont = length(cont2)
    println("Keep $(ncont) contingencies out of $(nlines).")
    return cont2
end

function generate_contingencies(case)
    instance = split(case, ".")[1]
    datafile = joinpath("/home/fpacaud/dev/matpower/data", case)
    model = ExaPF.PolarForm(datafile, CPU())
    contingencies = screen_contingencies(model)
    writedlm(joinpath(CONTINGENCIES_DIR, "$(instance).Ctgs"), contingencies)
end

