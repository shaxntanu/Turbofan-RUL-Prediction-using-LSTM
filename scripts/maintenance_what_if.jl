const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(ROOT_DIR, "data", "maintenance_what_if_inputs.csv")
const DEFAULT_OUTPUT = joinpath(ROOT_DIR, "web", "public", "results", "maintenance_what_if_scenarios.csv")
const URGENCY_THRESHOLDS = (15, 30, 45, 60, 90)
const CAPACITY_LEVELS = (0, 5, 10, 20, 30, 50, 75, 100)
const POLICIES = ("point_estimate", "conservative_interval")

struct EnginePrediction
    dataset::String
    model::String
    unit::Int
    rul_true::Float64
    rul_predicted::Float64
    interval_lower::Float64
    interval_upper::Float64
end

function load_predictions(path::AbstractString)
    isfile(path) || error("Prediction input not found: $path")
    lines = readlines(path)
    isempty(lines) && error("Prediction input is empty: $path")
    names = String.(strip.(split(first(lines), ',')))
    columns = Dict(name => index for (index, name) in enumerate(names))
    required = ("dataset", "model", "unit", "rul_true", "rul_predicted", "interval_lower", "interval_upper")
    missing = filter(name -> !haskey(columns, name), required)
    isempty(missing) || error("Prediction input is missing columns: $(join(missing, ", "))")

    records = EnginePrediction[]
    seen = Set{Tuple{String, String, Int}}()
    for (index, line) in enumerate(lines[2:end])
        line_number = index + 1
        isempty(strip(line)) && continue
        fields = strip.(split(line, ','))
        length(fields) == length(names) || error("Malformed CSV row at line $line_number")
        dataset = String(fields[columns["dataset"]])
        model = String(fields[columns["model"]])
        unit = parse(Int, fields[columns["unit"]])
        truth = parse(Float64, fields[columns["rul_true"]])
        prediction = parse(Float64, fields[columns["rul_predicted"]])
        lower = parse(Float64, fields[columns["interval_lower"]])
        upper = parse(Float64, fields[columns["interval_upper"]])
        all(isfinite, (truth, prediction, lower, upper)) || error("Non-finite prediction for $dataset engine $unit")
        0 <= lower <= upper || error("Invalid interval for $dataset engine $unit")
        key = (dataset, model, unit)
        key in seen && error("Duplicate prediction for $dataset engine $unit")
        push!(seen, key)
        push!(records, EnginePrediction(dataset, model, unit, truth, prediction, lower, upper))
    end
    isempty(records) && error("Prediction input contains no engine records")
    return records
end

policy_score(engine::EnginePrediction, policy::String) =
    policy == "point_estimate" ? engine.rul_predicted : engine.interval_lower

function score_scenario(engines, threshold::Int, capacity_pct::Int, policy::String)
    engine_count = length(engines)
    slot_limit = fld(engine_count * capacity_pct, 100)
    eligible = filter(engine -> policy_score(engine, policy) <= threshold, engines)
    sort!(eligible; by=engine -> (policy_score(engine, policy), engine.rul_predicted, engine.unit))
    scheduled = eligible[1:min(length(eligible), slot_limit)]
    scheduled_units = Set(engine.unit for engine in scheduled)
    urgent = filter(engine -> engine.rul_true <= threshold, engines)
    urgent_serviced = count(engine -> engine.unit in scheduled_units, urgent)
    missed_urgent = length(urgent) - urgent_serviced
    unnecessary_service = length(scheduled) - urgent_serviced
    recall = isempty(urgent) ? "NA" : string(round(urgent_serviced / length(urgent); digits=6))
    precision = isempty(scheduled) ? "NA" : string(round(urgent_serviced / length(scheduled); digits=6))
    utilization = slot_limit == 0 ? "NA" : string(round(length(scheduled) / slot_limit; digits=6))

    return (
        slot_limit,
        engine_count,
        length(urgent),
        length(eligible),
        length(scheduled),
        urgent_serviced,
        missed_urgent,
        unnecessary_service,
        recall,
        precision,
        utilization,
    )
end

function generate_scenarios(input_path::AbstractString, output_path::AbstractString)
    records = load_predictions(input_path)
    groups = Dict{Tuple{String, String}, Vector{EnginePrediction}}()
    for engine in records
        push!(get!(groups, (engine.dataset, engine.model), EnginePrediction[]), engine)
    end

    mkpath(dirname(output_path))
    open(output_path, "w") do file
        println(file, "dataset,model,seed,urgency_threshold_cycles,capacity_pct,slot_limit,engine_count,urgent_engine_count,eligible_engines,scheduled_engines,urgent_serviced,missed_urgent,unnecessary_service,urgent_recall,service_precision,capacity_utilization,policy")
        for (dataset, model) in sort!(collect(keys(groups)))
            engines = groups[(dataset, model)]
            for threshold in URGENCY_THRESHOLDS, capacity in CAPACITY_LEVELS, policy in POLICIES
                metrics = score_scenario(engines, threshold, capacity, policy)
                row = (dataset, model, 42, threshold, capacity, metrics..., policy)
                println(file, join(string.(row), ','))
            end
        end
    end
    return length(groups) * length(URGENCY_THRESHOLDS) * length(CAPACITY_LEVELS) * length(POLICIES)
end

function main(args)
    length(args) <= 2 || error("Usage: julia scripts/maintenance_what_if.jl [input.csv] [output.csv]")
    input_path = isempty(args) ? DEFAULT_INPUT : abspath(args[1])
    output_path = length(args) < 2 ? DEFAULT_OUTPUT : abspath(args[2])
    row_count = generate_scenarios(input_path, output_path)
    println("Wrote $row_count maintenance what-if scenarios to $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
