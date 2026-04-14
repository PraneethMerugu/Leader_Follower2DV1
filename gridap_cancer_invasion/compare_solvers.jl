"""
Cancer Invasion Solver Comparison Script v3 - Moderate Adhesion Version

Modified parameters with slightly increased adhesion:
- Increased adhesion (γ=1.0, g=1.5)
- Standard diffusion (δ=d=1.377)
- Moderate chemotaxis (ηy=15)
- High mesh resolution (80x80)

Settings:
- Chemotaxis parameter: ηy = 15
- Diffusion: δ = d = 1.377
- Adhesion: γ = 1.0, g = 1.5 (increased from baseline)
- Time span: t ∈ [0, 0.1]
- Mesh: 80×80 elements
- Snapshots: t = [0.0, 0.025, 0.05, 0.075, 0.1]
"""

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.Algebra
using Gridap.ODEs
using Gridap.MultiField
using LinearAlgebra
using SparseArrays
using DifferentialEquations
using Sundials
using Plots
using Printf
using LaTeXStrings

# Include both solver implementations
include("cancer_invasion_sundials.jl")
include("cancer_invasion_supg.jl")

# Export plot settings
gr()  # Use GR backend
ENV["GKSwstype"] = "nul"  # Set GR to non-interactive mode for headless

# Publication-ready default settings
default(
    fontfamily="Computer Modern",
    linewidth=1.5,
    framestyle=:box,
    grid=false,
    tick_direction=:out,
    markersize=6,
    markerstrokewidth=0.5,
    titlefontsize=11,
    guidefontsize=10,
    tickfontsize=9,
    legendfontsize=9,
    colorbar_titlefontsize=9,
    colorbar_tickfontsize=8,
    margin=5Plots.mm
)

# =============================================================================
# Configuration - MODERATE ADHESION VERSION
# =============================================================================

const CONFIG = (
    ηy=15.0,                    # Chemotaxis parameter (moderately increased)
    δ=1.377,                    # Leader diffusion (baseline)
    d=1.377,                    # Follower diffusion (baseline)
    γ=1.0,                      # Leader adhesion (increased from 0.6048)
    g=1.5,                      # Follower adhesion (increased from 1.1772)
    α=1.0,                      # Leader time scale
    a=0.44,                     # Follower time scale
    tspan=(0.0, 0.1),           # Standard time span
    partition=(80, 80),         # High mesh resolution 80x80
    snapshot_times=[0.0, 0.025, 0.05, 0.075, 0.1],  # Standard snapshots
    reltol=1e-4,                # Tighter tolerance
    abstol=1e-6,                # Tighter absolute tolerance
    dtmax=0.005,                # Standard timestep
    dtmin=1e-10,                # Smaller minimum timestep
    C_supg=0.5,                 # SUPG stabilization constant
)

# =============================================================================
# Solver Comparison Function
# =============================================================================

function run_comparison()
    @info "="^70
    @info "Cancer Invasion Solver Comparison v3 - MODERATE ADHESION"
    @info "  Chemotaxis parameter: ηy = $(CONFIG.ηy)"
    @info "  Diffusion: δ=$(CONFIG.δ), d=$(CONFIG.d)"
    @info "  Adhesion: γ=$(CONFIG.γ), g=$(CONFIG.g) (increased from baseline)"
    @info "  Time span: $(CONFIG.tspan)"
    @info "  Mesh: $(CONFIG.partition) (high resolution)"
    @info "  Snapshots: $(CONFIG.snapshot_times)"
    @info "="^70

    # Common parameters for both solvers
    base_params = (
        α=CONFIG.α, a=CONFIG.a,
        δ=CONFIG.δ, d=CONFIG.d,
        γ=CONFIG.γ, g=CONFIG.g,
        ηy=CONFIG.ηy,
        domain=(0.0, 1.0, 0.0, 1.0),
        partition=CONFIG.partition,
        tspan=CONFIG.tspan,
        reltol=CONFIG.reltol,
        abstol=CONFIG.abstol,
        dtmax=CONFIG.dtmax,
        dtmin=CONFIG.dtmin
    )

    # =========================================================================
    # Run Standard IDA Solver (Galerkin)
    # =========================================================================
    @info ""
    @info "Running Standard IDA Solver (Galerkin)..."

    params_ida = CancerInvasionIDA(; base_params...)

    sol_ida = nothing
    model_ida = nothing
    Ω_ida = nothing
    dΩ_ida = nothing
    U_ida = nothing
    N_ida = nothing
    ida_success = false

    try
        sol_ida, model_ida, Ω_ida, dΩ_ida, U_ida, N_ida = solve_ida(params_ida, verbose=true)

        # Check for solver success - check both symbol and string forms
        rc = sol_ida.retcode
        rc_str = string(rc)
        @info "IDA solver retcode: $(rc) (type: $(typeof(rc)), string: $(rc_str))"
        
        if rc == :Success || rc == :Terminated || occursin("Success", rc_str) || occursin("Terminated", rc_str)
            @info "IDA solver completed successfully"
            @info "  Total time steps: $(length(sol_ida.t))"
            ida_success = true
        else
            @warn "IDA solver failed with retcode: $(rc)"
            ida_success = false
        end
    catch e
        @warn "IDA solver threw exception: $e"
        ida_success = false
    end

    if !ida_success
        @error "IDA solver failed. Cannot run comparison without both solutions."
        @error "Try reducing adhesion parameters or increasing mesh resolution."
        return nothing
    end

    # =========================================================================
    # Run SUPG Solver
    # =========================================================================
    @info ""
    @info "Running SUPG Solver..."

    params_supg = CancerInvasionSUPG(;
        base_params...,
        C=CONFIG.C_supg,
        h=0.0,
        v_epsilon=1e-10
    )

    sol_supg = nothing
    model_supg = nothing
    Ω_supg = nothing
    dΩ_supg = nothing
    U_supg = nothing
    N_supg = nothing
    C_val = nothing

    try
        sol_supg, model_supg, Ω_supg, dΩ_supg, U_supg, N_supg, C_val = solve_supg(params_supg, verbose=true)

        # Check for solver failure
        if sol_supg.retcode != :Success && sol_supg.retcode != :Terminated
            @error "SUPG solver failed with retcode: $(sol_supg.retcode)"
            return nothing
        end

        @info "SUPG solver completed successfully"
        @info "  Total time steps: $(length(sol_supg.t))"
    catch e
        @error "SUPG solver threw exception: $e"
        return nothing
    end

    # =========================================================================
    # Generate Plots
    # =========================================================================
    @info ""
    @info "Generating plots..."

    plot_snapshots(sol_ida, sol_supg, U_ida, U_supg, N_ida, N_supg, model_ida)
    plot_error_evolution(sol_ida, sol_supg, U_ida, U_supg, N_ida, N_supg, dΩ_ida)

    @info ""
    @info "="^70
    @info "Comparison complete!"
    @info "  Plots saved: comparison_heatmaps_t*.png, comparison_errors.png"
    @info "="^70

    return (sol_ida=sol_ida, sol_supg=sol_supg)
end

# =============================================================================
# Snapshot Plotting
# =============================================================================

function plot_snapshots(sol_ida, sol_supg, U_ida, U_supg, N_ida, N_supg, model_ida)
    @info "  Creating heatmap plots..."

    # Use consistent grid size for plotting
    n_plot = 80  # Match mesh resolution
    x_grid = range(0, 1, length=n_plot)
    y_grid = range(0, 1, length=n_plot)

    # Common plot settings for heatmaps
    heatmap_kwargs = Dict(
        :xlims => (0, 1),
        :ylims => (0, 1),
        :aspect_ratio => :equal,
        :colorbar => true,
        :colorbar_titlefont => font(9, "Computer Modern"),
        :framestyle => :box,
        :grid => false,
        :tick_direction => :out,
        :xticks => ([0, 0.25, 0.5, 0.75, 1.0], ["0.0", "0.25", "0.5", "0.75", "1.0"]),
        :yticks => ([0, 0.25, 0.5, 0.75, 1.0], ["0.0", "0.25", "0.5", "0.75", "1.0"]),
        :titlefontsize => 10,
        :guidefontsize => 10,
        :tickfontsize => 8,
        :margin => 3Plots.mm
    )

    for (i, t) in enumerate(CONFIG.snapshot_times)
        @info "    Plotting t=$t..."

        # Get solutions at time t
        u_ida = sol_ida(t)
        u_supg = sol_supg(t)

        ρ_ida_vals = u_ida[1:N_ida]
        f_ida_vals = u_ida[N_ida+1:end]
        ρ_supg_vals = u_supg[1:N_supg]
        f_supg_vals = u_supg[N_supg+1:end]

        ρ_ida = FEFunction(U_ida, ρ_ida_vals)
        f_ida = FEFunction(U_ida, f_ida_vals)
        ρ_supg = FEFunction(U_supg, ρ_supg_vals)
        f_supg = FEFunction(U_supg, f_supg_vals)

        # Sample fields
        ρ_ida_grid = sample_field_fast(ρ_ida, x_grid, y_grid)
        f_ida_grid = sample_field_fast(f_ida, x_grid, y_grid)
        ρ_supg_grid = sample_field_fast(ρ_supg, x_grid, y_grid)
        f_supg_grid = sample_field_fast(f_supg, x_grid, y_grid)

        # Clean data
        ρ_ida_grid = clean_array(ρ_ida_grid)
        ρ_supg_grid = clean_array(ρ_supg_grid)
        f_ida_grid = clean_array(f_ida_grid)
        f_supg_grid = clean_array(f_supg_grid)

        # Color limits - use global min/max across both solvers for consistency
        ρ_min = min(minimum(ρ_ida_grid), minimum(ρ_supg_grid))
        ρ_max = max(maximum(ρ_ida_grid), maximum(ρ_supg_grid))
        f_min = min(minimum(f_ida_grid), minimum(f_supg_grid))
        f_max = max(maximum(f_ida_grid), maximum(f_supg_grid))

        # Ensure valid range for color limits
        if ρ_max <= ρ_min
            ρ_max = ρ_min + 0.001
        end
        if f_max <= f_min
            f_max = f_min + 0.001
        end

        # Format time for display
        t_str = @sprintf("%.4f", t)

        # Create plots with LaTeX-style labels
        # Row 1: Leader density ρ
        p1 = heatmap(collect(x_grid), collect(y_grid), ρ_ida_grid,
            title="Standard Galerkin",
            xlabel="x", ylabel="y",
            seriescolor=:viridis,
            clims=(Float64(ρ_min), Float64(ρ_max)),
            colorbar_title=L"\rho";
            heatmap_kwargs...)

        p2 = heatmap(collect(x_grid), collect(y_grid), ρ_supg_grid,
            title="SUPG (C=0.5)",
            xlabel="x", ylabel="y",
            seriescolor=:viridis,
            clims=(Float64(ρ_min), Float64(ρ_max)),
            colorbar_title=L"\rho";
            heatmap_kwargs...)

        ρ_diff = ρ_supg_grid - ρ_ida_grid
        ρ_dmax = maximum(abs.(ρ_diff))
        if ρ_dmax < 1e-10
            ρ_dmax = 0.001
        end
        p3 = heatmap(collect(x_grid), collect(y_grid), ρ_diff,
            title="Difference",
            xlabel="x", ylabel="y",
            seriescolor=:RdBu,
            clims=(-Float64(ρ_dmax), Float64(ρ_dmax)),
            colorbar_title=L"\Delta\rho";
            heatmap_kwargs...)

        # Row 2: Follower density f
        p4 = heatmap(collect(x_grid), collect(y_grid), f_ida_grid,
            title="",
            xlabel="x", ylabel="y",
            seriescolor=:viridis,
            clims=(Float64(f_min), Float64(f_max)),
            colorbar_title=L"f";
            heatmap_kwargs...)

        p5 = heatmap(collect(x_grid), collect(y_grid), f_supg_grid,
            title="",
            xlabel="x", ylabel="y",
            seriescolor=:viridis,
            clims=(Float64(f_min), Float64(f_max)),
            colorbar_title=L"f";
            heatmap_kwargs...)

        f_diff = f_supg_grid - f_ida_grid
        f_dmax = maximum(abs.(f_diff))
        if f_dmax < 1e-10
            f_dmax = 0.001
        end
        p6 = heatmap(collect(x_grid), collect(y_grid), f_diff,
            title="",
            xlabel="x", ylabel="y",
            seriescolor=:RdBu,
            clims=(-Float64(f_dmax), Float64(f_dmax)),
            colorbar_title=L"\Delta f";
            heatmap_kwargs...)

        # Layout with annotations
        plt = plot(p1, p2, p3, p4, p5, p6,
            layout=@layout([a b c; d e f]),
            size=(1400, 800),
            dpi=300,
            left_margin=5Plots.mm,
            right_margin=5Plots.mm,
            top_margin=5Plots.mm,
            bottom_margin=5Plots.mm)

        # Add overall title using annotation
        plot!(plt, plot_title="Moderate Adhesion Comparison at t = $(t_str), " * L"\eta_y" * " = $(CONFIG.ηy), γ=$(CONFIG.γ), g=$(CONFIG.g)",
            plot_titlefontsize=12,
            plot_titlevspan=0.08)

        fname = @sprintf("comparison_heatmaps_t%04d.png", round(Int, t * 10000))
        savefig(plt, fname)
        sleep(0.5)  # Give time for file write
        @info "      Saved: $fname (size: $(filesize(fname)) bytes)"
    end
end

function clean_array(A)
    """Replace NaN, Inf, and -Inf values with 0.0"""
    result = Float64.(A)
    for i in eachindex(result)
        if !isfinite(result[i])
            result[i] = 0.0
        end
    end
    return result
end

function sample_field_fast(uh::CellField, x_grid, y_grid)
    """Sample field on a regular grid"""
    n_x = length(x_grid)
    n_y = length(y_grid)
    values = zeros(n_y, n_x)

    for j in 1:n_y
        for i in 1:n_x
            p = Point(x_grid[i], y_grid[j])
            try
                values[j, i] = uh(p)
            catch
                values[j, i] = 0.0
            end
        end
    end

    return values
end

# =============================================================================
# Error Plotting
# =============================================================================

function plot_error_evolution(sol_ida, sol_supg, U_ida, U_supg, N_ida, N_supg, dΩ_ida)
    @info "  Creating error evolution plot..."

    times = sol_ida.t
    l2_rho = Float64[]
    l2_f = Float64[]
    linf_rho = Float64[]
    linf_f = Float64[]

    for t in times
        u_ida = sol_ida(t)
        u_supg = sol_supg(t)

        ρ_ida_vals = u_ida[1:N_ida]
        f_ida_vals = u_ida[N_ida+1:end]
        ρ_supg_vals = u_supg[1:N_supg]
        f_supg_vals = u_supg[N_supg+1:end]

        error_rho_vals = ρ_ida_vals - ρ_supg_vals
        error_f_vals = f_ida_vals - f_supg_vals

        # L∞ error
        push!(linf_rho, maximum(abs.(error_rho_vals)))
        push!(linf_f, maximum(abs.(error_f_vals)))

        # L2 error
        error_rho = FEFunction(U_ida, error_rho_vals)
        error_f = FEFunction(U_ida, error_f_vals)

        try
            l2_r = sqrt(sum(∫(error_rho * error_rho)dΩ_ida))
            l2_f_val = sqrt(sum(∫(error_f * error_f)dΩ_ida))
            push!(l2_rho, l2_r)
            push!(l2_f, l2_f_val)
        catch
            push!(l2_rho, NaN)
            push!(l2_f, NaN)
        end
    end

    # Common plot settings for publication quality
    line_kwargs = Dict(
        :lw => 2,
        :marker => :circle,
        :markersize => 5,
        :markerstrokewidth => 0.5
    )

    # Plot L2 error with LaTeX labels
    p1 = plot(times, l2_rho, label=L"\rho~\textrm{(Leader)}",
        xlabel="Time",
        ylabel=L"L^2~\textrm{Error}",
        title=L"L^2~\textrm{Error vs Time}",
        framestyle=:box,
        grid=true,
        gridalpha=0.3,
        minorgrid=true,
        minorgridalpha=0.15;
        line_kwargs...)
    plot!(p1, times, l2_f, label=L"f~\textrm{(Follower)}",
        marker=:square;
        line_kwargs...)

    # Plot L∞ error (log scale) with LaTeX labels
    valid_idx_rho = findall(x -> x > 0, linf_rho)
    valid_idx_f = findall(x -> x > 0, linf_f)

    p2 = plot(times[valid_idx_rho], linf_rho[valid_idx_rho],
        label=L"\rho~\textrm{(Leader)}",
        xlabel="Time",
        ylabel=L"L^\infty~\textrm{Error}",
        title=L"L^\infty~\textrm{Error vs Time}",
        yscale=:log10,
        framestyle=:box,
        grid=true,
        gridalpha=0.3,
        minorgrid=true,
        minorgridalpha=0.15;
        line_kwargs...)
    plot!(p2, times[valid_idx_f], linf_f[valid_idx_f],
        label=L"f~\textrm{(Follower)}",
        marker=:square;
        line_kwargs...)

    plt = plot(p1, p2,
        layout=(1, 2),
        size=(1100, 450),
        dpi=300,
        left_margin=8Plots.mm,
        right_margin=5Plots.mm,
        top_margin=5Plots.mm,
        bottom_margin=8Plots.mm,
        plot_title="Moderate Adhesion Error Evolution: SUPG vs Standard Galerkin, " *
                   L"\eta_y" * " = $(CONFIG.ηy), γ=$(CONFIG.γ), g=$(CONFIG.g)",
        plot_titlefontsize=11,
        plot_titlevspan=0.08)

    savefig(plt, "comparison_errors.png")
    @info "    Saved: comparison_errors.png"

    # Print summary with formatted numbers
    @info ""
    @info "Error Summary:"
    @info "  Max L² error (ρ): $(@sprintf("%.6e", maximum(filter(isfinite, l2_rho))))"
    @info "  Max L² error (f): $(@sprintf("%.6e", maximum(filter(isfinite, l2_f))))"
    @info "  Max L∞ error (ρ): $(@sprintf("%.6e", maximum(linf_rho)))"
    @info "  Max L∞ error (f): $(@sprintf("%.6e", maximum(linf_f)))"
    @info "  Final L² error (ρ): $(@sprintf("%.6e", l2_rho[end]))"
    @info "  Final L² error (f): $(@sprintf("%.6e", l2_f[end]))"
end

# =============================================================================
# Main Execution
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Starting solver comparison v3 with moderate adhesion..."

    result = run_comparison()

    if isnothing(result)
        @error "Comparison failed! Check logs above."
        exit(1)
    else
        @info ""
        @info "Success! All plots generated."
    end
end
