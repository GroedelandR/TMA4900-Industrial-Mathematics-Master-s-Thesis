from typing import Optional, Any, Literal, Union
import numpy as np
from scipy.stats import lognorm, multivariate_normal, norm
from scipy.special import logit
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from plotnine import (
    ggplot, aes, geom_line, labs, theme_minimal, geom_ribbon, geom_point,
    geom_density, scale_y_sqrt, geom_violin, geom_boxplot, scale_colour_hue, 
    theme, annotate, geom_path, facet_wrap, coord_equal, theme_bw, labeller,
    geom_area, geom_vline, coord_cartesian, scale_x_sqrt, element_blank,
    scale_x_log10, facet_grid
)
from contourpy import contour_generator, LineType
from itertools import product

tfd = tfp.distributions





def plot(
    config: dict[str, Any],
    Simulator: tf.Module,
    LatentStatePrior: tf.Module,
    GraphPrior: tf.Module,
    graph_encoding: Optional[tf.Tensor] = None,
    num_aggregate_samples: Optional[int] = None,
    folder: Optional[str] = None,
) -> None:
    
    if num_aggregate_samples is None:
        num_aggregate_samples = int(
            config["utils"]["plot"]["number_of_aggregate_samples"]
        )
    num_aggregate_samples = tf.constant(num_aggregate_samples, dtype=tf.int32)

    simulator = Simulator(config, num_aggregate_samples=num_aggregate_samples)
    latent_state_prior = LatentStatePrior(config)
    graph_prior = GraphPrior(config)
    
    R = tf.constant(config["general"]["number_of_reservoirs"], dtype=tf.int32)
    time_grid = tf.linspace(
        0.0, 
        config["utils"]["plot"]["max_time_simulation"], 
        int(config["utils"]["plot"]["grid_resolution"])
    )
    
    seed = tf.constant(config["general"]["seed"], dtype=tf.int32)
    if graph_encoding is None:
        latent_states_seed, graph_seed = tfp.random.split_seed(seed)
        latent_states = latent_state_prior.sample(
            num_samples=num_aggregate_samples,
            seed=latent_states_seed
        )        
        graph_encoding = graph_prior.sample(
            num_samples=num_aggregate_samples,
            seed=graph_seed
        )
    else:
        latent_states = latent_state_prior.sample(
                    num_samples=num_aggregate_samples,
                    seed=seed
                )
    
    states = simulator._get_states(
        latent_states=latent_states,
        graph_encoding=graph_encoding
    )
    events = simulator._simulate_events(
        states=states,
        graph_encoding=graph_encoding
    )
    
    def density_plot(
        times: Literal["entry_times", "migration_times"],
        events: dict[str, tf.Tensor]
    ) -> ggplot:
        
        idx = pd.MultiIndex.from_product(
            iterables=[
                tf.range(num_aggregate_samples).numpy(), 
                tf.range(1, R+1).numpy()
            ],
            names=["samples", "reservoir"]
        )

        df = (
            pd.DataFrame({times: events[times].numpy().ravel()}, index=idx)
            .reset_index()
            .loc[lambda df: np.isfinite(df[times])]
        )
        labels = {
            "entry_times": r"$T_i^{\mathrm{ent}} \ [year]$", 
            "migration_times": r"$T_i^{\mathrm{mig}} \ [year]$", 
        }
        
        return (
            ggplot(df, aes(x=times, colour="factor(reservoir)")) +
            geom_density(bw=10, alpha=0.5) +
            theme_minimal(base_size=18) +
            scale_y_sqrt() +
            coord_cartesian(xlim=(0, 
                            config["utils"]["plot"]["max_time_density_plot"])) +
            labs(x=labels[times], y="Density", colour="Reservoir")
        )        
    
    def quantity_over_time_plot(
        quantity: Literal["scaled_volume", "volume", "scaled_height", "height"],
        events: dict[str, tf.Tensor]
    ) -> list[ggplot]:
        
        interpolation = simulator._interpolate(
            events=events,
            times=time_grid
        )
        
        match quantity:
            case "volume":
                interpolation = (
                    config["physical"]["cubic_meter_to_barrels_of_oil"]
                    * 1e-6
                    * simulator._scaled_volume_to_volume(
                        scaled_volume=interpolation,
                        log_lateral_parameter=states["log_lateral_parameter"],
                        log_vertical_parameter=states["log_vertical_parameter"]
                    )
                )
            case "scaled_height":
                interpolation = simulator._scaled_volume_to_scaled_height(
                   scaled_volume=interpolation
                )
            case "height":
                interpolation = simulator._scaled_height_to_height(
                    scaled_height=simulator._scaled_volume_to_scaled_height(
                        scaled_volume=interpolation
                    ),
                    log_vertical_parameter=states["log_vertical_parameter"]
                )
        
        quantiles = tfp.stats.percentile(
            interpolation,
            q=[5.0, 50.0, 95.0],
            axis=0,
            interpolation="linear"
        )
        lower, median, upper = tf.unstack(quantiles)
         
        idx = pd.MultiIndex.from_product(
            iterables=[time_grid.numpy(), tf.range(1, R+1).numpy()],
            names=["years", "reservoir"]
        )
        df = pd.DataFrame({
            "median": median.numpy().ravel(),
            "lower": lower.numpy().ravel(),
            "upper": upper.numpy().ravel(),
        }, index=idx).reset_index()
        
        labels = {
            "scaled_volume": r"$y_i(t)$", 
            "scaled_height": r"$x_i(t)$", 
            "volume": r"$V_i(t) \ \text{[MMbbl]}$", 
            "height": r"$h_i(t) \ \text{[m]}$"
        }
        
        max_plot_time = (
            config["utils"]["plot"]["max_time_height_over_time_plot"]
        )
        plots = []
        plots.append(
            ggplot(df, aes(x="years", colour="factor(reservoir)")) +
            facet_wrap("~reservoir", scales="fixed", nrow=3) +
            geom_line(aes(y="median"), size=0.8) +
            geom_ribbon(aes(ymin="lower", ymax="upper",
                            fill="factor(reservoir)", ), 
                        alpha=0.2, linetype="None", show_legend=False) +
            geom_line(aes(y="lower"), linetype="dashed", size=1,
                      show_legend=False) +
            geom_line(aes(y="upper"), linetype="dashed", size=1,
                      show_legend=False) +
            theme_minimal(base_size=30) +
            scale_x_sqrt(breaks=[0, 2.5, 10, 25, 45, 70, 100],
                         labels=lambda l: [f"{x:g}" for x in l]) +
            coord_cartesian(xlim=(0, max_plot_time)) +
            labs(x=r"$t \ [year]$", y=labels[quantity], colour="Reservoir") +
            theme(strip_text=element_blank(), figure_size=(2*6.4, 3*4.8))
        )   
        plots.append(
            ggplot(df, aes(x="years", colour="factor(reservoir)")) +
            geom_line(aes(y="median"), size=0.8) +
            geom_line(aes(y="lower"), linetype="dashed", alpha=0.6) +
            geom_line(aes(y="upper"), linetype="dashed", alpha=0.6) +
            geom_ribbon(aes(ymin="lower", ymax="upper", 
                            fill="factor(reservoir)"), 
                        alpha=0.02, linetype="dashed", show_legend=False) +
            theme_minimal() +
            scale_x_sqrt(breaks=[0, 2.5, 10, 25, 45, 70, 100],
                         labels=lambda l: [f"{x:g}" for x in l]) +
            coord_cartesian(xlim=(0, max_plot_time)) +
            labs(x=r"$t \ [year]$", y=labels[quantity], colour="Reservoir")
        )
        return plots
        
    def quantity_at_failure_time_plot(
        quantity: Literal["scaled_volume", "volume", "scaled_height", "height"],
        events: dict[str, tf.Tensor]
    ) -> ggplot:
        
        interpolation = simulator._interpolate(
            events=events,
            times=events["migration_times"][:, -1][:, None]
        )
        
        match quantity:
            case "volume":
                interpolation = (
                    config["physical"]["cubic_meter_to_barrels_of_oil"]
                    * 1e-6
                    * simulator._scaled_volume_to_volume(
                        scaled_volume=interpolation,
                        log_lateral_parameter=states["log_lateral_parameter"],
                        log_vertical_parameter=states["log_vertical_parameter"]
                    )
                )
            case "scaled_height":
                interpolation = simulator._scaled_volume_to_scaled_height(
                   scaled_volume=interpolation
                )
            case "height":
                interpolation = simulator._scaled_height_to_height(
                    scaled_height=simulator._scaled_volume_to_scaled_height(
                        scaled_volume=interpolation
                    ),
                    log_vertical_parameter=states["log_vertical_parameter"]
                )
    
        idx = pd.MultiIndex.from_product(
            iterables=[
                tf.range(num_aggregate_samples).numpy(), 
                tf.range(1, R+1).numpy()
            ],
            names=["samples", "reservoir"]
        )
        
        df = pd.DataFrame({
            "quantity": interpolation.numpy().ravel()
        }, index=idx).reset_index()
        
        df_violin = df.loc[
            df.groupby("reservoir")["quantity"]
            .transform("nunique") > 1
        ]
        
        labels = {
                    "scaled_volume": r"$y_i(t)$", 
                    "scaled_height": r"$x_i(t)$", 
                    "volume": r"$V_i(t) \ \text{[MMbbl]}$", 
                    "height": r"$h_i(t) \ \text{[m]}$"
                }
            
        labels = {
            "scaled_volume": rf"$y_i(T_{R}^{{\mathrm{{mig}}}})$", 
            "scaled_height": rf"$x_i(T_{R}^{{\mathrm{{mig}}}})$", 
            "volume": rf"$V_i(T_{R}^{{\mathrm{{mig}}}}) \ \text{{[MMbbl]}}$", 
            "height": rf"$h_i(T_{R}^{{\mathrm{{mig}}}}) \ \text{{[m]}}$"
        }
        
        plot = (
            ggplot(df, aes(x="factor(reservoir)", y="quantity", 
                           fill="factor(reservoir)")) +
            geom_violin(alpha=0.4, trim=True, data=df_violin, 
                        show_legend=False) +
            geom_boxplot(width=0.15, outlier_alpha=1, show_legend=False) +
            theme_minimal(base_size=18) +
            labs(x="Reservoir", y=labels[quantity], fill="Reservoir")
        )
        if quantity in ("volume", "height"):
            max_height = (
                config["utils"]["plot"]["max_height_height_at_failure_time"]
            )
            plot += scale_y_sqrt(limits=(0, max_height), 
                                 breaks=[0, 10, 40, 80, 120, 160, 200],
                                 labels=lambda l: [f"{x:g}" for x in l]) 
        
        return plot
    
    #density_plot("entry_times", events).show()
    migration_time_plot = density_plot("migration_times", events)
    migration_time_plot.show()
    #quantity_over_time_plot("scaled_volume", events)
    #quantity_over_time_plot("volume", events)
    #quantity_over_time_plot("scaled_height", events)
    quantity_over_time_plots = quantity_over_time_plot("height", events)
    #[plot.show() for plot in quantity_over_time_plots]
    quantity_over_time_plots[0].show()
    
    #quantity_at_failure_time_plot("scaled_volume", events).show()
    #quantity_at_failure_time_plot("volume", events).show()
    #quantity_at_failure_time_plot("scaled_height", events).show()
    failure_time_plot = quantity_at_failure_time_plot("height", events)
    failure_time_plot.show()
    
    if folder is not None:
        migration_time_plot.save(path=folder)
        quantity_over_time_plots[0].save(path=folder)
        failure_time_plot.save(path=folder)
    
    
    
    
    
def plot_logistic_curves(
    smoothness_params: list[float],
    num_evals: int,
    x_min: float,
    x_max: float
) -> None:
    
    smoothness_params = tf.convert_to_tensor(
        smoothness_params, 
        dtype=tf.float32
    )

    logistic_curve_values = tf.math.sigmoid(tf.linspace(
        start=x_min / smoothness_params, 
        stop=x_max / smoothness_params, 
        num=num_evals
    ))
    
    idx = pd.MultiIndex.from_product(
        iterables=[
            tf.linspace(x_min, x_max, num_evals).numpy(),
            tf.range(tf.shape(smoothness_params)).numpy()
            
        ],
        names=["x_vals", "curve"]
    )

    df = pd.DataFrame({
        "y_vals": logistic_curve_values.numpy().ravel()
    }, index=idx).reset_index()

    plot = (
        ggplot(df, aes(x="x_vals", y="y_vals", colour="factor(curve)")) +
        geom_line(size=1) +
        annotate("segment", x=x_min, xend=0, y=0, yend=0, size=1) +
        annotate("segment", x=0, xend=x_max, y=1, yend=1, size=1) +
        annotate("point", x=0, y=0.5, size=3) +
        annotate("point", x=0, y=0, size=3, shape="o", fill="None") +
        annotate("point", x=0, y=1, size=3, shape="o", fill="None") +
        theme_minimal(base_size=18) +
        theme(figure_size=(12, 6)) +
        labs(
            x=r"$P_{i}(t) - e_{i, j} C_i$",
            y=r"$s_{i, j}^\mathrm{act}$",
            colour=""
        ) +
        scale_colour_hue(labels={
            i: r"$\tau^\mathrm{act} = $" + f"{smoothness_params[i]:.3}" 
            for i in tf.range(tf.shape(smoothness_params)).numpy()
        })
    )
    plot.show()





def sample_conditional_posteriors(
    config: dict[str, Any],
    ConditionalPosterior: tf.Module,
    LatentStateKernel: tf.Module,
    num_results: Optional[tf.Tensor] = None,
    num_burnin_steps: Optional[tf.Tensor] = None,
    allow_warm_start: bool = True
) -> dict[str, tf.Tensor]:
    
    if num_results is None:
        num_results = tf.cast(
            tf.constant(
                config["latent_state_kernel"]["number_of_resulting_samples"],
                dtype=tf.float32
            ),
            dtype=tf.int32
        )

    if num_burnin_steps is None:
        num_burnin_steps = tf.cast(
            tf.constant(
                config["latent_state_kernel"]["number_of_burnin_steps"],
                dtype=tf.float32
            ),
            dtype=tf.int32
        ) 
                
    graph_encodings = tf.cast(  # shape: [B, R, R]
        tf.constant(  
            config["posterior"]["graphs"],
            dtype=tf.int32
        ),
        dtype=tf.bool
    ) 
    
    indices = tf.constant(  # shape: [B, T]
        config["posterior"]["measurement_indices"],
        dtype=tf.int32
    )
    times = tf.gather(  # shape: [B, T]
        tf.constant(
            config["posterior"]["times"],
            dtype=tf.float32
        ),
        indices=indices    
    ) 
    measurements = tf.gather(  # shape: [B, T, R]
        tf.constant(
            config["posterior"]["measurements"],
            dtype=tf.float32
        ),
        indices = indices    
    ) 
    
    num_measurements = tf.shape(measurements)[1] 
    
    seeds = tfp.random.split_seed(
        tf.constant(config["general"]["seed"], dtype=tf.int32),
        n=num_measurements
    )
    
    results = {}
    initial_states = None
    for i in tf.range(num_measurements):
        
        accumulated_measurements = measurements[:, :(i+1)]
        accumulated_times = times[:, :(i+1)]
        
        conditional_posterior = ConditionalPosterior(
            config=config,
            measurements=accumulated_measurements,
            times=accumulated_times,
            graph_encoding=graph_encodings
        )
        kernel = LatentStateKernel(
            config=config,
            conditional_posterior=conditional_posterior
        )

        tf.print(
            "Currently sampling conditional prior(s) for graph(s):\n\n",
            graph_encodings,
            "\n\nwith measured height(s):\n\n",
            accumulated_measurements,
            "\n\nat time(s):\n\n",
            accumulated_times,
            "\n"
        )
        samples, trace, final_kernel_results = kernel.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            seed=seeds[i],
            initial_states=initial_states
        )
        tf.print("\n\nFinished sampling posterior states.\n\n")

        results[f"observation_{i+1}"] = {
            "samples": samples,
            "trace": trace,
            "final_kernel_results": final_kernel_results,
            "conditional_posterior": conditional_posterior,
            "measurements": accumulated_measurements,
            "times": accumulated_times,
            "measurement_indices": indices[:, :(i+1)]
        }
        
        if allow_warm_start:
            initial_states = {
                state: tf.stop_gradient(
                    samples[state][-1]
                )
                for state in LatentStateKernel.LATENT_STATES
            }

    return results





def plot_marginal_densities(
    parameter: np.ndarray,
    parameter_label: str,
    prior_densities: np.ndarray,
    density_labels: list[str],
) -> ggplot:
    
    df = pd.DataFrame({
        "parameter": np.tile(
            parameter,
            reps=prior_densities.shape[0]
        ),
        "prior_density": prior_densities.ravel(),
        "density_label": np.repeat(
            density_labels,
            repeats=parameter.size
        )
    })
    return (
        ggplot(df, aes(x="parameter", y="prior_density", 
                       colour="density_label")) +
        geom_line() +
        theme_minimal(base_size=12) +
        labs(x=parameter_label, y="Density", color="Hyperparameters")
    )
    




def plot_contours(
    prior_densities: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    highest_density_regions: np.ndarray,
    correlation: np.ndarray,
    parameter_labels: list[str]
) -> ggplot:
    
    sorted_densities = np.sort(
        prior_densities.reshape(correlation.size, x_axis.size*y_axis.size),
        axis=1
    )[:, ::-1]
    
    cumulative_probabilities = np.cumsum(
        sorted_densities,
        axis=1
    ) * (x_axis[1] - x_axis[0])  * (y_axis[1] - y_axis[0])

    idx = np.argmax(
        cumulative_probabilities[:, None, :] 
        >= highest_density_regions[None, :, None],
        axis=2
    )
    probability_thresholds = np.take_along_axis(
        sorted_densities,
        idx,
        axis=1
    )

    dfs = []
    for i, corr in enumerate(correlation):
        contours = (
            contour_generator(
                x=x_axis, 
                y=y_axis, 
                z=prior_densities[i],
                line_type=LineType.Separate
            )
            .multi_lines(probability_thresholds[i]) 
        )

        for j, contour in enumerate(contours):
            dfs.extend(
                pd.DataFrame({
                    "x": path[:, 0],
                    "y": path[:, 1],
                    "highest_density_regions": 
                        rf"{100*highest_density_regions[j]:g} %",
                    "path": f"{i}_{j}_{k}",
                    "correlation": rf"$\rho={corr:.1f}$"
                })
                for k, path in enumerate(contour)
            )
    df = pd.concat(dfs, ignore_index=True)

    return (
        ggplot(df, aes(x="x", y="y", colour="factor(highest_density_regions)", 
                       group="path")) +
        geom_path() +
        facet_wrap("~correlation") +
        coord_equal() +
        theme_bw(base_size=20) +
        theme(figure_size=(2.5*6.4, 4.8)) +
        labs(x=parameter_labels[0], y=parameter_labels[1], 
             colour="HDR")
    )





def get_covariance_matrix(
    standard_deviation: np.ndarray,
    correlation: np.ndarray
) -> np.ndarray:
    
    covariance_matrix = (
        correlation[:, None, None] 
        * np.outer(standard_deviation, standard_deviation)[None, :, :]
    )
    idx = np.arange(standard_deviation.size)
    covariance_matrix[:, idx, idx] = standard_deviation**2
    
    return covariance_matrix





def evaluate_density_CTP(
    unconstrained_mean: np.ndarray,
    unconstrained_std: np.ndarray,
    type: Literal["marginal", "joint"],
    correlation: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, Union[list[str], np.ndarray]]:

    match type:
        
        case "marginal":     
            upper_bound = max(
                lognorm.ppf(
                    0.99,
                    std,
                    scale=np.exp(mean)
                )
                for mean, std in product(unconstrained_mean, unconstrained_std)
            )

            CTP = np.linspace(1, upper_bound, 1000)
            
            prior_densities = np.array([
                lognorm.pdf(
                    CTP,
                    std,
                    scale=np.exp(mean)
                )
                for mean, std in product(unconstrained_mean, unconstrained_std)
            ])    
            
            density_labels = [
                rf"$\mu_i={mean}, \ \sigma_{{i}}={std}$"
                for mean, std in product(unconstrained_mean, unconstrained_std)
            ]

            return CTP, prior_densities, density_labels

        case "joint":
            lower_bound = lognorm.ppf(
                1e-4,
                unconstrained_std,
                scale=np.exp(unconstrained_mean)
            )
            upper_bound = lognorm.ppf(
                1-1e-4,
                unconstrained_std,
                scale=np.exp(unconstrained_mean)
            )

            x = np.linspace(lower_bound[0], upper_bound[0], 250)
            y = np.linspace(lower_bound[1], upper_bound[1], 250)
            xs, ys = np.meshgrid(x, y)
            log_points = np.dstack((np.log(xs), np.log(ys)))

            prior_densities = np.array([
                multivariate_normal.pdf(
                    x=log_points,
                    mean=unconstrained_mean,
                    cov=cov
                ) / (xs * ys)
                for cov in get_covariance_matrix(unconstrained_std, correlation)
            ])
            
            return x, y, prior_densities
        
        
        
        
        
def evaluate_density_relative_eCTP(
    unconstrained_mean: np.ndarray,
    unconstrained_std: np.ndarray,
    type: Literal["marginal", "joint"],
    correlation: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, Union[list[str], np.ndarray]]:

    match type:
            
        case "marginal":     
            relative_eCTP = np.linspace(1e-4, 1-1e-4, 1000)

            prior_densities = np.array([
                norm.pdf(
                    logit(relative_eCTP),
                    loc=mean,
                    scale=std
                ) / (relative_eCTP * (1 - relative_eCTP)) 
                for mean, std in product(unconstrained_mean, unconstrained_std)
            ])
            
            density_labels = [
                rf"$\mu_{{ij}}={mean}, \ \sigma_{{ij}}={std}$"
                for mean, std in product(unconstrained_mean, unconstrained_std)
            ]
            
            return relative_eCTP, prior_densities, density_labels

        case "joint":
            x = np.linspace(1e-4, 1-1e-4, 250)
            y = np.linspace(1e-4, 1-1e-4, 250)
            xs, ys = np.meshgrid(x, y)
            unconstrained_points = np.dstack((logit(xs), logit(ys)))
            
            prior_densities = np.array([
                multivariate_normal.pdf(
                    x=unconstrained_points,
                    mean=unconstrained_mean,
                    cov=cov
                ) / (xs * (1 - xs) * ys * (1 - ys))
                for cov in get_covariance_matrix(unconstrained_std, correlation)
            ])
            
            return x, y, prior_densities





def evaluate_density_relative_flow_rate(
    unconstrained_mean: np.ndarray,
    unconstrained_std: np.ndarray,
    type: Literal["marginal", "joint"],
    correlation: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, Union[list[str], np.ndarray]]:

    A = np.array([[1, 0, -1], [0, 1, -1]])
    
    x = np.linspace(1e-4, 1-1e-4, 250)
    y = np.linspace(1e-4, 1-1e-4, 250)
    xs, ys = np.meshgrid(x, y)

    reference_grid = 1 - xs - ys
    valid_idx = (reference_grid > 0)
    valid_log_reference_grid = np.log(reference_grid[valid_idx])

    valid_unconstrained_points = np.dstack((
        np.log(xs[valid_idx]) - valid_log_reference_grid, 
        np.log(ys[valid_idx]) - valid_log_reference_grid
    ))
    
    match type:
                
        case "marginal":     
            prior_densities = np.zeros((
                unconstrained_mean.size * unconstrained_std.size, 
                *xs.shape
            ))
            
            for i, (mean, std) in enumerate(
                product(unconstrained_mean, unconstrained_std)
            ):
                mean = np.array((mean, 0, 0))
                cov = std**2 * np.eye(3)
                
                prior_densities[i, valid_idx] = multivariate_normal.pdf(
                    x=valid_unconstrained_points,
                    mean=A @ mean,
                    cov=A @ cov @ A.T
                ) / (xs[valid_idx] * ys[valid_idx] * reference_grid[valid_idx])
                
            prior_densities = np.trapezoid(
                prior_densities,
                x=y,
                axis=1
            )
            
            density_labels = [
                rf"$\mu_{{ij}}={mean}, \ \sigma_{{ij}}={std}$"
                for mean, std in product(unconstrained_mean, unconstrained_std)
            ]
            
            return x, prior_densities, density_labels

        case "joint":
            prior_densities = np.zeros((correlation.size, *xs.shape))
            
            for i, cov in enumerate(
                get_covariance_matrix(unconstrained_std, correlation)
            ):
                prior_densities[i, valid_idx] = multivariate_normal.pdf(
                    x=valid_unconstrained_points,
                    mean=A @ unconstrained_mean,
                    cov=A @ cov @ A.T
                ) / (xs[valid_idx] * ys[valid_idx] * reference_grid[valid_idx])
            
            return x, y, prior_densities
        
        
        
        
        
def plot_likelihood(
    actual_height: np.ndarray,
    height_threshold: float,
    measurement_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    lower_bound = norm.ppf(
        1e-3,
        loc=actual_height[:, None],
        scale=measurement_std[None, :]
    ).min(axis=1)
    upper_bound = norm.ppf(
        1-1e-3,
        loc=actual_height[:, None],
        scale=measurement_std[None, :]
    ).max(axis=1)

    observed_height = np.linspace(lower_bound, upper_bound, 1000)  

    likelihood = norm.pdf(
        observed_height[:, :, None],
        loc=actual_height[None, :, None],
        scale=measurement_std[None, None, :]
    )
    
    df = (
        pd.DataFrame({
            "likelihood": likelihood.ravel(),
            "observed_height": np.broadcast_to(
                observed_height[:, :, None],
                shape=likelihood.shape        
            ).ravel(),
            "measurement_std": np.broadcast_to(
                measurement_std[None, None, :],
                shape=likelihood.shape
            ).ravel(),
            "actual_height": np.broadcast_to(
                actual_height[None, :, None],
                shape=likelihood.shape
            ).ravel()
        })
        .assign(
            censored = lambda x: x["observed_height"] < height_threshold,
            measurement_std_label = lambda x: x["measurement_std"].map(
                lambda std: rf"$\sigma_d = {std}$"
            )
        )
    )

    df_shaded_area = (
        df[df["censored"]]
        .groupby(
            ["actual_height", "observed_height"],
            as_index=False
        )["likelihood"]
        .max()
    )

    df_annotation_actual_height = pd.DataFrame({
        "actual_height": actual_height,
        "x_intercept": actual_height
    })
    
    df_annotation_height_threshold = (
        df
        .groupby("actual_height")["observed_height"]
        .agg(["min", "max"])
        .reset_index()
        .query("min <= @height_threshold <= max")
        .assign(x=height_threshold)
    )
    
    return (
        ggplot(
            df, 
            aes(x="observed_height", y="likelihood")
        ) +
        facet_wrap(
            "~actual_height", 
            scales="free_x", 
            labeller=labeller(
                actual_height = lambda height: 
                    rf"$h_i(t) = {height} \ \text{{m}}$"
            )
        ) +
        geom_line(
            aes(colour="factor(measurement_std_label)"),
            data=df[~df["censored"]]
        ) +
        geom_area(
            data=df_shaded_area, 
            fill="grey", 
            alpha=0.25
        ) +
        geom_line(
            aes(colour="factor(measurement_std_label)"),
            data=df[df["censored"]], 
            linetype="dashed"
        ) +
        geom_vline(
            aes(xintercept="x_intercept"), 
            data=df_annotation_actual_height, 
            linetype="dotted",
            inherit_aes=False
        ) +
        geom_vline(
            aes(xintercept="x"), 
            data=df_annotation_height_threshold, 
            linetype="solid",
            inherit_aes=False
        ) +
        theme_bw(base_size=20) +
        labs(
            x=r"$d_i(t) \ [m]$", 
            y="Density", 
            colour="Measurement\nNoise"
        ) +
        theme(figure_size=(2.5*6.4, 4.8))
    )
    
    
    
    
    
def plot_prior_posterior_parameter_density(
    latent_prior_samples: dict[str, Any],
    hmc_results: dict[str, Any]
) -> list[ggplot]:
     
    parameters = (
        "log_lateral_parameter", 
        "log_vertical_parameter", 
        "log_threshold_pressure"
    )
    parameter_labels = {
        "log_lateral_parameter": r"$\alpha_i \ \text{[m]}$", 
        "log_vertical_parameter": r"$\beta_i \ \text{[m]}$", 
        "log_threshold_pressure": r"$C_i \ \text{[kPa]}$"
    }

    num_samples, num_graphs, num_reservoirs = tf.shape(
        hmc_results["observation_1"]["samples"]["log_lateral_parameter"]
    ) 
    
    # dataframe with posterior samples 
    dfs = []
    for observation, result in hmc_results.items():    
        dfs.extend(
            pd.DataFrame({
                "latent_samples": 
                    result["samples"][parameter].numpy().ravel(),
                "reservoir":
                    np.tile(
                        np.arange(1, num_reservoirs+1),
                        reps=num_samples*num_graphs
                    ),
                "parameter_label":
                    parameter_labels[parameter],
                "observation": 
                    observation,
                "observation_label": 
                    rf"$D = {tf.shape(result["times"])[1]}$",
                "graph":
                    np.tile(
                        np.repeat(
                            np.arange(num_graphs),
                            repeats=num_reservoirs
                        ),
                        reps=num_samples
                    ),
                "distribution": "posterior"
            })
            for parameter in parameters
        )
    posterior_df = pd.concat(dfs, ignore_index=True)    
    
    # dataframe with prior samples 
    dfs = []
    for parameter in parameters:    
        dfs.append(
            pd.DataFrame({
                "latent_samples": 
                    latent_prior_samples[parameter].numpy().ravel(),
                "reservoir":
                    np.tile(
                        np.arange(1, num_reservoirs+1),
                        reps=num_samples
                    ),
                "parameter_label":
                    parameter_labels[parameter],
                "distribution": "prior",
                "observation_label":
                    r"$D = 0$"
            })
        )
    # independent of observations and graphs
    prior_df = (
        pd.concat(dfs, ignore_index=True)
        .merge(
            (
                posterior_df[["observation", "graph"]]
                .drop_duplicates()
                .reset_index(drop=True)
            ),
            how="cross"
        )
    ) 
    
    df = (
        pd.concat((posterior_df, prior_df), ignore_index=True)
        .assign(samples=lambda x: np.exp(x.latent_samples))
    )
    
    plots = []
    for i in range(num_graphs):
        plots.append(
            ggplot(df[df["graph"]==i], 
                   aes(x="samples", colour="factor(reservoir)")) +
            geom_density() +
            facet_grid("observation_label ~ parameter_label", scales="free") +
            labs(x="Parameter Value", y="Density", colour="Reservoir") +
            theme_bw(base_size=30) +
            theme(figure_size=(3*6.4, 2*4.8)) +
            scale_x_log10() +
            scale_y_sqrt()
        )
    
    return plots 
        
        
        
        
        
def plot_posterior_height_over_time(
    config: dict[str, Any],
    posterior_results: dict[str, tf.Tensor],
    Simulator: tf.Module,
) -> None:
    
    time_grid = tf.linspace(
        0.0, 
        config["utils"]["plot"]["max_time_simulation"], 
        int(config["utils"]["plot"]["grid_resolution"])
    )

    height_threshold = tf.constant(
        config["acquisition"]["height_threshold"],
        dtype=tf.float32
    )
    
    graph_encodings = tf.cast(  # shape: [B, R, R]
        tf.constant(  
            config["posterior"]["graphs"],
            dtype=tf.int32
        ),
        dtype=tf.bool
    )
  
    num_samples, num_graphs, num_reservoirs = tf.shape(
        posterior_results["observation_1"]["samples"]["log_lateral_parameter"]
    ) 
    
    simulator = Simulator(config, num_aggregate_samples=num_samples)
    
    plots = []
    for i in range(num_graphs):
        
        quantile_dfs, measurement_dfs = [], []
        for observation, result in posterior_results.items():
            
            measurements = result["measurements"][i].numpy()
            measurement_times = result["times"][i].numpy()
            num_times = len(measurement_times)
            
            measurement_dfs.append(
                pd.DataFrame({
                    "measurement": 
                        measurements.ravel(),
                    "height_threshold": 
                        np.tile(
                            height_threshold.numpy().ravel(),
                            reps=num_times
                    ),
                    "year": np.repeat(
                        measurement_times.ravel(),
                        repeats=num_reservoirs
                    ),
                    "reservoir": np.tile(
                        np.arange(1, num_reservoirs+1),
                        reps=num_times
                    ),
                    "observation": observation,
                    "observation_label": 
                        rf"$D = {measurement_times.size}$" 
                })
                .assign(
                    censored=lambda x: x["measurement"] == 0.0
                )
            )
            
            latent_states = {
                state: samples[:, i, :]
                for state, samples in result["samples"].items()
            }
            graph_encoding = tf.broadcast_to(
                graph_encodings[i][None, :, :], 
                shape=(num_samples, num_reservoirs, num_reservoirs)
            )
            
            heights = simulator.simulate(
                quantity="height",
                latent_states=latent_states,
                graph_encoding=graph_encoding,
                times=time_grid
            )["simulated_values"]
            quantiles = tfp.stats.percentile(
                heights,
                q=[5.0, 50.0, 95.0],
                axis=0,
                interpolation="linear"
            )
            lower, median, upper = tf.unstack(quantiles)
                
            idx = pd.MultiIndex.from_product(
                iterables=[
                    time_grid.numpy(), 
                    tf.range(1, num_reservoirs+1).numpy()
                ],
                names=["year", "reservoir"]
            )
            quantile_dfs.append(
                pd.DataFrame({
                    "median": median.numpy().ravel(),
                    "lower": lower.numpy().ravel(),
                    "upper": upper.numpy().ravel(),
                    "observation": observation,
                    "observation_label": 
                        rf"$D = {measurement_times.size}$"  
                }, index=idx).reset_index()
            )
            
        quantile_df = pd.concat(quantile_dfs, axis=0)
        measurement_df = pd.concat(measurement_dfs, axis=0)    
        for df in (quantile_df, measurement_df):
            # assign categories to the reservoirs manually, as quantile_df
            # might not contain all categories once filtered for censored and 
            # uncensored observations. 
            # ' colour="factor(reservoir)" ' is therefore problematic
            df["reservoir"] = pd.Categorical(
                df["reservoir"],
                categories=np.arange(1, num_reservoirs+1)
            )
        
        max_plot_time = (
            config["utils"]["plot"]["max_time_height_over_time_plot"]
        )
        plots.append(
            ggplot(quantile_df, aes(x="year", colour="reservoir")) +
            facet_grid("reservoir ~ observation_label", scales="fixed") +
            #geom_vline(aes(xintercept="year"), data=measurement_df, 
            #           inherit_aes=False, linetype="dashed") +
            geom_line(aes(y="median"), size=1) +
            geom_ribbon(aes(ymin="lower", ymax="upper", 
                            fill="reservoir"), 
                        alpha=0.2, linetype="None", show_legend=False) +
            geom_line(aes(y="lower"), linetype="dashed", size=1) +
            geom_line(aes(y="upper"), linetype="dashed", size=1) +
            geom_point(aes(y="height_threshold"), 
                       data=measurement_df[measurement_df["censored"]],
                       show_legend=False, shape="v", size=10) +
            geom_point(aes(y="measurement"), 
                       data=measurement_df[~measurement_df["censored"]],
                       show_legend=False, size=10) +
            theme_bw(base_size=30) +
            scale_x_sqrt(breaks=[0, 2.5, 10, 25, 45, 70, 100],
                         labels=lambda l: [f"{x:g}" for x in l]) +
            coord_cartesian(xlim=(0, max_plot_time)) +
            labs(x=r"$t \ [year]$", y=r"$h_i(t) \ \text{[m]}$", 
                 colour="Reservoir") +
            theme(strip_text_y=element_blank(), figure_size=(2*6.4, 5*4.8))
        )
    
    return plots