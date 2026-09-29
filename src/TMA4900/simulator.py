import numpy as np
from typing import Optional, Any, Literal
import pandas as pd
import tensorflow as tf
from plotnine import (ggplot, aes, geom_line, theme_minimal, facet_wrap)

class Simulator(tf.Module):
    
    def __init__(
        self, 
        config: dict[str, Any],
        num_aggregate_samples: Optional[int] = None,
        name: str = None
    ) -> None:
        super().__init__(name)
        
        # number of batches and reservoirs
        if num_aggregate_samples is not None:
            B = num_aggregate_samples
        else:
            B = config["general"]["number_of_batches"]        
        self.B = tf.constant(B, dtype=tf.int32)
        self.R = tf.constant(
            config["general"]["number_of_reservoirs"],
            dtype=tf.int32
        )
        
        # small constant used for numerical stability
        self.epsilon = tf.constant(
            config["simulator"]["other"]["epsilon"],
            dtype=tf.float32
        )
        
        # injection rate
        mass_injection_rate = tf.constant(
            config["simulator"]["other"]["mass_injection_rate"],
            dtype=tf.float32
        )[:self.R]
        self.mass_injection_rate = tf.broadcast_to(
            mass_injection_rate[None, :],
            shape=[self.B, self.R]
        )
        
        # physical quantities
        self.physical = {
            "co2_in_situ_density": tf.constant(
                config["physical"]["co2_in_situ_density"],
                dtype=tf.float32
            ),
            "co2_brine_density_difference": tf.constant(
                config["physical"]["co2_brine_density_difference"],
                dtype=tf.float32
            ),
            "standard_gravity": tf.constant(
                config["physical"]["standard_gravity"],
                dtype=tf.float32
            )
        }
        
        # smoothing parameters
        self.smoothing = {
            "activation": tf.constant(
                config["simulator"]["smoothing"]["activation"],
                dtype=tf.float32
            ),
            "softmin": tf.constant(
                config["simulator"]["smoothing"]["softmin"],
                dtype=tf.float32
            ),
            "softclip": tf.constant(
                config["simulator"]["smoothing"]["softclip"],
                dtype=tf.float32
            )
        }
              
    # function to compute softmin along an axis of a tensor
    @tf.function
    def reduce_softmin(
        self,
        tensor: tf.Tensor,
        axis: tf.Tensor,
        smoothing_parameter: tf.Tensor
    ) -> tf.Tensor:
        
        return (
            - smoothing_parameter 
            * tf.math.reduce_logsumexp(
                - tensor / smoothing_parameter, 
                axis=axis
            )
        )
    
    # function to transform latent states (tilde theta) to states (theta)
    @tf.function
    def _get_states(
        self,
        latent_states: dict[str, tf.Tensor],
        graph_encoding: tf.Tensor 
    ) -> dict[str, tf.Tensor]:

        ones = tf.ones((self.R, self.R), dtype=tf.float32)
        strictly_upper_triangular = (
            tf.linalg.band_part(ones, num_lower=0, num_upper=-1) 
            - tf.linalg.band_part(ones, num_lower=0, num_upper=0)
        )
        saturated_graph_encoding = tf.broadcast_to(
            strictly_upper_triangular[None, :, :],
            shape=[self.B, self.R, self.R]
        )
        
        # tf.function is not allowed to modify inputs directly, 
        # so create a shallow copy of latent_states
        states = latent_states.copy()  
        
        flattened_unconstrained_flow = tf.reshape(
            states.pop("unconstrained_flow"), 
            shape=[-1]
        ) 
        exponentiated_flow = tf.scatter_nd(
            indices=tf.where(saturated_graph_encoding),
            updates=tf.math.exp(flattened_unconstrained_flow),
            shape=saturated_graph_encoding.get_shape() 
        ) 
        exponentiated_flow = (
            exponentiated_flow * tf.cast(graph_encoding, dtype=tf.float32)
        )
        exponentiated_flow_per_reservoir = tf.reduce_sum(
            exponentiated_flow,
            axis=2
        )
        relative_flow_rate = tf.math.divide_no_nan(
            exponentiated_flow, 
            exponentiated_flow_per_reservoir[:, :, None] 
        ) 
        
        flattened_unconstrained_activation = tf.reshape(
            states.pop("unconstrained_activation"),
            shape=[-1]
        ) 
        relative_activation = tf.scatter_nd(
            indices=tf.where(saturated_graph_encoding),
            updates=tf.math.sigmoid(flattened_unconstrained_activation),
            shape=saturated_graph_encoding.get_shape() 
        ) 
        relative_activation = (
            relative_activation * tf.cast(graph_encoding, dtype=tf.float32)
        )
                
        states.update({
            "relative_flow_rate": relative_flow_rate,
            "relative_activation": relative_activation
        })
        return states
    
    # function to obtain scaled volume from scaled height
    @tf.function
    def _scaled_height_to_scaled_volume(
        self,
        scaled_height: tf.Tensor
    ) -> tf.Tensor:
        return 0.5 * scaled_height**2 * (3 - scaled_height)
    
    # function to obtain scaled height from scaled volume
    @tf.function
    def _scaled_volume_to_scaled_height(
        self,
        scaled_volume: tf.Tensor,
        tol: tf.Tensor=tf.constant(1e-5, dtype=tf.float32), 
        max_iter: tf.Tensor=tf.constant(20, dtype=tf.int32)
    ) -> tf.Tensor:
        
        def newton_raphson(scaled_volume):
            def cond(current_iter, change, scaled_height):
                return tf.logical_and(current_iter < max_iter, change > tol)
            
            def body(current_iter, change, scaled_height):
                func = (
                    self._scaled_height_to_scaled_volume(scaled_height) 
                    - scaled_volume
                ) 
                func_prime = 1.5 * scaled_height * (2.0 - scaled_height)
                
                # Newton-Raphson step, epsilon included for numerical stability 
                step = - func / tf.maximum(func_prime, self.epsilon)  
                
                return (
                    current_iter + 1,
                    tf.reduce_max(tf.abs(step)),
                    tf.clip_by_value(scaled_height + step, 0.0, 1.0)
                )
                
            initial_states = (
                tf.constant(0, dtype=tf.int32),
                tf.convert_to_tensor(np.inf, dtype=tf.float32),
                tf.clip_by_value(
                    scaled_volume**(1/3),  # initial scaled height estimate 
                    0.0, 
                    1.0
                )
            )

            return tf.while_loop(cond, body, initial_states)[2]
        
        # provide exact gradient for automatic differentiation so that 
        # TensorFlow does not have to propagate through Newton-Raphson
        @tf.custom_gradient 
        def compute_scaled_height(scaled_volume):
            
            scaled_height = tf.stop_gradient(newton_raphson(scaled_volume))    
                
            def grad(upstream):
                local_gradient = tf.math.divide_no_nan(
                    1.0,
                    1.5 * scaled_height * (2.0 - scaled_height) 
                )
                return upstream * local_gradient
            
            return scaled_height, grad

        return compute_scaled_height(scaled_volume)
    
    # function to determine whether migration can take place
    @tf.function
    def _is_migration_condition_met(
        self,
        scaled_volume: tf.Tensor,
        scaled_critical_volume: tf.Tensor
    ) -> tf.Tensor:
        return scaled_volume >= scaled_critical_volume - self.epsilon
    
    # function to obtain volume from scaled volume
    @tf.function
    def _scaled_volume_to_volume(
        self,
        scaled_volume: tf.Tensor,
        log_lateral_parameter: tf.Tensor, 
        log_vertical_parameter: tf.Tensor,
    ) -> tf.Tensor:
        
        max_volume = tf.math.exp(
            tf.math.log(tf.constant(2.0, dtype=tf.float32))
            + tf.math.log(tf.constant(np.pi, dtype=tf.float32))
            + 2.0 * log_lateral_parameter
            + log_vertical_parameter
            - tf.math.log(tf.constant(3.0, dtype=tf.float32))
        )
        
        return max_volume[:, None, :] * scaled_volume

    # function to obtain height from scaled height
    @tf.function
    def _scaled_height_to_height(
        self,
        scaled_height: tf.Tensor,
        log_vertical_parameter: tf.Tensor,
    ) -> tf.Tensor:
        
        max_height = tf.math.exp(log_vertical_parameter)
             
        return max_height[:, None, :] * scaled_height
        
    # function to compute scaled critical volume (y_ij^crit) and critical volume
    # of topmost reservoir  
    @tf.function
    def _get_scaled_critical_volume(
        self,
        log_vertical_parameter: tf.Tensor, 
        log_threshold_pressure: tf.Tensor,
        relative_activation: tf.Tensor,
        graph_encoding: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        
        log_scaled_height = (
            log_threshold_pressure 
            + 3 * tf.math.log(tf.constant(10, tf.float32))  # kPa -> Pa
            - tf.math.log(self.physical["co2_brine_density_difference"]) 
            - tf.math.log(self.physical["standard_gravity"]) 
            - log_vertical_parameter
        )
        
        scaled_critical_height_path = tf.clip_by_value(
            relative_activation
             * tf.cast(graph_encoding, dtype=tf.float32)
             * tf.math.exp(log_scaled_height)[:, :, None],
            0.0,
            1.0
        )
        
        scaled_critical_height_top_reservoir = tf.clip_by_value(
            tf.math.exp(log_scaled_height)[:, -1],
            0.0,
            1.0
        )
        
        return (
            self._scaled_height_to_scaled_volume(scaled_critical_height_path),
            self._scaled_height_to_scaled_volume(scaled_critical_height_top_reservoir)
        )
    
    # function to compute scaled net injection rate (z(t))
    @tf.function
    def _get_scaled_rate(
        self,
        scaled_volume: tf.Tensor,
        scaled_critical_volume_path: tf.Tensor,
        log_lateral_parameter: tf.Tensor, 
        log_vertical_parameter: tf.Tensor,
        relative_flow_rate: tf.Tensor,
        graph_encoding: tf.Tensor
    ) -> tf.Tensor:
        
        activation = tf.math.sigmoid(
            (scaled_volume[:, :, None] - scaled_critical_volume_path) /
            self.smoothing["activation"]
        ) 
        flow_tensor = (
            activation
            * relative_flow_rate
            * tf.cast(graph_encoding, dtype=tf.float32)
        )    
        
        throughput = tf.linalg.triangular_solve(
            matrix=(
                tf.eye(num_rows=self.R, batch_shape=[self.B]) 
                - tf.transpose(flow_tensor, perm=[0, 2, 1])
            ),
            rhs=self.mass_injection_rate[:, :, None],
            lower=True
        )
        throughput = tf.squeeze(throughput, axis=-1)
        
        ones = tf.ones((self.B, self.R), dtype=tf.float32)
        retention_factor = ones - tf.reduce_sum(flow_tensor, axis=2)
        mass_rate = retention_factor * throughput
        
        log_max_volume = (
            tf.math.log(tf.constant(2.0, dtype=tf.float32))
            + tf.math.log(tf.constant(np.pi, dtype=tf.float32))
            + 2.0 * log_lateral_parameter
            + log_vertical_parameter
            - tf.math.log(tf.constant(3.0, dtype=tf.float32))
        )

        # converting between Mtpa to 1/year
        scaling_factor = tf.math.exp(
            9 * tf.math.log(tf.constant(10.0, dtype=tf.float32))
            - tf.math.log(self.physical["co2_in_situ_density"])
            - log_max_volume
        )
        
        return scaling_factor * mass_rate
    
    # function to compute time to next event
    @tf.function
    def _get_event_time(
        self,
        scaled_volume: tf.Tensor,
        scaled_critical_volume_path: tf.Tensor,
        scaled_critical_volume_top_reservoir: tf.Tensor,
        scaled_rate: tf.Tensor,
        max_time_interval: Optional[tf.Tensor]=None
    ) -> tf.Tensor:
                
        breach_time = tf.math.divide_no_nan(
            scaled_critical_volume_path - scaled_volume[:, :, None],
            scaled_rate[:, :, None]
        )
        breach_time = tf.where(breach_time > 0, breach_time, np.inf)
        breach_time = self.reduce_softmin(
            breach_time, 
            axis=(2, 1), 
            smoothing_parameter=self.smoothing["softmin"]
        )
        correction = (
            self.smoothing["softmin"]
            * tf.math.log(tf.cast(self.R**2, dtype=tf.float32))
        )
        breach_time = breach_time + correction
        
        failure_time = tf.math.divide_no_nan(
            scaled_critical_volume_top_reservoir - scaled_volume[:, -1],
            scaled_rate[:, -1]
        )
        failure_time = tf.where(failure_time > 0, failure_time, np.inf)
        
        event_time = tf.stack((breach_time, failure_time), axis=1)
        event_time = self.reduce_softmin(
            event_time, 
            axis=1, 
            smoothing_parameter=self.smoothing["softmin"]
        ) 
        correction = (
            self.smoothing["softmin"] 
            * tf.math.log(tf.constant(2.0, dtype=tf.float32))
        )
        event_time = event_time + correction
        
        return event_time if max_time_interval is None else tf.where(
            event_time < max_time_interval, 
            event_time, 
            max_time_interval
        ) 
    
    # function to compute entry- and migration times as the simulator runs
    @tf.function
    def _update_entry_and_migration_time(
        self,
        entry_time: tf.Tensor,
        migration_time: tf.Tensor,
        current_time: tf.Tensor,
        current_scaled_volume: tf.Tensor,
        current_scaled_rate: tf.Tensor,
        next_time: tf.Tensor,
        next_scaled_volume: tf.Tensor,
        scaled_critical_volume_path: tf.Tensor,
        scaled_critical_volume_top_reservoir: tf.Tensor,
        batch_is_ongoing: tf.Tensor,
        graph_encoding: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
                
        entry_time_mask = (
            batch_is_ongoing[:, None] &
            tf.math.is_inf(entry_time) &
            (current_scaled_volume == 0.0) &
            (current_scaled_rate > 0.0)
        )
        updated_entry_time = tf.where(
            entry_time_mask,
            current_time[:, None],
            entry_time
        )
        
        migration_condition_lower_reservoirs = tf.reduce_any((
            graph_encoding &
            self._is_migration_condition_met(
                scaled_volume=next_scaled_volume[:, :, None],
                scaled_critical_volume=scaled_critical_volume_path
            )
        ), axis=2)[:, :-1]
        migration_condition_top_reservoir = self._is_migration_condition_met(
            scaled_volume=next_scaled_volume[:, -1],
            scaled_critical_volume=scaled_critical_volume_top_reservoir
        )
        migration_condition = tf.concat((
            migration_condition_lower_reservoirs,
            migration_condition_top_reservoir[:, None]
        ), axis=1)
        migration_time_mask = (
            batch_is_ongoing[:, None] &
            tf.math.is_inf(migration_time) &
            migration_condition 
        )
        updated_migration_time = tf.where(
            migration_time_mask,
            next_time[:, None],
            migration_time
        )
             
        return updated_entry_time, updated_migration_time
    
    # function to simulate all event times
    @tf.function
    def _simulate_events(
        self,
        states: dict[tf.Tensor],
        graph_encoding: tf.Tensor,
        max_iter: tf.Tensor=tf.constant(50, dtype=tf.int32),
        max_time_interval: Optional[tf.Tensor]=None,
    ) -> dict[str, tf.Tensor]:
         
        (
            scaled_critical_volume_path, 
            scaled_critical_volume_top_reservoir
        ) = self._get_scaled_critical_volume(
            log_vertical_parameter=states["log_vertical_parameter"],
            log_threshold_pressure=states["log_threshold_pressure"],
            relative_activation=states["relative_activation"],
            graph_encoding=graph_encoding
        )
        
        time_array = tf.TensorArray(
            dtype=tf.float32, 
            size=0, 
            dynamic_size=True, 
            clear_after_read=False
        )
        scaled_volume_array = tf.TensorArray(
            dtype=tf.float32, 
            size=0, 
            dynamic_size=True, 
            clear_after_read=False
        )
        
        time_array = time_array.write(
            index=0,
            value=tf.zeros(shape=self.B, dtype=tf.float32)
        )
        scaled_volume_array = scaled_volume_array.write(
            index=0,
            value=tf.zeros(shape=(self.B, self.R), dtype=tf.float32)
        )
    
        def cond(current_iter, time_array, scaled_volume_array, batch_ongoing,
                 entry_time, migration_time):
            return tf.logical_and(
                current_iter < max_iter,
                tf.reduce_any(batch_ongoing)
            )
        
        def body(current_iter, time_array, scaled_volume_array, batch_ongoing,
                 entry_time, migration_time):
            current_time = time_array.read(current_iter)
            current_scaled_volume = scaled_volume_array.read(current_iter)
            current_scaled_rate = self._get_scaled_rate(
                scaled_volume=current_scaled_volume,
                scaled_critical_volume_path=scaled_critical_volume_path,
                log_lateral_parameter=states["log_lateral_parameter"],
                log_vertical_parameter=states["log_vertical_parameter"],
                relative_flow_rate=states["relative_flow_rate"],
                graph_encoding=graph_encoding
            )
            
            event_time = self._get_event_time(
                scaled_volume=current_scaled_volume,
                scaled_critical_volume_path=scaled_critical_volume_path,
                scaled_critical_volume_top_reservoir=scaled_critical_volume_top_reservoir,
                scaled_rate=current_scaled_rate,
                max_time_interval=max_time_interval
            )
            # only advance ongoing batches that have not yet finished
            event_time = tf.where(
                batch_ongoing,
                event_time,
                0.0
            )
            
            next_scaled_volume = tf.clip_by_value( 
                current_scaled_volume 
                + current_scaled_rate * event_time[:, None],
                0.0,
                1.0
            )
            
            next_time = current_time + event_time
            next_iter = current_iter + 1
            
            time_array = time_array.write(
                next_iter, 
                next_time
            )
            scaled_volume_array = scaled_volume_array.write(
                next_iter, 
                next_scaled_volume
            )
        
            entry_time, migration_time = self._update_entry_and_migration_time(
                entry_time=entry_time,
                migration_time=migration_time,
                current_time=current_time,
                current_scaled_volume=current_scaled_volume,
                current_scaled_rate=current_scaled_rate,
                next_time=next_time,
                next_scaled_volume=next_scaled_volume,
                scaled_critical_volume_path=scaled_critical_volume_path,
                scaled_critical_volume_top_reservoir=scaled_critical_volume_top_reservoir,
                batch_is_ongoing=batch_ongoing,
                graph_encoding=graph_encoding
            )
            
            batch_ongoing = tf.logical_not(self._is_migration_condition_met(
                scaled_volume=next_scaled_volume[:, -1],
                scaled_critical_volume=scaled_critical_volume_top_reservoir
            ))
            
            return (next_iter, time_array, scaled_volume_array, batch_ongoing,
                    entry_time, migration_time)

        initial_states = (
            tf.constant(0, dtype=tf.int32),
            time_array,
            scaled_volume_array,
            tf.fill((self.B, ), value=True),
            tf.fill((self.B, self.R), value=np.inf),
            tf.fill((self.B, self.R), value=np.inf)
        )
        
        results = tf.while_loop(cond, body, initial_states)
        
        return {
            "number_of_events": results[0] + 1,
            "time_array": tf.transpose(
                results[1].stack()
            ),
            "scaled_volume_array": tf.transpose(
                tf.clip_by_value(
                    results[2].stack(),
                    0.0, 
                    1.0
                ),
                perm=[1, 0, 2]
            ),
            "batch_finished": tf.logical_not(results[3]),
            "entry_times": results[4],
            "migration_times": results[5]
        }
    
    # function to interpolate the event times
    @tf.function
    def _interpolate(
        self,
        events: dict[str, tf.Tensor],
        times: tf.Tensor
    ) -> tf.Tensor:
        
        if times.shape.rank == 1:
            times = tf.broadcast_to(
                times[None, :], 
                shape=[self.B, times.get_shape()[-1]]
            )
        
        right_idx = tf.searchsorted(
            events["time_array"],
            values=times,
            side="right"
        )
        left_idx = tf.maximum(right_idx - 1, 0)
        right_idx = tf.minimum(right_idx, events["number_of_events"] - 1)
        
        def gather(array, idx):
            return tf.gather(events[array], idx, batch_dims=1)
        
        left_times    = gather("time_array", left_idx)
        right_times   = gather("time_array", right_idx)
        left_volumes  = gather("scaled_volume_array", left_idx)
        right_volumes = gather("scaled_volume_array", right_idx)
        
        weights = tf.math.divide_no_nan(
            times - left_times,
            right_times - left_times
        )
        
        return (
            left_volumes + weights[:, :, None] * (right_volumes - left_volumes)
        )
        
    @tf.function
    def simulate(
        self,
        quantity: Literal["scaled_volume", "scaled_height", "volume", "height"],
        latent_states: tf.Tensor,
        graph_encoding: tf.Tensor,
        times: tf.Tensor,
        max_iter: tf.Tensor=tf.constant(50, dtype=tf.int32),
        max_time_interval: Optional[tf.Tensor]=None
    ) -> Optional[dict[str, tf.Tensor]]:
        """
        Run the simulation for a set of latent states (tilde theta) and graph encoding (G).
        The simulation returns the specified quantity at the specified times using linear interpolation.

        Args:
            quantity: Quantity to simulate.
            latent_states: Latent states (tilde theta).
            graph_encoding: Graph encoding (G).
            times: Times at which to compute the quantities.
            max_time_interval: Maximum time interval to consider when computing the event times. 
            Used as a failsafe against numerical errors.

        Returns:
            If valid quantity is specified, a dictionary containing the computed quantities at each time point,
            in addition to the entry- and migration times.
        """
        
        if quantity not in (
            "scaled_volume", "scaled_height", "volume", "height"
        ):
            tf.print(f'Quantity "{quantity}" is not supported!')
            return 
                
        states = self._get_states(
            latent_states=latent_states, 
            graph_encoding=graph_encoding
        )
        events = self._simulate_events(
            states=states,
            graph_encoding=graph_encoding,
            max_iter=max_iter, 
            max_time_interval=max_time_interval
        )
        
        interpolated_scaled_volume = self._interpolate(
            events=events,
            times=times
        )
        
        if quantity == "scaled_volume":
            simulated_values = interpolated_scaled_volume
        elif quantity == "volume":
            simulated_values = self._scaled_volume_to_volume(
                scaled_volume=interpolated_scaled_volume,
                log_lateral_parameter=states["log_lateral_parameter"],
                log_vertical_parameter=states["log_vertical_parameter"]
            )
        
        scaled_height = self._scaled_volume_to_scaled_height(
            interpolated_scaled_volume
        )
        
        if quantity == "scaled_height":
            simulated_values = scaled_height
        elif quantity == "height":
            simulated_values = self._scaled_height_to_height(
                scaled_height,
                log_vertical_parameter=states["log_vertical_parameter"]
            )
        
        return {
            "simulated_values": simulated_values,
            "entry_times": events["entry_times"],
            "migration_times": events["migration_times"] 
        }
    
    # LEGACY: function to plot a quantity e.g. height over time
    def plot(
        self,
        max_time: tf.Tensor,
        quantity: Literal["scaled_volume", "scaled_height", "volume", "height"],
        latent_states: tf.Tensor,
        graph_encoding: tf.Tensor,
        grid_resolution: tf.Tensor = tf.constant(250, dtype=tf.int32),
        max_iter: tf.Tensor = tf.constant(50, dtype=tf.int32),
        max_time_interval: Optional[tf.Tensor] = None
    ) -> None:
        
        time_grid = tf.linspace(
            0.0, 
            tf.cast(max_time, dtype=tf.float32), 
            tf.cast(grid_resolution, dtype=tf.int32)
        ) 
        
        result = self.simulate(
            quantity=quantity,
            latent_states=latent_states,
            graph_encoding=graph_encoding,
            times=time_grid, 
            max_iter=max_iter, 
            max_time_interval=max_time_interval
        )
        if result is None: 
            return  # "quantity" is unsupported
        else: 
            values = result["simulated_values"]
        
        idx = pd.MultiIndex.from_product(
            iterables=(
                np.arange(self.B),
                time_grid.numpy(),
                np.arange(1, self.R+1)
            ),
            names=(
                "batch",
                "year",
                "reservoir"
            )
        )
        df = pd.DataFrame(
            {quantity: values.numpy().ravel()}, 
            index=idx
        ).reset_index()
        
        plot = (
            ggplot(df, aes(x="year", y=quantity, colour="factor(reservoir)")) +
            geom_line() +
            facet_wrap("~batch", scales="free_y") +
            theme_minimal()
        )
        plot.show()