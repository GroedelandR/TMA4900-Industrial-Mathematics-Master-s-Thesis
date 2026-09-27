from typing import Optional, Any, Union
import tensorflow as tf
import tensorflow_probability as tfp

from TMA4900.simulator import Simulator
from TMA4900.priors import LatentStatePrior
from TMA4900.acquisition_model import AcquisitionModel

tfd = tfp.distributions

class ConditionalPosterior(tf.Module):
    
    def __init__(
        self, 
        config: dict[str, Any],
        measurements: tf.Tensor,  # shape: [B, T, R] 
        times: tf.Tensor,  # shape: [B, T]
        graph_encoding: tf.Tensor,  # shape: [B, R, R]
        Simulator: tf.Module = Simulator,
        LatentStatePrior: tf.Module = LatentStatePrior,
        AcquisitionModel: tf.Module = AcquisitionModel,
        name = None
    ) -> None:
        super().__init__(name)
    
        self.measurements = measurements 
        self.times = times
        self.graph_encoding = graph_encoding
        
        self.simulator = Simulator(
            config, 
            num_aggregate_samples=tf.shape(measurements)[0]
        )
        self.latent_state_prior = LatentStatePrior(config)
        self.acquisition_model = AcquisitionModel(config)
               
    @tf.function
    def log_prob(
        self,
        latent_states: dict[str, tf.Tensor],
    ) -> tf.Tensor:  # shape: [B]  

        results = self.simulator.simulate( 
            "height", 
            latent_states=latent_states,
            graph_encoding=self.graph_encoding,
            times=self.times
        )
        heights = results["simulated_values"] # shape: [B, T, R]
        
        log_likelihood = self.acquisition_model.log_likelihood(
            measurements=self.measurements,
            heights=heights
        )
        
        log_prior_prob = self.latent_state_prior.log_prob(
            latent_states=latent_states
        )
        
        return log_likelihood + log_prior_prob        
    
    
class LatentStateKernel(tf.Module):
    
    LATENT_STATES = (
        "log_lateral_parameter",
        "log_vertical_parameter",
        "log_threshold_pressure",
        "unconstrained_flow", 
        "unconstrained_activation",
    )
    
    def __init__(
        self, 
        config: dict[str, Any],
        conditional_posterior: ConditionalPosterior,
        name = None
    ) -> None:
        super().__init__(name)
        
        self.conditional_posterior = conditional_posterior
        
        self.seed = tf.constant(
            config["general"]["seed"],
            dtype=tf.int32
        )
        
        self.step_size = tf.constant(
            config["latent_state_kernel"]["step_size"],
            dtype=tf.float32
        )
        
        self.num_leapfrog_steps = tf.constant(
            config["latent_state_kernel"]["number_of_leapfrog_steps"],
            dtype=tf.int32
        ) 
    
        self.num_chains = tf.shape(self.conditional_posterior.measurements)[0]
    
    @staticmethod
    def _to_list(
        latent_states: dict[str, tf.Tensor]
    ) -> list[tf.Tensor]:
        return [
            latent_states[state] for state in LatentStateKernel.LATENT_STATES
        ]
    
    @staticmethod
    def _to_dict(
        latent_states: Union[list[tf.Tensor], tuple[tf.Tensor]]
    ) -> dict[str, tf.Tensor]:        
        return dict(zip(LatentStateKernel.LATENT_STATES, latent_states))
    
    def get_kernel(
        self,
        num_adaptation_steps: int,
        target_accept_prob: float
    ) -> tfp.mcmc.HamiltonianMonteCarlo: # type: ignore
        
        def target_log_prob_fn(
            *latent_states: tf.Tensor
        ) -> tf.Tensor:  # shape: [B]
            return self.conditional_posterior.log_prob(
                latent_states=self._to_dict(latent_states)
            )
          
        inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=[
                tf.broadcast_to(step, shape=(self.num_chains, 1))
                for step in self.step_size    
            ],
            num_leapfrog_steps=self.num_leapfrog_steps,
        )

        if num_adaptation_steps == 0:
            return inner_kernel
       
        return tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob,
        )
    
    def sample_chain(
        self,
        num_results: tf.Tensor,
        num_burnin_steps: tf.Tensor,
        seed: Optional[tf.Tensor] = None,
        initial_states: Optional[dict[str, tf.Tensor]] = None
    ):
        seed = self.seed if seed is None else seed
        initial_seed, hmc_seed = tfp.random.split_seed(seed)    
        
        if initial_states is None:
            initial_states = (
                self.conditional_posterior.latent_state_prior.sample(
                    seed=initial_seed,
                    num_samples=self.num_chains
                )
            )
        
        kernel = self.get_kernel(
            num_adaptation_steps=tf.cast(
                4 * num_burnin_steps / 5, 
                dtype=tf.int32
            ),
            target_accept_prob=0.75,
        )
        
        def trace_fn(states, previous_kernel_results):
            
            def print_remaining_steps():
                tf.print(
                    f"Currently on step {previous_kernel_results.step},",
                    f"with {num_results + num_burnin_steps 
                    - previous_kernel_results.step} steps remaining.",
                )
                return 0
            
            tf.cond(
                previous_kernel_results.step % 100 == 0,
                print_remaining_steps,
                lambda: 0   
            )
            
            return {
                "target_log_prob": (
                    previous_kernel_results
                    .inner_results
                    .accepted_results
                    .target_log_prob
                ),
                "is_accepted": (
                    previous_kernel_results
                    .inner_results
                    .is_accepted
                ),
                "log_accept_ratio": (
                    previous_kernel_results
                    .inner_results
                    .log_accept_ratio
                ),
                "step_size": (
                    previous_kernel_results
                    .inner_results
                    .accepted_results
                    .step_size
                )
            }
            
        states, trace, final_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=self._to_list(initial_states),
            kernel=kernel,
            seed=hmc_seed,
            trace_fn=trace_fn,
            return_final_kernel_results=True,
        )

        return self._to_dict(states), trace, final_kernel_results