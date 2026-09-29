import numpy as np
from typing import Optional, Any, Literal
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class LatentStatePrior(tf.Module):
    
    def __init__(
        self,
        config: dict[str, Any],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        
        # number of batches and reservoirs 
        self.B = tf.constant(
            config["general"]["number_of_batches"],
            dtype=tf.int32
        )
        self.R = tf.constant(
            config["general"]["number_of_reservoirs"],
            dtype=tf.int32
        )
        
        # seed for reproducibility
        self.seed = tf.constant(
            config["general"]["seed"],
            dtype=tf.int32
        )

        # number of arcs/edges in saturated system
        self.E = self.R * (self.R - 1) // 2

        # prior distribution
        self.distribution = self._construct_distribution(
            config=config["prior"]["latent_states"]
        )
    
    # function to construct prior distribution
    def _construct_distribution(
        self,
        config: dict[str, Any]
    ) -> tfd.Distribution:  # type: ignore
        
        def get_distribution(
            parameter: Literal[
                "log_lateral", "log_vertical", "log_threshold_pressure",
                "unconstrained_flow", "unconstrained_activation"]
        ) -> tfd.Distribution:  # type: ignore
            
            mean = tf.constant(
                config[parameter]["mean"], 
                dtype=tf.float32
            )
            variance = tf.constant(
                config[parameter]["variance"], 
                dtype=tf.float32
            ) 
            correlation = tf.constant(
                config[parameter]["correlation"], 
                dtype=tf.float32
            )
            
            num_dims = self.E if parameter in (
                "unconstrained_flow", "unconstrained_activation"
            ) else self.R
                   
            return tfd.MultivariateNormalTriL(
                loc=mean[:num_dims],
                scale_tril=tf.linalg.cholesky(
                    variance*correlation[:num_dims, :num_dims]
                )
            )
                
        return tfd.JointDistributionNamed({
            "log_lateral_parameter": 
                get_distribution("log_lateral"),
            "log_vertical_parameter": 
                get_distribution("log_vertical"),
            "log_threshold_pressure": 
                get_distribution("log_threshold_pressure"),
            "unconstrained_flow": 
                get_distribution("unconstrained_flow"),
            "unconstrained_activation": 
                get_distribution("unconstrained_activation")
        },
            batch_ndims=1,
            use_vectorized_map=True,
            validate_args=True
        )

    @tf.function
    def sample(
        self,
        num_samples: Optional[tf.Tensor] = None,
        seed: Optional[tf.Tensor] = None
    ) -> dict[str, tf.Tensor]:   
        """
        Sample from the prior distribution.

        Args:
            num_samples: Number of samples to generate from the prior. 
            If unspecified, generate 1 sample for each batch.
            seed: Seed for reproducibility. 
            If unspecified, default to internal seed.

        Returns:
            Sampled latent states (tilde theta).
        """
             
        
        B = num_samples if num_samples is not None else self.B
        seed = seed if seed is not None else self.seed
        
        return self.distribution.sample(B, seed=seed)
    
    @tf.function
    def log_prob(
        self,
        latent_states: tf.Tensor
    ) -> tf.Tensor:
        """
        Evaluate log-probability of latent states (tilde theta).

        Args:
            latent_states: Latent states (tilde theta).

        Returns:
            Computed log-probability.
        """
        return self.distribution.log_prob(latent_states)