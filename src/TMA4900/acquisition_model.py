import numpy as np
from typing import Any
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class AcquisitionModel(tf.Module):
    
    def __init__(
        self, 
        config: dict[str, Any],
        name = None
    ) -> None:
        super().__init__(name)
        
        # measurement threshold
        self.height_threshold = tf.constant(
            config["acquisition"]["height_threshold"],
            dtype=tf.float32
        )  
        
        # model noise variance
        self.acquisition_noise_variance = tf.constant(
            config["acquisition"]["acquisition_noise_variance"],
            dtype=tf.float32
        )
    
    @tf.function
    def log_likelihood(
        self,
        measurements: tf.Tensor,
        heights: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the log-likelihood of the plume heights for a given set of measurements.

        Args:
            measurements: Observed CO2 plume heights.
            heights: Actual CO2 plume heights.

        Returns:
            Computed log-likelihood.
        """
            
        kernel = tfd.Normal(
            loc=heights,
            scale=tf.math.sqrt(self.acquisition_noise_variance)
        )
        
        components = tf.where(
            measurements == 0.0, 
            kernel.log_cdf(self.height_threshold),
            # make sure uncensored measurements are higher than the threshold 
            tf.where(
                measurements >= self.height_threshold,
                kernel.log_prob(measurements),
                -np.inf
            )
        )    
        
        return tf.reduce_sum(components, axis=[1, 2])