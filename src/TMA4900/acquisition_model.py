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
        
        self.height_threshold = tf.constant(  # shape: [R]
            config["acquisition"]["height_threshold"],
            dtype=tf.float32
        )  
        self.acquisition_noise_variance = tf.constant(  # shape: [R]
            config["acquisition"]["acquisition_noise_variance"],
            dtype=tf.float32
        )
    
    @tf.function
    def log_likelihood(
        self,
        measurements: tf.Tensor,  # shape: [B, T, R]
        heights: tf.Tensor  # shape: [B, T, R]
    ) -> tf.Tensor: # shape: [B]
            
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