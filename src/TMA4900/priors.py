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
        
        self.B = tf.constant(
            config["general"]["number_of_batches"],
            dtype=tf.int32
        )
        self.R = tf.constant(
            config["general"]["number_of_reservoirs"],
            dtype=tf.int32
        )
        self.seed = tf.constant(
            config["general"]["seed"],
            dtype=tf.int32
        )

        self.E = self.R * (self.R - 1) // 2
    
        self.distribution = self._construct_distribution(
            config=config["prior"]["latent_states"]
        )
    
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
        
        B = num_samples if num_samples is not None else self.B
        seed = seed if seed is not None else self.seed
        
        return self.distribution.sample(B, seed=seed)
    
    @tf.function
    def log_prob(
        self,
        latent_states: tf.Tensor
    ) -> tf.Tensor:
        return self.distribution.log_prob(latent_states)
        
    
    
class GraphPrior(tf.Module):
    
    def __init__(
        self, 
        config: dict[str, Any],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        
        self.B = tf.constant(
            config["general"]["number_of_batches"],
            dtype=tf.int32
        )
        self.R = tf.constant(
            config["general"]["number_of_reservoirs"],
            dtype=tf.int32
        )
        self.seed = tf.constant(
            config["general"]["seed"],
            dtype=tf.int32
        )
        
        # the topmost reservoir has no outgoing edges and the 2nd topmost 
        # reservoir has only 1 outgoing edge (the one to the topmost reservoir)
        # the number of "stochastic" reservoirs are therefore 2 less than the 
        # total number of reservoirs    
        self.S = self.R - 2

        self.log_prob_num_paths = self._get_log_prob_num_paths(
            config=config["prior"]["graph"]
        )
        self.log_weight_paths = self._get_log_weight_paths(
            config=config["prior"]["graph"]
        )
    
    def _get_log_prob_num_paths(
        self,
        config: dict[str, Any]
    ) -> tf.Tensor:
        
        # get sparsity parameter from config
        sparsity_parameter = tf.constant(
            config["sparsity_parameter"],
            dtype=tf.float32
        )[:self.S]
        
        # possible values taken by kappa_i (the number of outgoing paths from 
        # the ith reservoir), for 1 < i <= S, note that some values lie outside
        # the support of the prior (these will be handled later)
        possible_values = tf.broadcast_to(  # shape: [R-1, S]
            tf.range(1.0, self.R)[:, None],
            shape=[self.R-1, self.S]
        )
        
        # compute the unnormalised log probability of kappa_i 
        poisson = tfd.Poisson(rate=sparsity_parameter)
        unormalised_log_prob = (  # shape: [R-1, S] 
            poisson.log_prob(possible_values)
        )
        
        # normalise the above log probabilities, using the maximum value taken 
        # by kappa_i (the number of outgoing paths from the ith reservoir)
        max_value = tf.range(self.R-1, 1.0, delta=-1.0)
        log_normalisers = tf.squeeze(tf.math.log(  # shape: [S]
            1.0 
            - poisson.survival_function(max_value)
            - poisson.cdf(tf.zeros_like(max_value))
        ))
        log_prob = (  # shape: [R-1, S]
            unormalised_log_prob - log_normalisers[None, :] 
        )
        
        # set the log probability of values outside the support of the prior to
        # negative infinity 
        log_prob = tf.where(
            possible_values <= max_value,
            log_prob,
            -np.inf
        )
        
        # return the transpose of the resulting probabilities, so that 
        # log_prob[i] corresponds to the log probability of values taken by 
        # kappa_i
        return tf.transpose(log_prob)  # shape: [S, R-1]
        
    def _get_log_weight_paths(
        self,
        config: dict[str, Any]
    ) -> tf.Tensor:
        
        # get distance parameter from config
        distance_parameter = tf.constant(
            config["distance_parameter"],
            dtype=tf.float32
        )[:self.S]
        
        # create a mask indicating the length of a path, i.e. 
        # distance_mask[i, j] should equal the length of a path from reservoir 
        # 1 < i < S to reservoir 1 < j < R 
        i, j = tf.range(self.S)[:, None], tf.range(self.R)[None, :]
        distance_mask = tf.cast(j - i, dtype=tf.float32)  # shape: [S, R]
        
        # compute the (unnormalised) log weights associated with each path,
        # where invalid paths, e.g. from reservoir 2 to reservoir 1, receive a 
        # log weight equal to negative infinity
        unnormalised_log_weights = tf.where(  # shape: [S, R]
            distance_mask > 0.0,
            - distance_parameter[:, None] * distance_mask,
            - np.inf
        )
           
        def iterate_log_ESPs(
            previous_log_ESPs: tf.Tensor,  # shape: [S, R]
            distance: tf.Tensor  # shape: [1]
        ) -> tf.Tensor:
            """ Iteration step for the computation of log ESPs.
            
            Implements a single iteration step for the computation of log 
            elementary symmetric polynomials (log ESPs) in the (unnormalised) 
            log weight of each path.
            
            The iteration is taken over the number of arguments of each log ESP,
            which corresponds to the maximum distance of the paths considered. 
            I.e., at iteration n, only paths of length less than or equal to n
            are considered as arguments for the log ESPs.

            Args:
                previous_log_ESPs (tf.Tensor): Tensor with shape [S, R] whose 
                (i,j)th element represents the ith reservoir's jth log ESP. 
                distance (tf.Tensor): Maximum distance of the paths considered, 
                i.e. the number of arguments of each log ESP.

            Returns:
                tf.Tensor: New log ESPs.
            """
            
            # distance of the longest possible outgoing path per reservoir
            max_distance = tf.range(self.R-1, 1.0, delta=-1.0)  # shape: [S]
            
            # compute the (unnormalised) log weight of each path of length 
            # "distance" in the graph structure, if no such path exists for a 
            # given reservoir, the corresponding log probability is set to 
            # negative infinity 
            log_weight = tf.where(  # shape: [S]
                distance <= max_distance,
                - distance_parameter * distance,
                - np.inf
            )
            
            # shift entries in "previous_log_ESPs" to the right, such that the 
            # (i,j)th entry in "shifted" corresponds to the ith reservoir's 
            # (j-1)th log ESP (provided that j >= 1)
            # note that the value of the first column, indexed by j = 0, is set
            # to negative infinity 
            # this is done so that the iterative step produces log ESPs that 
            # respect the boundary conditions of the algorithm, namely that the 
            # 0th log ESP is equal to 0
            shifted = tf.pad(  # shape: [S, R]
                previous_log_ESPs[:, :-1],
                paddings=[[0, 0], [1, 0]],
                constant_values=-np.inf
            )
            
            # iterative step, see main report for additional details
            return tf.reduce_logsumexp(tf.stack([  # shape: [S, R]
                previous_log_ESPs,
                log_weight[:, None] + shifted
            ], axis=0), axis=0)

        
        # initializer for the iterative algorithm computing log elementary 
        # symmetric polynomials (log ESPs) in the (unnormalised) log weight of 
        # each path
        initializer = tf.concat([  # shape: [S, R]
            tf.zeros([self.S, 1], dtype=tf.float32),
            - np.inf * tf.ones([self.S, self.R-1], dtype=tf.float32)
        ], axis=1)
        
        # iterate over paths of length 1, 2, ..., R-1 (longest possible path, 
        # going from reservoir 1 to reservoir R), to produce a set of log ESPs 
        log_ESPs = tf.scan(  # shape: [R-1, S, R]
            fn=iterate_log_ESPs,
            elems=tf.range(1.0, self.R),
            initializer=initializer
        )
        log_ESPs = tf.concat(  # shape: [R, S, R]
            [initializer[None, :, :], log_ESPs], 
            axis=0
        )
        
        
        return {
            "unnormalised_log_weights": unnormalised_log_weights,  
            "log_ESPs": log_ESPs
        }
       
    @tf.function
    def _construct_graph_encoding_from_sampled_edges(
        self,
        sampled_paths: tf.Tensor  # shape: [B, S, R-1]
    ) -> tf.Tensor:  # shape: [B, R, R]
        
        B = tf.shape(sampled_paths)[0]
        
        shift = tf.math.mod(  # shape: [S, R-1]
            tf.range(self.R-1)[None, :] - tf.range(self.S)[:, None],
            self.R-1 
        ) 
        indices = tf.broadcast_to(  # shape: [B, S, R-1]
            shift[None, :, :],
            shape=[B, self.S, self.R-1]
        )
        sampled_paths = tf.gather(  # shape: [B, S, R-1]
            sampled_paths,
            indices=indices,
            axis=2,
            batch_dims=2
        )
        
        graph_encodings = tf.concat([  # shape: [B, S, R]
            tf.zeros((B, self.S, 1), dtype=tf.bool),
            sampled_paths     
        ], axis=2)
        
        graph_encodings = tf.concat([  # shape: [B, R, R]
            graph_encodings,
            tf.broadcast_to(
                tf.cast(
                    tf.one_hot(self.R-1, self.R)[None, None, :],
                    dtype=tf.bool
                ),
                shape=[B, 1, self.R]
            ),
            tf.zeros((B, 1, self.R), dtype=tf.bool)
        ], axis=1)
        
        return graph_encodings 
        
    @tf.function
    def log_prob(
        self,
        graph_encoding: tf.Tensor  # shape: [B, R, R]
    ) -> tf.Tensor:
        
        B = tf.shape(graph_encoding)[0]
        
        graph_encoding_bool = tf.cast(graph_encoding, dtype=tf.bool)
        graph_encoding_float = tf.cast(graph_encoding, dtype=tf.float32)
        
        # determine whether the graph encoding corresponds to a directed 
        # acyclic graph (DAG), in which case it is strictly upper triangular
        is_strictly_upper_triangular = tf.reduce_all(tf.equal(
            graph_encoding_float, 
            tf.linalg.band_part(graph_encoding_float, 0, -1)
        ), axis=[1, 2])
        # also determine whether all reservoirs have at least a single outgoing
        # migration path (except the topmost reservoir)
        has_path_to_top_reservoir = tf.reduce_all(tf.reduce_any(
            graph_encoding_bool[:, :-1, :],
            axis=2
        ), axis=1)
        
        graph_is_valid = tf.logical_and(
            is_strictly_upper_triangular,
            has_path_to_top_reservoir
        )
        
        
        # if the graph encoding does not satisfy the above conditions, it 
        # represents a graph structure that lie outside the support of the prior 
        #if not (is_strictly_upper_triangular and has_path_to_top_reservoir):
           #return tf.constant(-np.inf, dtype=tf.float32)
        
        # compute the number of paths emerging from each "stochastic" reservoir,
        # i.e. all but the top 2 reservoirs who have 1 and 0 outgoing paths
        # respectively  
        num_paths = tf.maximum(1, tf.cast(  # shape: [S] 
            tf.reduce_sum(graph_encoding_float[:, :-2, :], axis=2),
            dtype=tf.int32  
        ))
        
        # add contribution from the number of outgoing paths per reservoir    
        indices = tf.stack([ 
            tf.broadcast_to(tf.range(self.S)[None, :], shape=[B, self.S]),
            num_paths - 1
        ], axis=2)
        log_prob = tf.reduce_sum(tf.gather_nd(
            self.log_prob_num_paths, 
            indices=indices
        ), axis=1)
        
        # add (unnormalised) contribution from the location of the paths
        log_prob += tf.reduce_sum(
            tf.where(
            graph_encoding_bool[:, :-2, :],  # consider stochastic reservoirs only
            tf.broadcast_to(
                self.log_weight_paths["unnormalised_log_weights"][None, :],
                shape=[B, self.S, self.R]
            ),
            0.0
        ),
            axis=[1, 2]
        )
 
        # normalise the log probability, by subtracting each reservoir's 
        # correct log elementary symmetric polynomial (log ESP), the indices of 
        # which correspond to the last iteration of the iterative algorithm, as
        # we need to consider all possible weights when normalising (i.e. 
        # weights of every length), the reservoir in which the paths originate
        # and the corresponding number of outgoing paths from that reservoir
        indices = tf.stack([ 
            tf.fill((B, self.S), value=(self.R - 1)),  # last iteration
            tf.broadcast_to(tf.range(self.S)[None, :], shape=(B, self.S)),  # outgoing reservoirs
            num_paths  # corresponding number of paths 
        ], axis=2)
        log_prob -= tf.reduce_sum(tf.gather_nd(
            self.log_weight_paths["log_ESPs"],
            indices=indices
        ), axis=1)
        
        return tf.where(
            graph_is_valid,
            log_prob,
            -np.inf 
        )
    
    @tf.function
    def sample(
        self,
        num_samples: Optional[tf.Tensor] = None,
        seed: Optional[tf.Tensor] = None
    ) -> tf.Tensor:      

        B = num_samples if num_samples is not None else self.B
        seed = seed if seed is not None else self.seed
        
        categorical_seed, uniform_seed = tfp.random.split_seed(seed)
        
        categorical_samples = tf.random.stateless_categorical(  # shape: [S, B]
            self.log_prob_num_paths,
            num_samples=B,
            seed=categorical_seed,
            dtype=tf.int32
        )
        
        num_paths = 1 + tf.transpose(categorical_samples)  # shape: [B, S]

        log_uniform_samples = tf.math.log(  # shape: [B, S, R-1]
            tf.random.stateless_uniform(
                [B, self.S, self.R-1],
                seed=uniform_seed,
                dtype=tf.float32
            )
        )
        
        def gather_log_weights(
            distance: tf.Tensor  # shape: [ ]
        ) -> tf.Tensor:  # shape: [S]

            i = tf.range(self.S)  # shape: [S]
            j = tf.math.mod(i + distance, self.R)  # shape: [S]
            
            return tf.gather_nd(
                self.log_weight_paths["unnormalised_log_weights"],
                indices=tf.stack([i, j], axis=1)
            )
            
        def gather_log_ESPs(
            distance: tf.Tensor,  # shape: [ ]
            num_paths_remaining: tf.Tensor  # shape: [B, S]
        ) -> tf.Tensor:  # shape: [S, B]
            
            distance_idx = tf.broadcast_to(  # shape: [B, S]
                distance,
                shape=[B, self.S]
            ) 
            reservoir_idx = tf.broadcast_to(  # shape: [B, S]
                tf.range(self.S)[None, :], 
                shape=[B, self.S]
            )
            
            indices = tf.stack([  # shape: [B, S, 3]
                distance_idx,
                reservoir_idx,
                tf.maximum(num_paths_remaining, 0) 
            ], axis=2)
            
            return tf.gather_nd(
                self.log_weight_paths["log_ESPs"],
                indices=indices
            )
        
        def cond(
            distance: tf.Tensor,  # shape: [ ] 
            num_paths_remaining: tf.Tensor,  # shape: [S, B] 
            paths_array: Any
        ):
            return distance > 0
        
        def body(
            distance, 
            num_paths_remaining, 
            paths_array
        ):
            log_prob = tf.where(
                num_paths_remaining > 0,
                gather_log_weights(distance)[None, :] 
                + gather_log_ESPs(distance-1, num_paths_remaining-1) 
                - gather_log_ESPs(distance, num_paths_remaining),
                -np.inf
            ) 
            
            edge_is_present = (log_prob > log_uniform_samples[:, :, distance-1])
             
            num_paths_remaining -= tf.cast(edge_is_present, dtype=tf.int32)
            
            paths_array = paths_array.write(
                distance - 1,
                edge_is_present
            )
            
            return distance - 1, num_paths_remaining, paths_array
    
        initial_states = (
            self.R-1, 
            num_paths,
            tf.TensorArray(
                dtype=tf.bool, 
                size=self.R-1, 
                dynamic_size=False, 
                clear_after_read=False
            )
        )
        
        result = tf.while_loop(cond, body, initial_states)
        num_paths_remaining = result[1]
        sampled_paths = tf.transpose(
            result[2].stack(),
            perm=(1, 2, 0)
        )
        
        tf.debugging.assert_equal(num_paths_remaining, 0)
        
        return self._construct_graph_encoding_from_sampled_edges(
            sampled_paths
        )