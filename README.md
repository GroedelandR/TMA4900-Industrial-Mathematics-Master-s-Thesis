# TMA4900 Industrial Mathematics Master's Thesis



## General Info

This repository is a consolidation of the source code produced as part of the author's Master's thesis in  
Industrial Mathematics, at the Norwegian University of Science and Technology.  

The code is written in Python, and relies heavily on [**TensorFlow**](https://www.tensorflow.org/api_docs/python/tf/all_symbols)  to vectorize, and thus speed up, various computations.  
As described in the thesis, [**TensorFlow Probability**](https://www.tensorflow.org/probability/api_docs/python/tfp)  is used to sample from the posterior distribution,  
using Hamiltonian Monte Carlo.  
Note that we have also used TensorFlow Probability to sample from the prior distributions considered in the thesis.  
A combination of [**Pandas**](https://pandas.pydata.org/docs/reference/index.html#api) and [**Plotnine**](https://plotnine.org/reference/) was then used to create the various plots used throughout the thesis.   

The code is licenced under a general MIT licence, see **LICENSE**.



## Python Environment

To make sure that the correct versions of each library is loaded (noting that we have also used other libraries  
such as NumPy and SciPy to a lesser degree), the user may load all the necessary dependencies from the provided  
**pyproject.toml** file, by cloning the repository and running the command  

*python -m pip install -e .*

in the directory where the aforementioned file resides.  
Note that this requires Python and pip to be preinstalled on the computer running the command.  



## Description of the Files in the Repository

The consolidated source code is found in the directory **src/TMA4900**.  
Below is a brief description of each of the files within this directory,  

- **__init__.py** Marks the parent directory as a Python package.

- **simulator.py** Contains the class *Simulator*, which is an implementation of the simulator discussed in the thesis.  
To simulate the plume heights at time *t*, the method *Simulator.simulate()* is called.
 
- **priors.py** Contains the classes *LatentStatePrior* and *GraphPrior*, which are implementations of the prior  
on unconstrained continuous parameters $\boldsymbol{\tilde \theta}$ and DAGs $G$ respectively.  
We note that in the case of the unconstrained continuous parameters, the simulator makes the necessary transformations  
to obtain the corresponding constrained parameters $\boldsymbol{\theta}$.  
This is done so that the McMC kernel can work in unconstrained space.  
To sample from the priors, the methods *LatentStatePrior.sample()* and *GraphPrior.sample()* are called.  
To compute the log-probability of a specific sample, the methods *LatentStatePrior.log_prob()* and  
*GraphPrior.log_prob()* are called instead.

- **acquisition_model.py** Contains the class *AcquisitionModel*, which is an implementation of the  
acquisition/likelihood model discussed in the thesis.  
To compute the log-likelihood of a given observation, the method *AcquisitionModel.log_likelihood()* is called.

- **mcmc.py** Contains the classes *ConditionalPosterior* and *LatentStateKernel*, which are implementations of  
the full conditional distribution of unconstrained parameters $\boldsymbol{\tilde \theta}$ and the HMC kernel  
respectively.  
To evaluate the full conditional log-probability of $\boldsymbol{\tilde \theta}$, the method  
*ConditionalPosterior.log_prob()* is called.  
To sample a Markov chain on $\boldsymbol{\tilde \theta}$, the method *LatentStateKernel.sample_chain()*  
is called instead.  
Note that to obtain the corresponding constrained parameters $\boldsymbol{\theta}$, the internal method  
*Simulator._get_states()* have to be called.  
This method is called automatically when calling *Simulator.simulator()* however.  

- **utils.py** Contains various functions for easy plotting.  

The interactive python notebook **figures.ipynb** in the **src** directory then uses the methods described above to  
create the various plots included in the thesis.  

Apart from these files, the repository also includes the following,

- **config.toml** Configures the aforementioned methods, e.g. by providing necessary parameter settings.

- **.gitignore** Blacklists metadata from being included in the repository when creating commits/pushes.

- **LICENSE** MIT license.

- **README.mc** The current markdown file you are reading.

- **pyproject.toml** Lists the dependencies of the source code.
