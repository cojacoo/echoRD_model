# echoRD model — eco-hydrological particle model based on representative structured domains

This repository is accopmanying the publication Jackisch & Zehe 2015 in Water Resources Research (currently submitted). The echoRD model is a novel Lagrangian stochastic-physical hydrological model framework simulating soil water flow by means of a space domain random walk of water particles in a representative, structured model domain.

This repository is providing:
1. the echoRD model
2. testcases and setups of the model
3. reference data
4. stored model runs

For all theory and detailed description, please refer to our paper.

## The echoRD model
Core of the model is the description of water itself as particles. In order to conserve the topology of macroporous soil structures such as earthworm burrows, cracks and root networks the model domain consists of a representative ensemble of 1D macropores and a continuous 2D matrix domain (with cyclic lateral boundary).

The model is found in folder ./echoRD

To run the model the folder testcases holds ~run_echoRD.py~ with a collection of controllers of the model core plus several plotting functions.

## Testcases
In line with our publication, there are several tests with different level of complexity in the model. Although it refers to the same model core, especially the 1D version and the artificial macropore are very special setups.

### 1D non-linear space domain model for diffusive flux
[echoRD 1D] holds the first testcase comparing the space domain random walk approach agains two solvers of the Richards equation and observed nocturnal diffusion. This script requires SimPEG Flow as reference model.

### 1D echoRD for sprinkler experiment
echoRD1D Sprinkler 



## Disclaimer and dependencies
The model is developed and tested based on Python 2.7.6. The examples are given as IPython 1.1 Notebooks and as standalone scripts. The packages NumPy, SciPy, Pandas and Matplotlib are required. The preprocessor requests more specific packages as outlined there.

All software and data is given under GNU General Public License (GPLv3) and Creative Commons License (CC BY-NC-SA 4.0) respectively. *This is scientific, experimental code without any warranty nor liability in any case.* However, you are invited to use, test and expand the model at your own risk. If you do so, please contact me to keep informed about bugs and modifications.
