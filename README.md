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
echoRD1D Sprinkler is adding advection to the 1D random walk which is based on observed tracer recovery profiles. Diffusive and advective flow is jointly modelled in a lumed manner. The result is, that without a criterion to stop the advection the new, advective particles simply bypass the domain sooner or later.

### 2D echoRD for column experiment
The full echoRD model was referred to an experiment with one centred "artificial macropore" (coarse sand) in a half-cylindrical sandbox (quartzite sand). 2D column trial holds a script analysing the observations from the experiment, where the "artificial macropore" was irrigated with constant flux of 3.8 l/h, and a comparative model run.

### 2D echoRD Weiherbach
The full echoRD model requires data about macropore depth and density distribution. In order to reduce the effect of preprocessing we set up a run based on observed macropore settings in the Weiherbach basin (the observations are converted into horizontal images which are then interpreted by the preprocessor). ~echoRD2D_sprinkler_Weiherbach~ holds the model setup for the testcase recalculating a sprinkler experiment.

### 2D echoRD Process Hypotheses
The echoRD model has several process hypotheses for infiltration, advective velocity definition, macropore-matrix exchange and flow in the macropores. While all alternative hypotheses are included in the model (controlled by respective flags) the script ~echoRD2D_sprinkler_Weiherbach_Comparison~ reads the results of all combinations and compares them.

### 2D echoRD Noised Ks
It is well known that local heterogeneity of pedo-physical properties may emerge local disequilibrium and preferential processes. We used the Weiherbach setup and imposed a noised saturated hydraulic conductivity as primary control for diffusive soil water dynamics in the range of observed variance. ~ echoRD2D_sprinkler_Weiherbach_noiseKs~ presents the respective model setup.

### 2D echoRD Attert
In the northern Attert basin sprinkler experiments revealed a complex geogene macropore setting in young soils on periglacial deposits. Based on field and laboratory analyses the echoRD model is also initialised with photos of excavated horizontal dye stains in different depths. ~echoRD_sprinkler_Attert~ holds the model setup for the testcase recalculating the sprinkler experiment in a less well-defined setting.

## General Helpers
Since the model includes a preprocessor the script ~Preprocessing~ is intended to help the setup for any other plot. Please note, that I use pickle to store the setups for further usage.

### In case of any questions, pls. feel free to ask.

## Disclaimer and dependencies
The model is developed and tested based on Python 2.7.6. The examples are given as IPython 1.1 Notebooks and as standalone scripts. The packages NumPy, SciPy, Pandas and Matplotlib are always required. The preprocessor requests more specific packages as outlined there.

All software and data is given under GNU General Public License (GPLv3) and Creative Commons License (CC BY-NC-SA 4.0) respectively. *This is scientific, experimental code without any warranty nor liability in any case.* However, you are invited to use, test and expand the model at your own risk. If you do so, please contact me to keep informed about bugs and modifications.
