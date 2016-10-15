Automatic, Robust, and Globally Optimal Segmentation of Tubular Structures
==========================================================================

Presented at MICCAI 2016, Athens, Greece.

Purpose
=======

The provided code recreates the phantom experiment that is presented in our
MICCAI 2016 paper

> Simon Pezold, Antal Horváth, Ketut Fundana, Charidimos Tsagkas, Michaela
> Andělová, Katrin Weier, Michael Amann, and Philippe C. Cattin: Automatic,
> Robust, and Globally Optimal Segmentation of Tubular Structures.

The camera-ready version of the paper can be found as `pdfs/pezold2016.pdf`.

In the experiment, we compare isotropic total variation (TV) with anisotropic
total variation regularization (ATV) for tubular structure segmentation. For a
fair comparison, we make a grid search over the two parameters of the
nonterminal cost/capacity, which we modeled as an edge detector term (see
paper for equation). Running the full experiment thus might take some hours. 
Note that since running the experiment for the actual publication, we widened
the search range for the parameters a bit, which slightly increases the Dice
coefficients for TV (TV dice at level 1.5 before widening search range: 0.8837,
after: 0.8895, ATV dice at level 1.5: 0.9332). The original search range can
still be found as a comment in `phantom_experiment.py`.

The experiment can be launched via

```
python src/phantom_experiment.py
```
    
using Python 2.X. For speeding up the result presentation, the following lines
in `phantom_experiment.py` should be commented out:

```python
self.nt_weightings = np.logspace(-2.5, 0.5, 24 + 1)
self.nt_decays     = np.logspace(-1, 3, 16 + 1)
self.noise_levels  = np.linspace(0.1, 1.5, 15)
```

They should be replaced by:

```python
self.nt_weightings = [10 ** -1.625, 10 ** -1.25]
self.nt_decays     = [10 ** 1, 10 ** 1.5]
self.noise_levels  = [1.5]
```

which will result in calculating the best combinations for the highest noise
level only.

Requirements
============
Currently, the provided code runs on Python 2.X only. Apart from `numpy` and
`scipy`, it requires the following packages:

* `mayavi`
* `pyopencl`

Note that `pyopencl` needs OpenCL-capable hardware and drivers. I only
successfully ran the code on Nvidia GPUs so far. Compiling the `.cl` files for
use with an Intel driver caused me problems with includes that were not found.
Any help or advice is greatly appreciated.

Provided Functionality
======================

* `src/helpers/helix_phantom.py`: create the noise-free phantom image volumes
* `src/processing/qfrangi_gpu.py`: calculate Frangi's vesselness feature on the
GPU
* `src/processing/adjust_directions.py`: adjust the vesselness main directions
so that neighboring vectors point in approximately the same rather than the
opposite direction, as described in Section 2.3 of our paper (mostly on the
GPU).
* `src/processing/gradient_vector_flow.py`: calculate gradient vector flow on
the GPU
* `src/processing/continuous_cut_anisotropic_gpu.py`: actual segmentation code,
running on the GPU
