#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import itertools
from mayavi import mlab
import numpy as np
import scipy.ndimage
from scipy.ndimage.morphology import binary_dilation

from helpers.cl_helpers import ClProgramLoader
from helpers.helix_phantom import create_toric_helix
from helpers.misc import normalize
from helpers.parameters import Parameters
from helpers.status import Status
from processing.adjust_directions import AdvancedAdjuster, TransformationCompleter
from processing.continuous_cut_anisotropic_gpu import CmfCutQAnisotropicGpu
from processing.gradient_vector_flow import GvfParameters, GvfGpu
from processing.qfrangi_gpu import VesselnessCalculation, FeatureFilter
from collections import namedtuple

"""
Create helical phantom, add Gaussian noise to its volume image, try to segment
it (a) with (isotropic) total variation (TV) regularization, (b) with
anisotropic total variation (ATV) regularization.
"""

def normpdf(x, mean, var):
    """
    Return the values on a normal distribution with the given <mean> and
    variance <var> for the given <x> values.
    """
    return (np.exp((x - mean) ** 2 * (1 / (-2 * var))) * 
            (1 / np.sqrt(2 * np.pi * var)))


def fast_dice_for(segmentation, ground_truth):
    """
    Calculate a fast estimate of the Dice coefficient by simply taking the
    whole images into account, i.e. not ignoring unconnected noisy parts
    detected as foreground.
    """
    seg_binarized = segmentation >= 0.5
    
    intersection = np.multiply(seg_binarized, ground_truth)
    # ^ "logical and", allowing for floats
    dice = 2 * np.sum(intersection) / (np.sum(seg_binarized) + np.sum(ground_truth))
    return dice


def select_largest(segmentation, return_size=False, threshold=0.5):
    """
    Return the largest connnected component in <segmentation>, after
    thresholding the "almost binary" segmentation with the given <threshold>.
    
    Return
        If <return_size> is True: Return (component, size), where <component>
        is the largest connected component and <size> is its number of voxels;
        if <return_size> is False: return the largest connected component
    """
    
    seg_binarized = segmentation >= threshold
    seg_labeled = scipy.ndimage.label(seg_binarized)[0]
    labels = np.unique(seg_labeled)[1:]  # "1:" to exclude the background
    
    # Find the largest blob
    seg_largest = None
    size_largest = -1
    
    for l in labels:
        
        seg_l = (seg_labeled == l)
        size_l = np.sum(seg_l)
        
        if size_l > size_largest:
            
            seg_largest = seg_l
            size_largest = size_l
            
    if return_size:
        return seg_largest, size_largest
    else:
        return seg_largest


def dice_for(segmentation, ground_truth):
    """
    Calculate the Dice coefficient for <segmentation> w.r.t. <ground_truth>,
    using the following steps: (1) Binarize the segmentation with a threshold
    of 0.5. (2) Determine the largest connected component in the binarized
    segmentation. (3) Calculate the Dice coefficient for the largest connected
    component w.r.t. <ground_truth>.
    
    <ground_truth>
        Should be binary.
    """
    seg_largest, size_largest = select_largest(segmentation, return_size=True)
            
    # Calculate the dice for the largest blob
    intersection = np.multiply(seg_largest, ground_truth)
    dice = 2 * np.sum(intersection) / (size_largest + np.sum(ground_truth))
    return dice


def show_surface(h):
    
    src = mlab.pipeline.scalar_field(h)
    mlab.pipeline.iso_surface(src, contours=[0.5])
    mlab.figure(mlab.gcf(), bgcolor=(1., 1., 1.,))
    mlab.show()


Result = namedtuple("Result", ["dice_true", "dice_fast", "seg", "nt_weighting", "nt_decay"])
Result.__str__ = lambda r : "Result(dice_true=%s, dice_fast=%s, seg=[not shown], nt_weighting=%s, nt_decay=%s)" % (r.dice_true, r.dice_fast, r.nt_weighting, r.nt_decay)


GvfCombination = namedtuple("GvfCombination", ["mu", "iterations"])

    
class PhantomExperiment(object):
    
    class HelixParameters(Parameters):
        
        def __init__(self):
            
            self.r_helix            = 25.0
            self.r_tube             = 6.0
            self.slope              = 0.1
            self.n_windings         = 3.0
            self.factor_helix       = 0.5
            self.factor_tube        = 0.5
            self.factor_slope       = 1.0
            
            self.points_per_circle  = 100
            self.points_per_winding = 100
            
            self.write_output       = False
            self.output_path        = None
            self.voxel_size         = (1, 1, 1)
            self.margins            = 0.0
            self.show               = False

    
    def __init__(self):

        self._loader = ClProgramLoader()
        
        # Parameterization
        self.helix_parameters = self.HelixParameters()
        self.ridge_thresholds = (0.01, 0.3)
        self.gvf_combination = GvfCombination(mu=10 ** 0.5, iterations=300)
        self.eps_reg = 1e-9
        self.a_atv = 10 ** (-1.5)
        self.a_tv  = 1
        
        # FIXME: This was the combination we used for the graphics in the paper
        # It gave a slight disadvantage to TV at higher noise levels, as the
        # nt_weighting was always at the lowest possible value, starting from
        # noise level 1.0 and higher. Allowing smaller values, like we do now,
        # does not change the observation of better performance for ATV though
        # (TV dice at level 1.5 before: 0.8837, after: 0.8895,
        # ATV dice at level 1.5: 0.9332)
        #
        # self.nt_weightings = np.logspace(-1.5, 0.5, 16 + 1)

        self.nt_weightings = np.logspace(-2.5, 0.5, 24 + 1)
        self.nt_decays     = np.logspace(-1, 3, 16 + 1)
        self.noise_levels  = np.linspace(0.1, 1.5, 15)

#         # FIXME: For fast display of the final results: just use the best
#         # combinations for ATV and TV regularization at the highest noise level
#         self.nt_weightings = [10 ** -1.625, 10 ** -1.25]
#         self.nt_decays     = [10 ** 1, 10 ** 1.5]
#         self.noise_levels  = [1.5]
         
        self.vfilter = self._create_vesselness_filter()
        self.dir_adjuster = self._create_direction_adjuster()
        self.dir_completer = TransformationCompleter(dirs=None, loader=self._loader)
        self.cutter = self._create_cutter()
        
    def create_helix_for(self, noise_level, noise_seed=42):
        """
        Create a toric helix phantom with tube radii in the range of 3--6
        voxels, with ground truth intensities in {0, 1} and additive Gaussian
        noise.
        
        <noise_level>
            Standard deviation of the Gaussian noise.
        <noise_seed>
            Seed for the noise pseudo random number generator (for reproducible
            results). If None, the seed will change every time.
        Return
            Two tuple (noisy, true), both three-dimensional Numpy arrays of
            same size, each dimension guaranteed to hold a multiple of 64
            voxels. <noisy> is the helix phantom with added noise (note that
            values may lie outside [0, 1]); <true> is the ground truth, i.e.
            the phantom without noise, with values in {0, 1}.
        """
        random = np.random.RandomState(seed=noise_seed)
        
        params = self.helix_parameters
        h = create_toric_helix(params.r_helix, params.r_tube,
                               params.slope, params.n_windings,
                               params.factor_helix, params.factor_tube,
                               params.factor_slope,
                               params.points_per_circle, params.points_per_winding,
                               params.write_output, params.output_path, 
                               params.voxel_size, params.margins, params.show)
        h = h / 100.
        
        # Make the size of h a multiple of 64
        h_padded = np.zeros([64 * np.ceil(s / 64.).astype(np.int) for s in h.shape], dtype=h.dtype)
        lower_bounds = [(s_p - s) // 2 for s_p, s in zip(h_padded.shape, h.shape)]
        
        h_padded[lower_bounds[0]:lower_bounds[0] + h.shape[0],
                 lower_bounds[1]:lower_bounds[1] + h.shape[1],
                 lower_bounds[2]:lower_bounds[2] + h.shape[2]] = h
                 
        # Create and add noise
        noise = random.normal(size=h_padded.shape, scale=noise_level)
        h_noisy = h_padded + noise
                 
        return h_noisy, h_padded

    def _create_vesselness_filter(self):
        
        hparams = self.helix_parameters
        r_min = np.minimum(hparams.r_tube, hparams.r_tube * hparams.factor_tube)
        r_max = np.maximum(hparams.r_tube, hparams.r_tube * hparams.factor_tube)
        
        vparams = VesselnessCalculation.Parameters()
        vparams.bright_objects = True
        vparams.scales = (r_min / 1.2, r_max * 1.2, 16)
        vfilter = FeatureFilter(vparams, self._loader)

        return vfilter
    
    def _create_direction_adjuster(self):
        
        dir_adjuster = AdvancedAdjuster(threshold=self.ridge_thresholds,
                                        qeigs=None, response=None,
                                        voxel_size=(1, 1, 1),
                                        ridge_detection_only=False,
                                        clean_ridge=True,
                                        loader=self._loader)
        return dir_adjuster
    
    def _create_cutter(self):
        
        params = CmfCutQAnisotropicGpu.CutParameters()
        params.lsize_3d = (64, 4, 1)
        params.error_bound = 1e-6
        params.num_iter = 3000
        params.update = 30
        
        cutter = CmfCutQAnisotropicGpu(params, cl_program_loader=self._loader)
        
        return cutter
    
    def _create_gvf_for(self, vfield):
        
        def callback(v, i, mag_delta):
            print i, mag_delta
            
        voxel_size = (1, 1, 1)
        
        params = GvfParameters()
        params.mu = self.gvf_combination.mu
        params.tol = 0
        params.maxiter = self.gvf_combination.iterations
        params.callback = callback
        params.update = params.maxiter // 10
        params.normalize = False
        
        gvf = GvfGpu(vfield, params, voxel_size, cl_program_loader=self._loader)
        
        return gvf
    
    def _calculate_bgfg_weights_for(self, image, noise_level):
        """
        Return background weights, foreground weights
        """
        var = noise_level ** 2
        eps = self.eps_reg
        
        bgp = normpdf(image, 0, var)
        fgp = normpdf(image, 1, var)
        
        bgp_hat = normpdf(0, 0, var)
        fgp_hat = normpdf(1, 1, var)
        
        p_hat = np.maximum(bgp_hat, fgp_hat)  # Should be the same, anyway
        
        r = np.log((fgp + eps) / (bgp + eps))
        rq = 1 / np.log((p_hat + eps) / eps)
        
        bgw = rq * np.maximum(-r, 0)
        fgw = rq * np.maximum( r, 0)
        
        return bgw, fgw
    
    def execute(self):
        
        vfilter = self.vfilter
        dir_adjuster = self.dir_adjuster
        dir_completer = self.dir_completer
        cutter = self.cutter
        
        num_combinations = len(self.nt_decays) * len(self.nt_weightings) * len(self.noise_levels)
        
        combination_count = 0
        
        results_atv = {}
        results_tv  = {}
        # key: noise_level, value: Result (named tuple, see above class)
        
        
        for noise_level in self.noise_levels[::-1]:
            # ^ Start with the most problematic ones
            
            s = Status("Render phantom (noise level: %s)" % noise_level)
            ph_noisy, ph_true = self.create_helix_for(noise_level)
            
            s.update("Calculate vesselness (noise level: %s)" % noise_level)
            vesselness, vdirs_raw, unused = vfilter.execute_for(ph_noisy, roi=None, voxel_size=(1, 1, 1))
            vesselness = normalize(vesselness, dst_min=0.0, dst_max=1.0, perc_low=None, perc_up=None, clip=True)
            
            s.update("Adjust directions for GVF (noise level: %s)" % noise_level)
            dir_adjuster.qeigs = vdirs_raw
            dir_adjuster.response = vesselness
            vdirs = dir_adjuster.execute()

            s.update("Apply GVF (noise level: %s)" % noise_level)
            vdirs_scaled = np.rollaxis(np.rollaxis(vdirs, -1) * vesselness, 0, vdirs.ndim)
            gvf = self._create_gvf_for(vdirs_scaled)
            vdirs_smoothed = gvf.execute()

            s.update("Complete main directions (noise level: %s)" % noise_level)
            dir_completer.dirs = vdirs_smoothed
            vdirs_completed = dir_completer.complete()
            
            s.update("Calculate weights (noise level: %s)" % noise_level)
            bgw, fgw = self._calculate_bgfg_weights_for(ph_noisy, noise_level)
            grad_mag_sq = np.linalg.norm(np.gradient(ph_noisy), axis=0) ** 2

            best_atv = Result(dice_true=None, dice_fast=0, seg=None, nt_weighting=None, nt_decay=None)
            best_tv =  Result(dice_true=None, dice_fast=0, seg=None, nt_weighting=None, nt_decay=None)
            
            for nt_weighting, nt_decay in itertools.product(self.nt_weightings, self.nt_decays):
                combination_count += 1
                
                print
                print
                s.update("Evaluation %s/%s: w=%s, sigma=%s (noise level: %s)" %
                         (combination_count, num_combinations, nt_weighting, nt_decay, noise_level))

                ss = Status("Calculate nonterminal weights")
                ntw = nt_weighting * np.exp(((-1.) / (nt_decay ** 2)) * grad_mag_sq)
                
                ss.update("Calculate the cut, ATV")
                eigs_atv = (1, self.a_atv, self.a_atv)
                seg_atv = cutter.execute(ph_noisy, vdirs_completed, eigs_atv, ntw, fgw, bgw, labels=None, scaling=(1, 1, 1))
                
                ss.update("Calculate the cut, TV")
                
                eigs_tv = (1, self.a_tv, self.a_tv)
                seg_tv = cutter.execute(ph_noisy, vdirs_completed, eigs_tv, ntw, fgw, bgw, labels=None, scaling=(1, 1, 1))
                
                ss.update("Evaluate Dice coefficients (fast)")
                dice_fast_atv = fast_dice_for(seg_atv, ph_true)
                dice_fast_tv  = fast_dice_for(seg_tv,  ph_true)
                print "ATV:", dice_fast_atv
                print "TV: ", dice_fast_tv
                if dice_fast_atv > best_atv.dice_fast:
                    best_atv = Result(dice_true=None, dice_fast=dice_fast_atv, seg=seg_atv, nt_weighting=nt_weighting, nt_decay=nt_decay)
                if dice_fast_tv > best_tv.dice_fast:
                    best_tv  = Result(dice_true=None, dice_fast=dice_fast_tv,  seg=seg_tv,  nt_weighting=nt_weighting, nt_decay=nt_decay)

                del ss
                
            s.update("Re-evaluate Dice coefficients for best combinations (precise)")
            dice_true_atv = dice_for(best_atv.seg, ph_true)
            dice_true_tv  = dice_for(best_tv.seg,  ph_true)
            print "ATV:", dice_true_atv
            print "TV: ", dice_true_tv
            
            results_atv[noise_level] = Result(dice_true=dice_true_atv, dice_fast=best_atv.dice_fast, seg=best_atv.seg, nt_weighting=best_atv.nt_weighting, nt_decay=best_atv.nt_decay)
            results_tv[noise_level]  = Result(dice_true=dice_true_tv,  dice_fast=best_tv.dice_fast,  seg=best_tv.seg,  nt_weighting=best_tv.nt_weighting,  nt_decay=best_tv.nt_decay)
            
            del s
            
        return results_atv, results_tv

if __name__ == "__main__":
    
    e = PhantomExperiment()
    results_atv, results_tv = e.execute()
    
    print
    print
    print "Best ATV results:"
    for noise_level in sorted(results_atv.keys()):
        print noise_level, results_atv[noise_level]
    print "Best TV results:"
    for noise_level in sorted(results_tv.keys()):
        print noise_level, results_tv[noise_level]

    # Show the surface for the highest noise level: ATV above, TV below
    # ('cleaned up': only show the largest connected component)
    noise_level = sorted(e.noise_levels)[-1]
    
    seg_atv_raw = results_atv[noise_level].seg
    seg_tv_raw  = results_tv[noise_level].seg

    seg_atv_bin = binary_dilation(select_largest(seg_atv_raw))
    seg_tv_bin  = binary_dilation(select_largest(seg_tv_raw))

    seg_atv = seg_atv_raw * seg_atv_bin
    seg_tv  = seg_tv_raw  * seg_tv_bin

    stacked = np.dstack((seg_tv, seg_atv))  # Lowest goes first
    show_surface(stacked)
    