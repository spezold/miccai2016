#!/usr/bin/env python
# coding: utf-8

"""
A module for creating a helical image phantom.
"""

from __future__ import division

from mayavi import mlab
# import nibabel  FIXME: only needed if the phantom image should be saved to
# disk as a NIfTI file
import numpy as np
from tvtk.api import tvtk
from tvtk.common import configure_input, is_old_pipeline


def create_toric_helix(r_helix, r_tube, slope, n_windings,
                       factor_helix=1.0, factor_tube=1.0, factor_slope=1.0,
                       points_per_circle=100, points_per_winding=100,
                       write_output=False, output_path=None,
                       voxel_size=1.0, margins=0.0, show=False):
    """
    Create a 'toric helix', i.e. a tube that winds around itself in a helical
    manner.
    
    The parameters are as follows: <r_helix> defines the radius of the helix,
    i.e. the distance from the winding center to the tube center; <r_tube>
    defines the radius of the tube, <slope> defines the slope of the helix,
    <n_windings> defines the number of windings; <factor_helix> defines the
    linear growth or shrinkage of the helix radius along its course (e.g. if
    <r_helix> is 100 and <factor_helix> is 0.25, then the helix radius will be
    100 at the beginning of the first winding and 25 at the end of the last
    winding); similarly, <factor_tube> defines the linear growth or shrinkage
    of the tube radius and <factor_slope> defines the linear growth or
    shrinkage of the slope along the course of the helix; <points_per_circle>
    defines at how many points each circular cross-section of the tube shall be
    sampled; <points_per_winding> defines how many cross-sections shall be
    sampled for each winding.
    
    Sample the helix as an image volume with the given <voxel_size> (scalar or
    three-tuple expected), with samples inside the helix set to 100 and samples
    outside the helix set to 0. If <write_output> is True, write the result as
    a NIfTI image [*] at the given <output_path>. If <margins> are given
    (scalar, three-tuple [axis 0, axis 1, axis 2], or six-tuple [axis 0 start,
    axis 1 start, axis 2 start, axis 0 end, axis 1 end, axis 2 end] expected),
    add the respective margins of background values at the volume boundaries,
    otherwise the volume will (more or less) exactly enclose the helix.
    
    If <show> is True, show a surface plot of the created helix. In any case,
    return the resulting sampled image volume as a 3D Numpy array.
    
    References
    [*] http://nifti.nimh.nih.gov/ (20150713)
    """
    # Sampling positions for the curve parameter <t> and the tube angle <theta>
    # (the tube boundaries are overlapping for now)
    t, theta = np.mgrid[0 : n_windings : (n_windings * points_per_winding * 1j),
                        0 : (2 * np.pi) : ((points_per_circle + 1) * 1j)]
    
    # Taking care of <factor_*>
    r_h = lambda t : r_helix * ((factor_helix - 1) * t / n_windings + 1)
    r_t = lambda t : r_tube  * ((factor_tube  - 1) * t / n_windings + 1)
    # The ascent is not that simple, as both the slope and the helix radius
    # depend on t (note to self: see notes from 13.07.2015)
    z = lambda t : 2 * np.pi * r_helix * slope * t * (1 + t * (factor_helix + factor_slope - 2) / (2 * n_windings) + t ** 2 * (factor_helix - 1) * (factor_slope - 1) / (3 * n_windings ** 2))
    
    # Actual helix coordinates
    c0 = (r_h(t) + r_t(t) * np.cos(theta)) * np.cos(2 * np.pi * t)
    c1 = (r_h(t) + r_t(t) * np.cos(theta)) * np.sin(2 * np.pi * t)
    c2 = r_t(t) * np.sin(theta) + z(t)

    # Closing the helix is necessary for converting its surface representation
    # into an image volume; take care of the overlapping tube boundaries there
    closed_poly_data = _close_helix(c0, c1, c2)
    
    if show:
#         # Render the (open) surface        
#         mlab.mesh(c0, c1, c2)

        # Also render the closed surface
        mapper = tvtk.PolyDataMapper(input=closed_poly_data,
                                     scalar_visibility=False)
        actor  = tvtk.Actor(mapper=mapper)
#         actor.property.representation = "wireframe"
        actor.property.color = (0., 1., 0.)
        mlab.gcf().scene.add_actor(actor)
        mlab.figure(mlab.gcf(), bgcolor=(1., 1., 1.))
            
        mlab.show()
    
    # 'Expand' voxel size and margins values if necessary
    voxel_size = np.multiply(voxel_size, [1, 1, 1], dtype=np.float)
    try:
        if len(margins) == 3:
            margins = np.r_[margins, margins].astype(np.float)
        else:
            margins = np.asarray(margins, dtype=np.float)
    except TypeError:  # Scalar case
        margins = np.multiply(margins, np.ones(6), dtype=np.float)

    # Actually create the volume  
    vol = _poly_data_to_volume(closed_poly_data, voxel_size, margins)
    
    if write_output:
        # FIXME:
#         affine = np.diag(np.r_[voxel_size, 1].astype(np.float))
#         nibabel.Nifti1Image(vol, affine).to_filename(output_path)
        pass
        
    return vol
        

def _close_helix(c0, c1, c2):
    """
    Close the start and end of the given open helix coordinates, return the
    result as a <tvtk.PolyData> instance.
    """
    # We do not rely on the triangulation of mlab.mesh(), as (1) it seems to
    # fail for small numbers of points, (2) we have to close the two ends of
    # the tube manually anyway    
    
    assert c0.shape == c1.shape == c2.shape
    n_circles = c0.shape[0]
    points_per_circle = c0.shape[1] - 1
    cells_per_circle = points_per_circle * 2
    open_num = n_circles * points_per_circle

    # Make the coordinates one array, remove overlapping seam points, flatten
    open_points = np.empty((n_circles, points_per_circle, 3), dtype=np.float)
    open_points[..., 0] = c0[:, :-1]
    open_points[..., 1] = c1[:, :-1]
    open_points[..., 2] = c2[:, :-1]
    open_points = open_points.reshape(-1, 3)
    
    # Create triangles between circles of the tube
    open_cells = np.empty((cells_per_circle * (n_circles - 1), 3), dtype=np.int)
    for i_circle in range(n_circles - 1):
        
        current_circle_ids = np.arange((i_circle + 0) * points_per_circle,
                                       (i_circle + 1) * points_per_circle)
        next_circle_ids    = np.arange((i_circle + 1) * points_per_circle,
                                       (i_circle + 2) * points_per_circle)
    
        # Connect the neighboring circles' points, i.e. create the cells
        # (a, b, c) and (b, d, c) according to the following scheme:
        #
        #     current a-c ...
        #             |/|
        #     next    b-d ...
        #
        current_cells = np.empty((cells_per_circle, 3), dtype=np.int)
        for i_vertex in range(points_per_circle):
            
            a = current_circle_ids[i_vertex]
            c = current_circle_ids[(i_vertex + 1) % points_per_circle]
            b =    next_circle_ids[i_vertex]
            d =    next_circle_ids[(i_vertex + 1) % points_per_circle]
            
            current_cells[2 * i_vertex]     = a, b, c
            current_cells[2 * i_vertex + 1] = b, d, c
            
        open_cells[cells_per_circle *  i_circle : 
                   cells_per_circle * (i_circle + 1)] = current_cells
    
    # Calculate the centers for the first and last circle (we can simply use
    # the mean of the points, as they are evenly spaced)
    first_circle = open_points[:points_per_circle]
    first_center = np.mean(first_circle, axis=0)
    
    last_circle = open_points[-points_per_circle:]
    last_center = np.mean(last_circle, axis=0)
    
    # Get the ids for the circle points
    first_ids = np.arange(0, points_per_circle)
    last_ids = np.arange(open_num - points_per_circle, open_num)

    # Create a new point array that additionally holds the center points
    closed_points = np.empty((open_num + 2, 3), dtype=np.float)
    closed_points[:-2] = open_points
    closed_points[-2] = first_center
    closed_points[-1] = last_center
    first_center_id = open_num
    last_center_id = open_num + 1
    
    # Create the cells that close the first and last circle
    first_cells = np.empty((points_per_circle, 3), dtype=np.int)
    for i in range(points_per_circle):
        first_cells[i] = first_center_id, first_ids[i], first_ids[(i + 1) % points_per_circle]
        
    last_cells = np.empty((points_per_circle, 3), dtype=np.int)
    for i in range(points_per_circle):
        last_cells[i] = last_center_id, last_ids[i], last_ids[(i + 1) % points_per_circle]
        
    # Stack the cells and convert them to tvtk.CellArray, then create and
    # return new tvtk.PolyData instance
    closed_cells = tvtk.CellArray()
    closed_cells.from_array(np.vstack((open_cells, first_cells, last_cells)))
    closed_poly_data = tvtk.PolyData(points=closed_points, polys=closed_cells)
    return closed_poly_data
    

def _poly_data_to_volume(data, voxel_size, margins):
    """
    Render the given, closed <tvtk.PolyData> into an image volume and return it
    as a Numpy array.
    """
    # Following [*] for the necessary steps.
    #
    # References
    # [*] http://www.paraview.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataToImageData (20150710)

    # Calculate the necessary number of voxels, extent, and origin  
    bounds = np.array(data.bounds)
    n_vox = np.ceil((bounds[1::2] - bounds[0::2] +
                     np.sum(margins.reshape(2, 3), axis=0)) /
                    voxel_size).astype(np.int)

    extent = [0, n_vox[0] - 1, 0, n_vox[1] - 1, 0, n_vox[2] - 1]
    origin = bounds[0::2] - margins[:3] + 0.5 * voxel_size
    
    # Create the image volume
    volume = tvtk.ImageData(spacing=voxel_size,
                            dimensions=n_vox,
                            extent=extent,
                            origin=origin)
    if is_old_pipeline():
        volume.scalar_type = "unsigned_char"
    voxel_data = (np.ones(n_vox, dtype=np.uint8) * 100).ravel()
    volume.point_data.scalars = voxel_data
    if is_old_pipeline():
        volume.update()
    
    # Actual transformation from polygon to image data
    voxelizer = tvtk.PolyDataToImageStencil(output_origin=origin,
                                            output_spacing=voxel_size,
                                            output_whole_extent=volume.extent,
                                            tolerance=0.0)
    configure_input(voxelizer, data)
    voxelizer.update()
    
    # Draw the result to a new image, extract Numpy array and return
    painter = tvtk.ImageStencil(reverse_stencil=False,
                                background_value=0)
    configure_input(painter, volume)
    if is_old_pipeline():
        painter.stencil = voxelizer.output
    else:
        painter.set_stencil_connection(voxelizer.output_port)
    painter.update()
    voxel_data = painter.output.point_data.scalars.to_array().reshape(n_vox[::-1])
    voxel_data = np.transpose(voxel_data, [2, 1, 0])
    return voxel_data
