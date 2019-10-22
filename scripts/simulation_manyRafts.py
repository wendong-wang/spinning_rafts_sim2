# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:08:51 2019

@author: wwang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scipy.io
from scipy.spatial import distance as scipyDistance
from scipy.spatial import Voronoi as scipyVoronoi
from scipy.integrate import solve_ivp
import cv2 as cv
#import time

import os, glob
import shelve
import progressbar

def DrawRafts(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    ''' draw circles around rafts
    '''
    
    circle_thickness = int(2)
    circle_color = (0,0,255) # openCV: BGR

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id,0], rafts_loc[raft_id,1]), rafts_radii[raft_id], circle_color, circle_thickness)
    
    return output_img

def DrawRaftsRhinoCoord(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    ''' draw circles in Rhino coordinate, right-handed xy coordinate
    '''
    circle_thickness = int(2)
    circle_color = (0,0,255) # openCV: BGR

    output_img = img_bgr
    height, width, _ = img_bgr.shape
    xAxis_start = (0, height - 10)
    xAxis_end = (width, height - 10)
    yAxis_start = (10, 0)
    yAxis_end = (10, height)
    output_img = cv.line(output_img, xAxis_start, xAxis_end, (0,0,0), 4)
    output_img = cv.line(output_img, yAxis_start, yAxis_end, (0,0,0), 4)
    
    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id,0], height - rafts_loc[raft_id,1]), rafts_radii[raft_id], circle_color, circle_thickness)
    
    return output_img
    
 
def DrawRaftOrientations(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    ''' draw lines to indicte the dipole orientation of each raft, as indicated by rafts_ori
        draw lines to indicate the capillary edge positions
    '''

    line_thickness = int(2)
    line_color = (255,0,0)
        
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id,0], rafts_loc[raft_id,1])
        line_end = (int(rafts_loc[raft_id,0] + np.cos(rafts_ori[raft_id]*np.pi/180)*rafts_radii[raft_id]), 
                    int(rafts_loc[raft_id,1] - np.sin(rafts_ori[raft_id]*np.pi/180)*rafts_radii[raft_id])) #note that the sign in front of the sine term is "-"
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img

def DrawRaftOrientationsRhinoCoord(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    ''' draw lines to indicte the dipole orientation of each raft, as indicated by rafts_ori
        draw lines to indicate the capillary edge positions
    '''

    line_thickness = int(2)
    line_color = (255,0,0)
        
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id,0], height - rafts_loc[raft_id,1])
        line_end = (int(rafts_loc[raft_id,0] + np.cos(rafts_ori[raft_id]*np.pi/180)*rafts_radii[raft_id]), 
                    height - int(rafts_loc[raft_id,1] + np.sin(rafts_ori[raft_id]*np.pi/180)*rafts_radii[raft_id])) #note that the sign in front of the sine term is "+"
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img

def DrawSym4Positions(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    ''' draw lines to indicate the capillary edge positions
    '''

    line_thickness = int(2)
    line_color2 = (0,255,0)
    raft_sym = 4
    cap_gap = 360/raft_sym
    cap_offset = 45 # the angle between the dipole direction and the first capillary peak

        
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            line_start = (rafts_loc[raft_id,0], rafts_loc[raft_id,1])
            line_end = (int(rafts_loc[raft_id,0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap)*np.pi/180)*rafts_radii[raft_id]), 
                        int(rafts_loc[raft_id,1] - np.sin((rafts_ori[raft_id] + cap_offset + capID * cap_gap)*np.pi/180)*rafts_radii[raft_id])) #note that the sign in front of the sine term is "-"
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img

def DrawSym4PositionsRhinoCoord(img_bgr, rafts_loc, rafts_ori, cap_offset, rafts_radii, num_of_rafts):
    ''' draw lines to indicate the capillary edge positions
    '''

    line_thickness = int(2)
    line_color2 = (0,255,0)
    raft_sym = 4
    cap_gap = 360/raft_sym
#    cap_offset = 45 # the angle between the dipole direction and the first capillary peak

        
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            line_start = (rafts_loc[raft_id,0], height - rafts_loc[raft_id,1])
            line_end = (int(rafts_loc[raft_id,0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap)*np.pi/180)*rafts_radii[raft_id]), 
                        height - int(rafts_loc[raft_id,1] + np.sin((rafts_ori[raft_id] + cap_offset + capID * cap_gap)*np.pi/180)*rafts_radii[raft_id])) #note that the sign in front of the sine term is "+"
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img

def DrawSym6PositionsRhinoCoord(img_bgr, rafts_loc, rafts_ori, cap_offset, rafts_radii, num_of_rafts):
    ''' draw lines to indicate the capillary edge positions
    '''

    line_thickness = int(2)
    line_color2 = (0,255,0)
    raft_sym = 6
    cap_gap = 360/raft_sym
#    cap_offset = 45 # the angle between the dipole direction and the first capillary peak

        
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            line_start = (rafts_loc[raft_id,0], height - rafts_loc[raft_id,1])
            line_end = (int(rafts_loc[raft_id,0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap)*np.pi/180)*rafts_radii[raft_id]), 
                        height - int(rafts_loc[raft_id,1] + np.sin((rafts_ori[raft_id] + cap_offset + capID * cap_gap)*np.pi/180)*rafts_radii[raft_id])) #note that the sign in front of the sine term is "+"
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img

def DrawRaftNumber(img_bgr, rafts_loc, num_of_rafts):
    ''' draw the raft number at the center of the rafts
    '''
    
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0,0,0) # BGR
    font_thickness = 2
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id+1),fontFace, font_scale, font_thickness)
        output_img = cv.putText(output_img,str(raft_id+1),(rafts_loc[raft_id,0] - textSize[0]//2, rafts_loc[raft_id,1] + textSize[1]//2), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
        
    return output_img

def DrawRaftNumberRhinoCoord(img_bgr, rafts_loc, num_of_rafts):
    ''' draw the raft number at the center of the rafts
    '''
    
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0,0,0) # BGR
    font_thickness = 2
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    
    for raft_id in np.arange(num_of_rafts):
        textSize, _ = cv.getTextSize(str(raft_id+1),fontFace, font_scale, font_thickness)
        output_img = cv.putText(output_img,str(raft_id+1),(rafts_loc[raft_id,0] - textSize[0]//2, height - (rafts_loc[raft_id,1] + textSize[1]//2)), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
        
    return output_img

def DrawFrameInfoManyRafts(img_bgr, time_step_num, hexOrder_avg_norm, hexOrder_norm_avg, entropy_by_distances):
    ''' draw information on the output frames
    '''
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0,0,0) # BGR
    font_thickness = 1
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    textSize, _ = cv.getTextSize(str(time_step_num),fontFace, font_scale, font_thickness)
    line_padding = 2
    left_padding = 20
    top_padding = 20
    output_img = cv.putText(output_img, 'time step: {}'.format(time_step_num), (left_padding, top_padding), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
    output_img = cv.putText(output_img, 'hexOrder_avg_norm: {:03.2f}'.format(hexOrder_avg_norm), (left_padding, top_padding + (textSize[1] + line_padding) * 1 ), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
    output_img = cv.putText(output_img, 'hexOrder_norm_avg: {:03.2f}'.format(hexOrder_norm_avg), (left_padding, top_padding + (textSize[1] + line_padding) * 2 ), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
    output_img = cv.putText(output_img, 'entropy_by_distances: {:03.2f}'.format(entropy_by_distances), (left_padding, top_padding + (textSize[1] + line_padding) * 3 ), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
#    output_img = cv.putText(output_img, str('velocity_torque_coupling: {}'.format(velocity_torque_coupling)), (10, 10 + (textSize[1] + line_padding) * 4 ), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
#    output_img = cv.putText(output_img, str('magnetic_dipole_force: {}'.format(magnetic_dipole_force)), (10, 10 + (textSize[1] + line_padding) * 5), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
#    output_img = cv.putText(output_img, str('capillary_force: {}'.format(capillary_force)), (10, 10 + (textSize[1] + line_padding) * 6), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
#    output_img = cv.putText(output_img, str('hydrodynamic_force: {}'.format(hydrodynamic_force)), (10, 10 + (textSize[1] + line_padding) * 7), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
#    output_img = cv.putText(output_img, str('B-field_torque: {}'.format(B-field_torque)), (10, 10 + (textSize[1] + line_padding) * 8), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
#    output_img = cv.putText(output_img, str('mag_dipole_torque: {}'.format(mag_dipole_torque)), (10, 10 + (textSize[1] + line_padding) * 9), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
#    output_img = cv.putText(output_img, str('cap_torque: {}'.format(cap_torque)), (10, 10 + (textSize[1] + line_padding) * 10), fontFace, font_scale,font_color,font_thickness,cv.LINE_AA)
    
    return output_img


def DrawVoronoiRhinoCoord(img_bgr, rafts_loc):
    ''' draw Voronoi patterns
    '''
    height, width, _ = img_bgr.shape
    points = rafts_loc
    points[:,1] =  height - points[:,1]
    vor = scipyVoronoi(points)
    output_img = img_bgr
    # drawing Voronoi vertices
    vertex_size = int(3)
    vertex_color = (255,0,0)
    for x, y in zip(vor.vertices[:,0], vor.vertices[:,1]):
        output_img = cv.circle(output_img, (int(x), int(y)), vertex_size, vertex_color)
    
    # drawing Voronoi edges
    edge_color = (0, 255, 0)
    edge_thickness = int(2)
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            output_img = cv.line(output_img, (int(vor.vertices[simplex[0], 0]), int(vor.vertices[simplex[0], 1])), (int(vor.vertices[simplex[1], 0]), int(vor.vertices[simplex[1], 1])), edge_color, edge_thickness)

    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 200
            output_img = cv.line(output_img, (int(vor.vertices[i,0]), int(vor.vertices[i,1])), (int(far_point[0]), int(far_point[1])), edge_color, edge_thickness)
    return output_img



def MapAnglesTo0To360(angleInDeg):
    ''' map the angleInDeg to the interval [0, 360)
    '''
    while angleInDeg >= 360:
        angleInDeg = angleInDeg - 360
    while angleInDeg < 0:
        angleInDeg = angleInDeg + 360
    
    return angleInDeg

def FFTDistances(sampling_rate, distances):
    ''' given sampling rate and distances, and output frequency vector and one-sided power spectrum
        sampling_rate: unit Hz
        distances: numpy array, unit micron
    '''
#    sampling_interval = 1/sampling_rate # unit s
#    times = np.linspace(0,sampling_length*sampling_interval, sampling_length)
    sampling_length = len(distances) # total number of frames
    fft_distances = np.fft.fft(distances)
    P2 = np.abs(fft_distances/sampling_length)
    P1 = P2[0:int(sampling_length/2)+1]
    P1[1:-1] = 2*P1[1:-1] # one-sided powr spectrum
    frequencies = sampling_rate/sampling_length * np.arange(0,int(sampling_length/2)+1)
    
    return frequencies, P1

def AdjustPhases(phases_input):
    ''' adjust the phases to get rid of the jump of 360 when it crosses from -180 to 180, or the reverse
        adjust single point anormaly. 
    '''
    phase_diff_threshold = 200
    
    phases_diff = np.diff(phases_input)
    
    index_neg = phases_diff < -phase_diff_threshold
    index_pos = phases_diff > phase_diff_threshold
    
    insertion_indices_neg = np.nonzero(index_neg)
    insertion_indices_pos = np.nonzero(index_pos)
    
    
    phase_diff_corrected = phases_diff.copy()
    phase_diff_corrected[insertion_indices_neg[0]] += 360
    phase_diff_corrected[insertion_indices_pos[0]] -= 360
    
    phases_corrected = phases_input.copy()
    phases_corrected[1:] = phase_diff_corrected[:]
    phases_adjusted = np.cumsum(phases_corrected)
    
    return phases_adjusted

def ShannonEntropy(c):
    """calculate the Shannon entropy of 1 d data. The unit is bits """
    
    c_normalized = c / float(np.sum(c))
    c_normalized_nonzero = c_normalized[np.nonzero(c_normalized)] # gives 1D array
    H = -sum(c_normalized_nonzero* np.log2(c_normalized_nonzero))  # unit in bits
    return H

def SquareSpiral(num_of_rafts, spacing, origin):
    '''
    initialize the raft positions using square spirals
    ref: 
    https://stackoverflow.com/questions/398299/looping-in-a-spiral
    '''
    raft_locations = np.zeros((numOfRafts, 2))
#    X =Y = int(np.sqrt(num_of_rafts))
    x = y = 0
    dx = 0
    dy = -1
    for i in range(num_of_rafts):
#        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
#            print (x, y)
            # DO STUFF...
        raft_locations[i,:] = np.array([x,y]) * spacing + origin
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    return raft_locations

#%% load capillary force and torque
rootFolderNameFromWindows = r'C:\\SimulationFolder\spinningRaftsSimulationCode'

os.chdir(rootFolderNameFromWindows)

os.chdir('2019-05-13_capillaryForceCalculations-sym6') # this is for sym4 rafts

shelveName = 'capillaryForceAndTorque_sym6'
shelveDataFileName = shelveName + '.dat'
listOfVariablesToLoad = ['eeDistanceCombined', 'forceCombinedDistancesAsRowsAll360', 'torqueCombinedDistancesAsRowsAll360']

if os.path.isfile(shelveDataFileName):
    tempShelf = shelve.open(shelveName)
    capillaryEEDistances = tempShelf['eeDistanceCombined'] #unit: m
    capillaryForcesDistancesAsRowsLoaded = tempShelf['forceCombinedDistancesAsRowsAll360'] # unit: N
    capillaryTorquesDistancesAsRowsLoaded = tempShelf['torqueCombinedDistancesAsRowsAll360'] # unit: N.m

os.chdir('..')

#further data treatment on capillary force profile
#insert the force and torque at eeDistance = 1um as the value for eedistance = 0um. 
capillaryEEDistances = np.insert(capillaryEEDistances, 0, 0)
capillaryForcesDistancesAsRows = np.concatenate((capillaryForcesDistancesAsRowsLoaded[:1,:], capillaryForcesDistancesAsRowsLoaded), axis = 0)
capillaryTorquesDistancesAsRows = np.concatenate((capillaryTorquesDistancesAsRowsLoaded[:1,:], capillaryTorquesDistancesAsRowsLoaded), axis = 0)

#add angle=360, the same as angle = 0
capillaryForcesDistancesAsRows = np.concatenate((capillaryForcesDistancesAsRows, capillaryForcesDistancesAsRows[:,0].reshape(1001,1)), axis = 1)
capillaryTorquesDistancesAsRows = np.concatenate((capillaryTorquesDistancesAsRows, capillaryTorquesDistancesAsRows[:,0].reshape(1001,1)), axis = 1)

# correct for the negative sign of the torque
capillaryTorquesDistancesAsRows = - capillaryTorquesDistancesAsRows

# some extra treatment for the force matrix
# note the sharp attraction at the peak-peak position (45 deg): only 1 deg difference, the force changes from attraction to repulsion. 
# consider replacing values at eeDistance = 0, 1, 2, ... with values at eeDistance = 5um. 
nearEdgeSmoothingThresholdDist = 1 # unit: micron; if 1, then it is equivalent to no smoothing. 
for distanceToEdge in np.arange(nearEdgeSmoothingThresholdDist):
    capillaryForcesDistancesAsRows[distanceToEdge,:] = capillaryForcesDistancesAsRows[nearEdgeSmoothingThresholdDist,:]
    capillaryTorquesDistancesAsRows[distanceToEdge,:] = capillaryTorquesDistancesAsRows[nearEdgeSmoothingThresholdDist,:]

# select a cut-off distance below which all the attractive force (negative-valued) becomes zero, due to raft wall-wall repulsion
attractionZeroCutoff = 1# unit: micron
mask = np.concatenate((capillaryForcesDistancesAsRows[:attractionZeroCutoff,:] < 0, np.zeros((capillaryForcesDistancesAsRows.shape[0]-attractionZeroCutoff, capillaryForcesDistancesAsRows.shape[1]), dtype = int)), axis = 0)
capillaryForcesDistancesAsRows[mask.nonzero()] = 0

# realign the first peak-peak direction with an angle = capillaryPeakOffset from the x-axis. 
capillaryPeakOffset = 0
capillaryForcesDistancesAsRows = np.roll(capillaryForcesDistancesAsRows, capillaryPeakOffset, axis = 1) # 45 is due to original data
capillaryTorquesDistancesAsRows = np.roll(capillaryTorquesDistancesAsRows, capillaryPeakOffset, axis = 1)


#%% magnetic force and torque calculation: 
miu0 = 4 * np.pi * 1e-7 # unit: N/A**2, or T.m/A, or H/m

# from the data 2018-09-28, 1st increase: (1.4e-8 A.m**2 for 14mT), (1.2e-8 A.m**2 for 10mT), (0.96e-8 A.m**2 for 5mT), (0.78e-8 A.m**2 for 1mT)
# from the data 2018-09-28, 2nd increase: (1.7e-8 A.m**2 for 14mT), (1.5e-8 A.m**2 for 10mT), (1.2e-8 A.m**2 for 5mT), (0.97e-8 A.m**2 for 1mT)
magneticMomentOfOneRaft = 1e-8 # unit: A.m**2

orientationAngles = np.arange(0,361) # unit: degree; think in terms of Rhino's xy coordinates. 0 deg corresponds to attraction. 90 deg corresponds to repulsion. 
orientationAnglesInRad = np.radians(orientationAngles)

magneticDipoleEEDistances = np.arange(0,10001) / 1e6 # unit: m

radiusOfRaft = 1.5e-4 # unit: m

magneticDipoleCCDistances = magneticDipoleEEDistances + radiusOfRaft * 2 # unit: m

magDpForceOnAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles))) # unit: N
magDpForceOffAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles))) # unit: N
magDpTorque = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles))) # unit: N.m


for index, d in enumerate(magneticDipoleCCDistances):
#    magneticEnergyDistancesAsRows[index,:] = miu0 * magneticMomentOfOneRaft**2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d**3)
    magDpForceOnAxis[index,:] = 3* miu0 * magneticMomentOfOneRaft**2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d**4)
    magDpForceOffAxis[index,:] = 3* miu0 * magneticMomentOfOneRaft**2 * (2*np.cos(orientationAnglesInRad) * np.sin(orientationAnglesInRad)) / (4 * np.pi * d**4)
    magDpTorque[index,:] = miu0 * magneticMomentOfOneRaft**2 * (3 * np.cos(orientationAnglesInRad) * np.sin(orientationAnglesInRad)) / (4 * np.pi * d**3)

#magnetic force at 1um(attrationZeroCutoff) should have no attraction, due to wall-wall repulsion. Treat it similarly as capillary cutoff
attractionZeroCutoff = 0 # unit: micron
mask = np.concatenate((magDpForceOnAxis[:attractionZeroCutoff,:] < 0, np.zeros((magDpForceOnAxis.shape[0]-attractionZeroCutoff, magDpForceOnAxis.shape[1]), dtype = int)), axis = 0)
magDpForceOnAxis[mask.nonzero()] = 0

magneticMaxRepulsion = magDpForceOnAxis.max(axis = 1)

#%% lubrication equation coefficients:
RforCoeff = 150.0 # unit: micron
stepSizeForDist = 0.1
lubCoeffScaleFactor = 1/stepSizeForDist
eeDistancesForCoeff = np.arange(0,15+stepSizeForDist,stepSizeForDist, dtype = 'double') # unit: micron

eeDistancesForCoeff[0] = 1e-10 # unit: micron

x = eeDistancesForCoeff / RforCoeff # unit: 1

lubA = x * (-0.285524 * x + 0.095493 * x * np.log(x) + 0.106103) / RforCoeff # unit: 1/um

lubB = ((0.0212764 * (- np.log(x)) + 0.157378) * (- np.log(x)) + 0.269886) / (RforCoeff * (- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549) # unit: 1/um

#lubC = ( (-0.0212758 * (- np.log(x)) - 0.089656) * (- np.log(x)) + 0.0480911) / (RforCoeff**2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549) ) # unit: 1/um^2

#lubD = (0.0579125 * (- np.log(x)) + 0.0780201) / (RforCoeff**2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549) ) # unit: 1/um^2

lubG = ((0.0212758 * (- np.log(x)) + 0.181089) * (- np.log(x)) + 0.381213 ) /(RforCoeff**3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549)) # unit: 1/um^3

lubC = - RforCoeff * lubG

#lubH = (0.265258 * (- np.log(x)) + 0.357355) / (RforCoeff**3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549) ) # unit: 1/um^3

#lubCoeffCombined = np.column_stack((lubA,lubB,lubC,lubD,lubG,lubH))
#%% simulation of many-raft system

# check the dipole orientation and capillary orientation
#eeDistanceForPlotting = 40
#fig, ax = plt.subplots(ncols=2, nrows=1)
#ax[0].plot(capillaryForcesDistancesAsRows[eeDistanceForPlotting,:], label = 'capillary force') # 0 deg is the peak-peak alignment - attraction. 
#ax[0].plot(magDpForceOnAxis[eeDistanceForPlotting,:], label = 'magnetic force') # 0 deg is the dipole-dipole attraction
#ax[0].set_xlabel('angle')
#ax[0].set_ylabel('force (N)')
#ax[0].legend()
#ax[0].set_title('force at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting, capillaryPeakOffset))
#ax[1].plot(capillaryTorquesDistancesAsRows[eeDistanceForPlotting,:], label = 'capillary torque') # 0 deg is the peak-peak alignment - attraction. 
#ax[1].plot(magDpTorque[eeDistanceForPlotting,:], label = 'magnetic torque') # 0 deg is the dipole-dipole attraction
#ax[1].set_xlabel('angle')
#ax[1].set_ylabel('torque (N.m)')
#ax[1].legend()
#ax[1].set_title('torque at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting, capillaryPeakOffset))
#
## plot the various forces and look for the transition rps
#densityOfWater = 1e-15 # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
#raftRadius = 1.5e2 # unit: micron 
#magneticFieldRotationRPS = 20
#omegaBField = magneticFieldRotationRPS * 2 * np.pi
#hydrodynamicRepulsion = densityOfWater * omegaBField**2 * raftRadius**7 * 1e-6/ np.arange(raftRadius * 2 + 1, raftRadius * 2 + 1002)**3  #unit: N
#sumOfAllForces = capillaryForcesDistancesAsRows.mean(axis = 1) + magDpForceOnAxis.mean(axis = 1)[:1001] + hydrodynamicRepulsion
#fig, ax = plt.subplots(ncols = 1, nrows = 1)
#ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1), label = 'angle-averaged capillary force')
#ax.plot(magDpForceOnAxis.mean(axis = 1)[:1000], label = 'angle-averaged magnetic force')
#ax.plot(hydrodynamicRepulsion, label = 'hydrodynamic repulsion')
#ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1) + magDpForceOnAxis.mean(axis = 1)[:1001], label = 'angle-avaraged sum of magnetic and capillary force')
#ax.plot(sumOfAllForces[20:], label = 'sum of angle-averaged magnetic and capillary forces and hydrodynamic force ')
#ax.legend()


os.chdir(r'C:\SimulationFolder') 
listOfVariablesToSave = ['arenaSize','numOfRafts', 'magneticFieldStrength','magneticFieldRotationRPS','omegaBField','timeStepSize','numOfTimeSteps',
                         'timeTotal', 'outputImageSeq','outputVideo','outputFrameRate','intervalBetweenFrames',
                         'raftLocations', 'raftOrientations', 'raftRadii', 
                         'entropyByNeighborDistances', 'hexaticOrderParameterAvgs', 'hexaticOrderParameterAvgNorms', 
                         'hexaticOrderParameterMeanSquaredDeviations', 'hexaticOrderParameterModuliiAvgs', 'hexaticOrderParameterModuliiStds',
                         'deltaR', 'radialRangeArray', 'binEdgesNeighborDistances',
                         'radialDistributionFunction', 'spatialCorrHexaOrderPara','spatialCorrHexaBondOrientationOrder',
#                         'velocityTorqueCouplingTerm', 'magDipoleForceOnAxisTerm', 'capillaryForceTerm', 'hydrodynamicForceTerm', 
#                         'stochasticTerm', 'curvatureForceTerm', 'wallRepulsionTerm', 'boundaryForceTerm',
#                         'magneticFieldTorqueTerm', 'magneticDipoleTorqueTerm', 'capillaryTorqueTerm',
                         'currentStepNum', 'currentFrameBGR', 'dfNeighbors', 'dfNeighborsAllFrames']

# constant of proportionalities
cm = 1 # coefficient for the magnetic force term
cc = 1 # coefficient for the capillary force term
ch = 1 # coefficient for the hydrodynamic force term
tb = 1 # coefficient for the magnetic field torque term
tm = 1 # coefficient for the magnetic dipole-dipole torque term
tc = 1 # coefficient for the capillary torque term
forceDueToCurvature = 5e-10 #5e-9 #1e-10 # unit: N
#forceDueToBoundary = 0 # unit: N
wallRepulsionForce = 1e-7 # unit: N
#elasticWallThickness = 2 # unit: micron

unitVectorX = np.array([1,0])
unitVectorY = np.array([0,1])

arenaSize = 1.5e4 # unit: micron
R = raftRadius = 1.5e2 # unit: micron
centerOfArena = np.array([arenaSize/2, arenaSize/2])
cutoffDistance = 100000 # unit: micron. Above which assume that the rafts do not interact. 
#radiusOfCurvatureFreeCenter = 10 * raftRadius # unit: micron

#all calculations are done in SI numbers, and only in drawing are the variables converted to pixel unit
canvasSizeInPixel = int(1000) # unit: pixel 
scaleBar = arenaSize/canvasSizeInPixel # unit: micron/pixel

densityOfWater = 1e-15 # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
miu = 1e-15 #dynamic viscosity of water; unit conversion: 1e-3 Pa.s = 1e-3 N.s/m^2 = 1e-15 N.s/um^2 
piMiuR = np.pi*miu*raftRadius # unit: N.s/um

numOfRafts = 100
magneticFieldStrength = 10e-3 #10e-3 # unit: T
initialPositionMethod = 2 # 1 -random positions, 2 - fixed initial position, 3 - starting positions are the last positions of the previous spin speeds
lastPositionOfPreviousSpinSpeeds = np.zeros((numOfRafts,2))
lastOmegaOfPreviousSpinSpeeds = np.zeros((numOfRafts))
firstSpinSpeedFlag = 1

timeStepSize = 1e-3 # unit: s
numOfTimeSteps = 10000
timeTotal = timeStepSize * numOfTimeSteps

lubEqThreshold = 15 # unit micron, if the eeDistance is below this value, the torque velocity coupling term changes to rigid body rotation
stdOfFluctuationTerm = 0.00

deltaR = 1
radialRangeArray = np.arange(2, 100, deltaR)
binEdgesNeighborDistances = np.arange(2,10,0.5).tolist() + [100]

outputImageSeq = 0
outputVideo = 1
outputFrameRate = 10.0
intervalBetweenFrames = int(10) # unit: steps
blankFrameBGR = np.ones((canvasSizeInPixel, canvasSizeInPixel, 3), dtype = 'int')*255

solverMethod = 'RK45' # RK45,RK23, Radau, BDF, LSODA

def Fun_drdt_dalphadt(t, raft_loc_orient):
    '''
    Two sets of ordinary differential equations that define dr/dt and dalpha/dt above and below the threshold value 
    for the application of lubrication equations
    '''
#    raft_loc_orient = raftLocationsOrientations
    raft_loc = raft_loc_orient[0 : numOfRafts*2].reshape(numOfRafts, 2) # in um
    raft_orient = raft_loc_orient[numOfRafts*2 : numOfRafts*3] # in deg
    
    drdt = np.zeros((numOfRafts, 2)) #unit: um
    raft_spin_speeds_inRads = np.zeros(numOfRafts) # in rad
    dalphadt = np.zeros(numOfRafts) # unit: deg
    
    mag_Dipole_Force_OnAxis_Term = np.zeros((numOfRafts, 2))
    capillary_Force_Term = np.zeros((numOfRafts, 2))
    hydrodynamic_Force_Term = np.zeros((numOfRafts, 2))
    mag_Dipole_Force_OffAxis_Term = np.zeros((numOfRafts, 2))
    velocity_Torque_Coupling_Term = np.zeros((numOfRafts, 2))
    velocity_Mag_Fd_Torque_Term = np.zeros((numOfRafts, 2))
    wall_Repulsion_Term = np.zeros((numOfRafts, 2))
    stochastic_Term = np.zeros((numOfRafts, 2))
    curvature_Force_Term = np.zeros((numOfRafts, 2))
    boundary_Force_Term = np.zeros((numOfRafts, 2))
    
    magnetic_Field_Torque_Term = np.zeros(numOfRafts)
    magnetic_Dipole_Torque_Term = np.zeros(numOfRafts)
    capillary_Torque_Term = np.zeros(numOfRafts)
    
    # loop for torques and calculate raft_spin_speeds_inRads
    for raftID in np.arange(numOfRafts): 
        # raftID = 0       
        ri = raft_loc[raftID,:] # unit: micron
        
        # magnetic field torque:
        magnetic_Field_Torque = magneticFieldStrength * magneticMomentOfOneRaft * np.sin(np.deg2rad(magneticFieldDirection - raft_orient[raftID])) # unit: N.um
        magnetic_Field_Torque_Term[raftID] = tb * magnetic_Field_Torque * 1e6 /(8*piMiuR*R**2) # unit: 1/s
        
        rji_eeDist_smallest = R; # initialize
        
        for neighborID in np.arange(numOfRafts):
            if neighborID == raftID: 
                continue
            rj = raft_loc[neighborID,:] # unit: micron
            rji = ri - rj # unit: micron
            rji_Norm = np.sqrt(rji[0]**2 + rji[1]**2) # unit: micron
            rji_eeDist = rji_Norm - 2 * R # unit: micron
            rji_Unitized = rji/rji_Norm # unit: micron
            rji_Unitized_CrossZ = np.asarray((rji_Unitized[1], -rji_Unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raftID]) % 360 # unit: deg; assuming both rafts's orientations are the same 
#            if phi_ji == 360: phi_ji = 0 
            
#            print('{}, {}'.format(int(phi_ji), (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raftID])))
#             torque terms:
            if rji_eeDist < lubEqThreshold and rji_eeDist < rji_eeDist_smallest:
                rji_eeDist_smallest = rji_eeDist
                if rji_eeDist_smallest >= 0:
                    magnetic_Field_Torque_Term[raftID] = lubG[int(rji_eeDist_smallest * lubCoeffScaleFactor)] * magnetic_Field_Torque * 1e6 / miu # unit: 1/s
                elif rji_eeDist_smallest < 0:
                    magnetic_Field_Torque_Term[raftID] = lubG[0] * magnetic_Field_Torque * 1e6 / miu # unit: 1/s
                
            if rji_eeDist < 10000 and rji_eeDist >= 0: 
                magnetic_Dipole_Torque_Term[raftID] = magnetic_Dipole_Torque_Term[raftID] + tm * magDpTorque[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * 1e6/ (8*piMiuR*R**2)
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                magnetic_Dipole_Torque_Term[raftID] = magnetic_Dipole_Torque_Term[raftID] + tm *lubG[int(rji_eeDist * lubCoeffScaleFactor)] * magDpTorque[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * 1e6 / miu # unit: 1/s
            elif rji_eeDist < 0: 
                magnetic_Dipole_Torque_Term[raftID] = magnetic_Dipole_Torque_Term[raftID] + tm *lubG[0] * magDpTorque[0, int(phi_ji + 0.5)] * 1e6 / miu # unit: 1/s
                
            if rji_eeDist < 1000 and rji_eeDist >= lubEqThreshold: 
                capillary_Torque_Term[raftID] = capillary_Torque_Term[raftID] + tc * capillaryTorquesDistancesAsRows[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * 1e6 / (8*piMiuR*R**2) # unit: 1/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                capillary_Torque_Term[raftID] = capillary_Torque_Term[raftID] + tc * lubG[int(rji_eeDist * lubCoeffScaleFactor)] * capillaryTorquesDistancesAsRows[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * 1e6 / miu # unit: 1/s
            elif rji_eeDist < 0: 
                capillary_Torque_Term[raftID] = capillary_Torque_Term[raftID] + tc * lubG[0] * capillaryTorquesDistancesAsRows[0, int(phi_ji + 0.5)] * 1e6 / miu # unit: 1/s
            
            # debug use: 
#            raftRelativeOrientationInDeg[neighborID, raftID, currentStepNum] = phi_ji
        
        #debug use
#        capillaryTorqueTerm[raftID, currentStepNum] = capillary_Torque_Term[raftID]
            
        raft_spin_speeds_inRads[raftID] = magnetic_Field_Torque_Term[raftID] + magnetic_Dipole_Torque_Term[raftID] + capillary_Torque_Term[raftID]

    
    # loop for forces
    for raftID in np.arange(numOfRafts): 
        # raftID = 0       
        ri = raft_loc[raftID,:] # unit: micron
        omegai = raft_spin_speeds_inRads[raftID]
        
        # meniscus cuvature force term
        if forceDueToCurvature != 0: 
            ri_center = centerOfArena - ri
#            ri_center_Norm = np.sqrt(ri_center[0]**2 + ri_center[1]**2)
#            ri_center_Unitized = ri_center / ri_center_Norm
            curvature_Force_Term[raftID, :] = forceDueToCurvature / (6*piMiuR) * ri_center / (arenaSize/2)

        # boundary lift force term
        dToLeft = ri[0]
        dToRight = arenaSize - ri[0]
        dToBottom = ri[1]
        dToTop = arenaSize - ri[1]
        boundary_Force_Term[raftID, :] = 1e-6 * densityOfWater * omegai**2 * R**7 / (6*piMiuR) * ((1/dToLeft**3 - 1/dToRight**3) * unitVectorX + (1/dToBottom**3 - 1/dToTop**3) * unitVectorY)
        
        # magnetic field torque:
        magnetic_Field_Torque = magneticFieldStrength * magneticMomentOfOneRaft * np.sin(np.deg2rad(magneticFieldDirection - raft_orient[raftID])) # unit: N.um
        magnetic_Field_Torque_Term[raftID] = tb * magnetic_Field_Torque * 1e6 /(8*piMiuR*R**2) # unit: 1/s
        
        for neighborID in np.arange(numOfRafts):
            if neighborID == raftID: 
                continue
            rj = raft_loc[neighborID,:] # unit: micron
            rji = ri - rj # unit: micron
            rji_Norm = np.sqrt(rji[0]**2 + rji[1]**2) # unit: micron
            rji_eeDist = rji_Norm - 2 * R # unit: micron
            rji_Unitized = rji/rji_Norm # unit: micron
            rji_Unitized_CrossZ = np.asarray((rji_Unitized[1], -rji_Unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raftID]) % 360 # unit: deg; assuming both rafts's orientations are the same, modulo operation remember! 
#            if phi_ji == 360: phi_ji = 0 
#            raft_Relative_Orientation_InDeg[neighborID, raftID] = phi_ji
            
            # force terms:
            omegaj = raft_spin_speeds_inRads[neighborID] # need to come back and see how to deal with this. maybe you need to define it as a global variable. 
            
            if rji_eeDist < 10000 and rji_eeDist >= lubEqThreshold: 
                mag_Dipole_Force_OnAxis_Term[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + cm * magDpForceOnAxis[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * rji_Unitized / (6*piMiuR) # unit: um/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                mag_Dipole_Force_OnAxis_Term[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + cm *lubA[int(rji_eeDist * lubCoeffScaleFactor)] * magDpForceOnAxis[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * rji_Unitized / miu # unit: um/s
            elif rji_eeDist < 0: 
                mag_Dipole_Force_OnAxis_Term[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + cm *lubA[0] * magDpForceOnAxis[0, int(phi_ji + 0.5)] * rji_Unitized / miu # unit: um/s
            
            if rji_eeDist < 1000 and rji_eeDist >= lubEqThreshold: 
                capillary_Force_Term[raftID, :] = capillary_Force_Term[raftID, :] + cc * capillaryForcesDistancesAsRows[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * rji_Unitized / (6*piMiuR) # unit: um/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                capillary_Force_Term[raftID, :] = capillary_Force_Term[raftID, :] + cc *lubA[int(rji_eeDist * lubCoeffScaleFactor)] * capillaryForcesDistancesAsRows[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * rji_Unitized / miu # unit: um/s
            elif rji_eeDist < 0:
                capillary_Force_Term[raftID, :] = capillary_Force_Term[raftID, :] + cc *lubA[0] * capillaryForcesDistancesAsRows[0, int(phi_ji + 0.5)] * rji_Unitized / miu # unit: um/s

            if rji_eeDist >= lubEqThreshold:
                hydrodynamic_Force_Term[raftID, :] = hydrodynamic_Force_Term[raftID, :] + ch * 1e-6 * densityOfWater * omegaj**2 * R**7 * rji / rji_Norm**4 / (6*piMiuR) # unit: um/s; 1e-6 is used to convert the implicit m to um in Newton in miu
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                hydrodynamic_Force_Term[raftID, :] = hydrodynamic_Force_Term[raftID, :] + ch *lubA[int(rji_eeDist * lubCoeffScaleFactor)] * (1e-6 * densityOfWater * omegaj**2 * R**7 / rji_Norm**3) * rji_Unitized / miu # unit: um/s
            
            if rji_eeDist < 0:
                wall_Repulsion_Term[raftID, :] = wall_Repulsion_Term[raftID, :] + wallRepulsionForce * (-rji_eeDist/R) * rji_Unitized / (6*piMiuR)
            
            if rji_eeDist < 10000 and rji_eeDist >= lubEqThreshold: 
                mag_Dipole_Force_OffAxis_Term[raftID, :] = mag_Dipole_Force_OffAxis_Term[raftID, :] + magDpForceOffAxis[int(rji_eeDist + 0.5 ), int(phi_ji + 0.5)] * rji_Unitized_CrossZ / (6*piMiuR)
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                mag_Dipole_Force_OffAxis_Term[raftID, :] = mag_Dipole_Force_OffAxis_Term[raftID, :] + lubB[int(rji_eeDist * lubCoeffScaleFactor)] * magDpForceOffAxis[int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * rji_Unitized_CrossZ / miu # unit: um/s
            elif rji_eeDist < 0: 
                mag_Dipole_Force_OffAxis_Term[raftID, :] = mag_Dipole_Force_OffAxis_Term[raftID, :] + lubB[0] * magDpForceOffAxis[0, int(phi_ji + 0.5)] * rji_Unitized_CrossZ / miu # unit: um/s
            
            if rji_eeDist >= lubEqThreshold:
                velocity_Torque_Coupling_Term[raftID, :] = velocity_Torque_Coupling_Term[raftID, :] - R**3 * omegaj * rji_Unitized_CrossZ / (rji_Norm ** 2) # unit: um/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                velocity_Mag_Fd_Torque_Term[raftID, :] = velocity_Mag_Fd_Torque_Term[raftID, :] + lubC[int(rji_eeDist * lubCoeffScaleFactor)] * magnetic_Field_Torque * 1e6 * rji_Unitized_CrossZ / miu# unit: um/s
            elif rji_eeDist < 0:
                velocity_Mag_Fd_Torque_Term[raftID, :] = velocity_Mag_Fd_Torque_Term[raftID, :] + lubC[0] * magnetic_Field_Torque * 1e6 * rji_Unitized_CrossZ / miu # unit: um/s
            
#                if rji_eeDist >= lubEqThreshold and currentStepNum > 1:
#                    prev_drdt = (raftLocations[raftID,currentStepNum,:] -  raftLocations[raftID,currentStepNum-1,:]) / timeStepSize
#                    stochasticTerm[raftID, currentStepNum,:] =  stochasticTerm[raftID, currentStepNum,:] + np.sqrt(prev_drdt[0]**2 + prev_drdt[1]**2) * np.random.normal(0, stdOfFluctuationTerm, 1) * rjiUnitized 

        # update drdr and dalphadt
        drdt[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + capillary_Force_Term[raftID, :] + hydrodynamic_Force_Term[raftID, :] + wall_Repulsion_Term[raftID, :] \
                + mag_Dipole_Force_OffAxis_Term[raftID, :] + velocity_Torque_Coupling_Term[raftID, :] + velocity_Mag_Fd_Torque_Term[raftID, :] \
                + stochastic_Term[raftID, :] +  curvature_Force_Term[raftID, :] + boundary_Force_Term[raftID, :]
                
    dalphadt = raft_spin_speeds_inRads / np.pi * 180  # in deg
    
    drdt_dalphadt = np.concatenate((drdt.flatten(), dalphadt))
    
    return drdt_dalphadt



#for forceDueToCurvature in np.array([0]):
for magneticFieldRotationRPS in np.arange(-25, -40, -5): # negative means clockwise in Rhino coordinate, and positive means counter-clockwise in Rhino coordinate
#    magneticFieldRotationRPS = -10 # unit: rps (rounds per seconds)
    
    omegaBField = magneticFieldRotationRPS*2*np.pi # unit: rad/s
#    forceDueToCurvature = 0 #1e-10 # unit: N
#    forceDueToBoundary = 1e-6 # unit: N
    
    #initialize key dataset
    raftLocations = np.zeros((numOfRafts, numOfTimeSteps, 2)) # in microns
    raftOrientations = np.zeros((numOfRafts, numOfTimeSteps)) # in deg
    raftRadii = np.ones(numOfRafts) * raftRadius # in micron
    raftRotationSpeedsInRad = np.zeros((numOfRafts, numOfTimeSteps)) # in rad
    
    
#    magDipoleForceOnAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    capillaryForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    hydrodynamicForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    magDipoleForceOffAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    velocityTorqueCouplingTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    velocityMagDpTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    wallRepulsionTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    stochasticTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    curvatureForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    boundaryForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
#    
#    magneticFieldTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
#    magneticDipoleTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
#    capillaryTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    
    #initialize variables for order parameters:
    entropyByNeighborDistances = np.zeros(numOfTimeSteps)
    
    hexaticOrderParameterAvgs = np.zeros(numOfTimeSteps, dtype = np.csingle)
    hexaticOrderParameterAvgNorms = np.zeros(numOfTimeSteps)
    hexaticOrderParameterMeanSquaredDeviations = np.zeros(numOfTimeSteps, dtype = np.csingle)
    hexaticOrderParameterModuliiAvgs = np.zeros(numOfTimeSteps)
    hexaticOrderParameterModuliiStds = np.zeros(numOfTimeSteps)
    
    radialDistributionFunction = np.zeros((numOfTimeSteps, len(radialRangeArray))) # pair correlation function: g(r)
    spatialCorrHexaOrderPara = np.zeros((numOfTimeSteps, len(radialRangeArray))) # spatial correlation of hexatic order paramter: g6(r)
    spatialCorrHexaBondOrientationOrder = np.zeros((numOfTimeSteps, len(radialRangeArray))) # spatial correlation of bond orientation parameter: g6(r)/g(r)
            
    dfNeighbors = pd.DataFrame(columns = ['frameNum', 'raftID', 'hexaticOrderParameter', 
                                              'neighborDistances', 'neighborDistancesAvg'])
    
    dfNeighborsAllFrames = pd.DataFrame(columns = ['frameNum', 'raftID', 'hexaticOrderParameter', 
                                                   'neighborDistances', 'neighborDistancesAvg'])
    
    
    currentStepNum = 0
    if initialPositionMethod == 1:
        #initialize the raft positions in the first frame, check pairwise ccdistance all above 2R
        paddingAroundArena = 5 # unit: radius
        ccDistanceMin = 2.5 # unit: radius
        raftLocations[:,currentStepNum,:] = np.random.uniform(0 + raftRadius*paddingAroundArena, arenaSize - raftRadius*paddingAroundArena, (numOfRafts, 2))
        pairwiseDistances = scipyDistance.cdist(raftLocations[:,currentStepNum,:], raftLocations[:,currentStepNum,:], 'euclidean')
        np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
        raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
        raftsToRelocate = np.unique(raftsToRelocate)
        
        while len(raftsToRelocate) > 0: 
            raftLocations[raftsToRelocate,currentStepNum,:] = np.random.uniform(0 + raftRadius*paddingAroundArena, arenaSize - raftRadius*paddingAroundArena, (len(raftsToRelocate), 2))
            pairwiseDistances = scipyDistance.cdist(raftLocations[:,currentStepNum,:], raftLocations[:,currentStepNum,:], 'euclidean')
            np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
            raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
            raftsToRelocate = np.unique(raftsToRelocate)
    elif initialPositionMethod == 2 or (initialPositionMethod == 3 and firstSpinSpeedFlag == 1):
        raftLocations[:,currentStepNum,:] = SquareSpiral(numOfRafts, raftRadius*2 + 100, centerOfArena)
        firstSpinSpeedFlag = 0
    elif initialPositionMethod == 3 and firstSpinSpeedFlag == 0:
        raftLocations[0,currentStepNum,:] = lastPositionOfPreviousSpinSpeeds[0,:]
        raftLocations[1,currentStepNum,:] = lastPositionOfPreviousSpinSpeeds[1,:]
     
#        check the initial position of rafts   
#        currentFrameBGR = DrawRaftsRhinoCoord(blankFrameBGR.copy(), np.int64(raftLocations[:,currentStepNum,:]/scaleBar), np.int64(raftRadii/scaleBar), numOfRafts)
#        currentFrameBGR = DrawSym6PositionsRhinoCoord(currentFrameBGR, np.int64(raftLocations[:,currentStepNum,:]/scaleBar), 
#                                               raftOrientations[:,currentStepNum],capillaryPeakOffset, np.int64(raftRadii/scaleBar), numOfRafts)
#        currentFrameBGR = DrawRaftOrientationsRhinoCoord(currentFrameBGR, np.int64(raftLocations[:,currentStepNum,:]/scaleBar), 
#                                               raftOrientations[:,currentStepNum],np.int64(raftRadii/scaleBar), numOfRafts) 
#        currentFrameBGR = DrawRaftNumberRhinoCoord(currentFrameBGR, np.int64(raftLocations[:,currentStepNum,:]/scaleBar), numOfRafts)
#        plt.imshow(currentFrameBGR)
        
    
    outputFileName = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_' + str(magneticFieldRotationRPS).zfill(3) + 'rps_B' + str(magneticFieldStrength) + 'T_m' + str(magneticMomentOfOneRaft) + \
                            'Am2_capPeak' + str(capillaryPeakOffset) + '_curvF' + str(forceDueToCurvature) + \
                            '_startPosMeth' + str(initialPositionMethod) + \
                            '_lubEqThres' + str(lubEqThreshold) + '_timeStep' + str(timeStepSize) + '_' + str(timeTotal) + 's'
    
    if outputVideo == 1:
        outputVideoName = outputFileName + '.mp4'
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        frameW, frameH, _ = blankFrameBGR.shape
        videoOut = cv.VideoWriter(outputVideoName,fourcc, outputFrameRate, (frameH, frameW), 1)
    
    
    for currentStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
        # currentStepNum = 0
#        if currentStepNum == 10000:
#            forceDueToCurvature = 0
        
        magneticFieldDirection = MapAnglesTo0To360(magneticFieldRotationRPS * 360 * currentStepNum * timeStepSize)
        
        raftLocationsOrientations = np.concatenate((raftLocations[:, currentStepNum, :].flatten(), raftOrientations[:, currentStepNum]))
        
        sol = solve_ivp(Fun_drdt_dalphadt, (0, timeStepSize), raftLocationsOrientations, method = solverMethod)
        
        raftLocations[:, currentStepNum+1, :] = sol.y[0:numOfRafts*2,-1].reshape(numOfRafts, 2)
        raftOrientations[:, currentStepNum+1] = sol.y[numOfRafts*2 : numOfRafts*3, -1]
        
        # Voronoi calculation: 
        vor = scipyVoronoi(raftLocations[:,currentStepNum,:])
        allVertices = vor.vertices
        neighborPairs = vor.ridge_points # row# is the index of a ridge, columns are the two point# that correspond to the ridge 
        ridgeVertexPairs = np.asarray(vor.ridge_vertices) # row# is the index of a ridge, columns are two vertex# of the ridge
        pairwiseDistances = scipyDistance.cdist(raftLocations[:,currentStepNum,:], raftLocations[:,currentStepNum,:], 'euclidean')
        
        # calculate hexatic order parameter and entropy by neighbor distances 
        for raftID in np.arange(numOfRafts): 
            # raftID = 0       
            ri = raftLocations[raftID,currentStepNum,:] # unit: micron
            
            # neighbors of this particular raft: 
            ridgeIndices0 =  np.nonzero(neighborPairs[:,0] == raftID)
            ridgeIndices1 =  np.nonzero(neighborPairs[:,1] == raftID)
            ridgeIndices = np.concatenate((ridgeIndices0, ridgeIndices1), axis = None)
            neighborPairsOfOneRaft = neighborPairs[ridgeIndices,:]
            NNsOfOneRaft = np.concatenate((neighborPairsOfOneRaft[neighborPairsOfOneRaft[:,0] == raftID,1], neighborPairsOfOneRaft[neighborPairsOfOneRaft[:,1] == raftID,0]))
            neighborDistances = pairwiseDistances[raftID, NNsOfOneRaft]
            
            # calculate hexatic order parameter of this one raft
            neighborLocations = raftLocations[NNsOfOneRaft,currentStepNum,:]
            neighborAnglesInRad = np.arctan2(-(neighborLocations[:,1] - ri[1]),(neighborLocations[:,0] - ri[0])) # note the negative sign, it is to make the angle Rhino-like
            raftHexaticOrderParameter = np.cos(neighborAnglesInRad*6).mean() + np.sin(neighborAnglesInRad*6).mean()*1j
            
            
            dfNeighbors.at[raftID, 'frameNum'] = currentStepNum
            dfNeighbors.at[raftID, 'raftID'] = raftID
            dfNeighbors.at[raftID, 'hexaticOrderParameter'] = raftHexaticOrderParameter
            dfNeighbors.at[raftID, 'neighborDistances'] = neighborDistances
            dfNeighbors.at[raftID, 'neighborDistancesAvg'] = neighborDistances.mean()
            
        
        # calculate order parameters for the current time step: 
        hexaticOrderParameterList =  dfNeighbors['hexaticOrderParameter'].tolist()
        neighborDistancesList = np.concatenate(dfNeighbors['neighborDistances'].tolist())
        
        hexaticOrderParameterArray = np.array(hexaticOrderParameterList)
        hexaticOrderParameterAvgs[currentStepNum] = hexaticOrderParameterArray.mean()
        hexaticOrderParameterAvgNorms[currentStepNum] = np.sqrt(hexaticOrderParameterAvgs[currentStepNum].real ** 2 + hexaticOrderParameterAvgs[currentStepNum].imag ** 2)
        hexaticOrderParameterMeanSquaredDeviations[currentStepNum] = ((hexaticOrderParameterArray - hexaticOrderParameterAvgs[currentStepNum]) ** 2).mean()
        hexaticOrderParameterMolulii = np.absolute(hexaticOrderParameterArray)
        hexaticOrderParameterModuliiAvgs[currentStepNum] = hexaticOrderParameterMolulii.mean()
        hexaticOrderParameterModuliiStds[currentStepNum] = hexaticOrderParameterMolulii.std()
        
        count, _ = np.histogram(np.asarray(neighborDistancesList)/raftRadius, binEdgesNeighborDistances)
        entropyByNeighborDistances[currentStepNum] = ShannonEntropy(count)
        
        ## g(r) and g6(r) for this frame
        for radialIndex, radialIntervalStart in enumerate(radialRangeArray): 
            radialIntervalEnd =  radialIntervalStart + deltaR
            # g(r)
            js, ks = np.logical_and(pairwiseDistances>=radialIntervalStart, pairwiseDistances<radialIntervalEnd).nonzero()
            count = len(js)
            density = numOfRafts / arenaSize**2 
            radialDistributionFunction[currentStepNum, radialIndex] =  count / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
            # g6(r)
            sumOfProductsOfPsi6 = (hexaticOrderParameterArray[js] * np.conjugate(hexaticOrderParameterArray[ks])).sum().real
            spatialCorrHexaOrderPara[currentStepNum, radialIndex] = sumOfProductsOfPsi6 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts-1))
            # g6(r)/g(r)
            if radialDistributionFunction[currentStepNum, radialIndex] != 0: 
                spatialCorrHexaBondOrientationOrder[currentStepNum, radialIndex] = spatialCorrHexaOrderPara[currentStepNum, radialIndex] / radialDistributionFunction[currentStepNum, radialIndex]

#        dfNeighborsAllFrames = dfNeighborsAllFrames.append(dfNeighbors,ignore_index=True)
        
        # draw current frame
        if (outputImageSeq == 1 or outputVideo == 1) and (currentStepNum % intervalBetweenFrames == 0):
            currentFrameBGR = DrawRaftsRhinoCoord(blankFrameBGR.copy(), np.int64(raftLocations[:,currentStepNum,:]/scaleBar), np.int64(raftRadii/scaleBar), numOfRafts)
            currentFrameBGR = DrawRaftOrientationsRhinoCoord(currentFrameBGR, np.int64(raftLocations[:,currentStepNum,:]/scaleBar), 
                                                   raftOrientations[:,currentStepNum],np.int64(raftRadii/scaleBar), numOfRafts)
            currentFrameBGR = DrawSym6PositionsRhinoCoord(currentFrameBGR, np.int64(raftLocations[:,currentStepNum,:]/scaleBar), 
                                                   raftOrientations[:,currentStepNum],capillaryPeakOffset, np.int64(raftRadii/scaleBar), numOfRafts)
            currentFrameBGR = DrawRaftNumberRhinoCoord(currentFrameBGR, np.int64(raftLocations[:,currentStepNum,:]/scaleBar), numOfRafts)
            currentFrameBGR = DrawFrameInfoManyRafts(currentFrameBGR, currentStepNum, hexaticOrderParameterAvgNorms[currentStepNum], hexaticOrderParameterModuliiAvgs[currentStepNum], entropyByNeighborDistances[currentStepNum])

            if outputImageSeq == 1:
                outputImageName =  outputFileName + str(currentStepNum).zfill(7) + '.jpg'
                cv.imwrite(outputImageName,currentFrameBGR)
            if outputVideo == 1:
                videoOut.write(np.uint8(currentFrameBGR))
    
    if outputVideo == 1: 
        videoOut.release()
    
    
    tempShelf = shelve.open(outputFileName)
    for key in listOfVariablesToSave:
        try:
            tempShelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, tempShelf, and imported modules can not be shelved.
            #
            #print('ERROR shelving: {0}'.format(key))
            pass
    tempShelf.close()
    
#    lastPositionOfPreviousSpinSpeeds[:,:] = raftLocations[:,currentStepNum,:]
#    lastOmegaOfPreviousSpinSpeeds[:] = raftRotationSpeedsInRad[:, currentStepNum]


#%% load simulated data in one main folder

rootFolderNameforSimulation = r'C:\SimulationFolder'
os.chdir(rootFolderNameforSimulation)
rootFolderTreeGen = os.walk(rootFolderNameforSimulation)
_, mainFolders, _ = next(rootFolderTreeGen) 

mainFolderID = 28
os.chdir(mainFolders[mainFolderID])

dataFileList = glob.glob('*.dat')
dataFileList.sort()

mainDataList = []
variableListsForAllMainData = []

for dataID in range(len(dataFileList)):
    dataFileToLoad = dataFileList[dataID].partition('.dat')[0]
    
    tempShelf = shelve.open(dataFileToLoad)
    variableListOfOneMainDataFile = list(tempShelf.keys())
    
    expDict = {}
    for key in tempShelf:
        try:
            expDict[key] = tempShelf[key]
        except TypeError:
            pass
    
    tempShelf.close()
    mainDataList.append(expDict)
    variableListsForAllMainData.append(variableListOfOneMainDataFile)


#%% output results to csv files
    
dfOrderParameters = pd.DataFrame(columns = ['time(s)'])
dfEntropies = pd.DataFrame(columns = ['time(s)'])
selectEveryNPoint = 100


for dataID in range(len(mainDataList)):
    numOfTimeSteps = mainDataList[dataID]['numOfTimeSteps']
    timeStepSize = mainDataList[dataID]['timeStepSize']
    
    dfOrderParameters['time(s)'] = np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize
    dfEntropies['time(s)'] = np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize
    
    magneticFieldRotationRPS = mainDataList[dataID]['magneticFieldRotationRPS']
    hexaticOrderParameterAvgNorms = mainDataList[dataID]['hexaticOrderParameterAvgNorms']
    hexaticOrderParameterModuliiAvgs = mainDataList[dataID]['hexaticOrderParameterModuliiAvgs']
    hexaticOrderParameterModuliiStds = mainDataList[dataID]['hexaticOrderParameterModuliiStds']
    entropyByNeighborDistances = mainDataList[dataID]['entropyByNeighborDistances']
    
    colName = str(-mainDataList[dataID]['magneticFieldRotationRPS']).zfill(4)
    
    dfOrderParameters[colName + '_avgNorm'] = hexaticOrderParameterAvgNorms[0::selectEveryNPoint]
    dfOrderParameters[colName + '_ModuliiAvg'] = hexaticOrderParameterModuliiAvgs[0::selectEveryNPoint]
    dfOrderParameters[colName + '_ModuliiStds'] = hexaticOrderParameterModuliiStds[0::selectEveryNPoint]
    dfEntropies[colName] = entropyByNeighborDistances[0::selectEveryNPoint]
    

dfOrderParameters.to_csv('orderParameters.csv', index = False)
dfEntropies.to_csv('entropies.csv', index = False)

#%% load one specific simulation data
    
dataID = 2

variableListFromSimulatedFile = list(mainDataList[dataID].keys())

for key, value in mainDataList[dataID].items(): # loop through key-value pairs of python dictionary
    globals()[key] = value

# hexatic order parameter vs time, error bars
selectEveryNPoint = 100
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.errorbar(np.arange(0,numOfTimeSteps, selectEveryNPoint) * timeStepSize, hexaticOrderParameterModuliiAvgs[0::selectEveryNPoint], yerr = hexaticOrderParameterModuliiStds[0::selectEveryNPoint], label = '<|phi6|>')
ax.set_xlabel('Time (s)',size=20)
ax.set_ylabel('order parameter',size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show() 



# hexatic order parameter vs time
selectEveryNPoint = 100
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(0,numOfTimeSteps, selectEveryNPoint) * timeStepSize, hexaticOrderParameterModuliiAvgs[0::selectEveryNPoint], label = '<|phi6|>')
ax.plot(np.arange(0,numOfTimeSteps, selectEveryNPoint) * timeStepSize, hexaticOrderParameterAvgNorms[0::selectEveryNPoint], label = '|<phi6>|')
ax.set_xlabel('Time (s)',size=20)
ax.set_ylabel('order parameter',size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show() 

# entropy vs time
selectEveryNPoint = 100
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(0,numOfTimeSteps, selectEveryNPoint) * timeStepSize, entropyByNeighborDistances[0::selectEveryNPoint], label = 'entropy by distances')
ax.set_xlabel('Time (s)',size=20)
ax.set_ylabel('entropy by neighbor distances',size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show() 


