"""
The function definition files.

"""

import cv2 as cv
import numpy as np
from scipy.spatial import Voronoi as scipyVoronoi


def draw_rafts_lh_coord(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    """
    draw circles in the left-handed coordinate system of openCV
    positive x is pointing right
    positive y is pointing down
    :param numpy array img_bgr: input bgr image in numpy array
    :param numpy array rafts_loc: locations of the rafts
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    circle_thickness = int(2)
    circle_color = (0, 0, 255)  # openCV: BGR

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1]), rafts_radii[raft_id],
                               circle_color, circle_thickness)

    return output_img


def draw_rafts_rh_coord(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    """
    draw circles in the right-handed coordinate system
    x pointing right
    y pointing up
    :param numpy array img_bgr: input bgr image in numpy array
    :param numpy array rafts_loc: locations of the rafts
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """
    circle_thickness = int(2)
    circle_color = (0, 0, 255)  # openCV: BGR

    output_img = img_bgr
    height, width, _ = img_bgr.shape
    x_axis_start = (0, height - 10)
    x_axis_end = (width, height - 10)
    y_axis_start = (10, 0)
    y_axis_end = (10, height)
    output_img = cv.line(output_img, x_axis_start, x_axis_end, (0, 0, 0), 4)
    output_img = cv.line(output_img, y_axis_start, y_axis_end, (0, 0, 0), 4)

    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1]),
                               rafts_radii[raft_id], circle_color, circle_thickness)

    return output_img


def draw_b_field_in_rh_coord(img_bgr, b_orient):
    """
    draw the direction of B-field in right-handed xy coordinate
    :param numpy array img_bgr: bgr image file
    :param float b_orient: orientation of the magnetic B-field, in deg
    :return: bgr image file
    """

    output_img = img_bgr
    height, width, _ = img_bgr.shape

    line_length = 200
    line_start = (width // 2, height // 2)
    line_end = (int(width // 2 + np.cos(b_orient * np.pi / 180) * line_length),
                height - int(height // 2 + np.sin(b_orient * np.pi / 180) * line_length))
    output_img = cv.line(output_img, line_start, line_end, (0, 0, 0), 1)
    return output_img


def draw_raft_orientations_lh_coord(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the dipole orientation of each raft,
    as indicated by rafts_ori, in left-handed coordinate system
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
        # note that the sign in front of the sine term is "-"
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    int(rafts_loc[raft_id, 1] - np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]))
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def draw_raft_orientations_rh_coord(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the dipole orientation of each raft,
    as indicated by rafts_ori, in a right-handed coordinate system
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)

    output_img = img_bgr
    height, width, _ = img_bgr.shape

    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1])
        # note that the sign in front of the sine term is "+"
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    height - int(rafts_loc[raft_id, 1] +
                                 np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]))
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def draw_cap_peaks_lh_coord(img_bgr, rafts_loc, rafts_ori, raft_sym, cap_offset, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the capillary peak positions
    in left-handed coordinate system
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param int raft_sym: the symmetry of raft
    :param int cap_offset: the angle between the dipole direction
    and the first capillary peak, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return bgr image file
    """

    line_thickness = int(2)
    line_color2 = (0, 255, 0)
    cap_gap = 360 / raft_sym

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
            line_end = (int(rafts_loc[raft_id, 0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                           * np.pi / 180) * rafts_radii[raft_id]),
                        int(rafts_loc[raft_id, 1] - np.sin((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                           * np.pi / 180) * rafts_radii[raft_id]))
            # note that the sign in front of the sine term is "-"
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img


def draw_cap_peaks_rh_coord(img_bgr, rafts_loc, rafts_ori, raft_sym, cap_offset, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the capillary peak positions
    in right-handed coordinate
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param int raft_sym: the symmetry of raft
    :param int cap_offset: the angle between the dipole direction
    and the first capillary peak, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    line_thickness = int(2)
    line_color2 = (0, 255, 0)
    cap_gap = 360 / raft_sym
    #    cap_offset = 45 # the angle between the dipole direction and the first capillary peak

    output_img = img_bgr
    height, width, _ = img_bgr.shape
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            # note that the sign in front of the sine term is "+"
            line_start = (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1])
            line_end = (int(rafts_loc[raft_id, 0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                           * np.pi / 180) * rafts_radii[raft_id]),
                        height - int(rafts_loc[raft_id, 1] + np.sin((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                                    * np.pi / 180) * rafts_radii[raft_id]))
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img


def draw_raft_number(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    in the left-handed coordinate
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 2
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1),
                                (rafts_loc[raft_id, 0] - text_size[0] // 2, rafts_loc[raft_id, 1] + text_size[1] // 2),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_raft_num_rh_coord(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    in the right-handed coordinate
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 2
    output_img = img_bgr
    height, width, _ = img_bgr.shape

    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1), (rafts_loc[raft_id, 0] - text_size[0] // 2,
                                                               height - (rafts_loc[raft_id, 1] + text_size[1] // 2)),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_frame_info(img_bgr, time_step_num, distance, orientation, b_field_direction, rel_orient):
    """
    draw information on the output frames
    :param numpy array img_bgr: input bgr image
    :param int time_step_num: current step number
    :param float distance: separation distance between two rafts
    :param float orientation: orientation of the raft 0 (same for all rafts)
    :param float b_field_direction: orientation of the B-field
    :param float rel_orient: relative orientation phi_ji
    :return: bgr image
    """
    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 1
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    text_size, _ = cv.getTextSize(str(time_step_num), font_face, font_scale, font_thickness)
    line_padding = 2
    left_padding = 20
    top_padding = 20
    output_img = cv.putText(output_img, 'time step: {}'.format(time_step_num), (left_padding, top_padding), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'distance: {:03.2f}'.format(distance),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 1), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'orientation of raft 0: {:03.2f}'.format(orientation),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 2), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'B_field_direction: {:03.2f}'.format(b_field_direction),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 3), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'relative orientation phi_ji: {:03.2f}'.format(rel_orient),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 4), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('magnetic_dipole_force: {}'.format(magnetic_dipole_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 5), font_face,
    #                         font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('capillary_force: {}'.format(capillary_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 6),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('hydrodynamic_force: {}'.format(hydrodynamic_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 7),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('B-field_torque: {}'.format(B-field_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 8),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('mag_dipole_torque: {}'.format(mag_dipole_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 9),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('cap_torque: {}'.format(cap_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 10),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)

    return output_img


def draw_voronoi_rh_coord(img_bgr, rafts_loc):
    """
    draw Voronoi patterns in the right-handed coordinates
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :return: bgr image file
    """
    height, width, _ = img_bgr.shape
    points = rafts_loc
    points[:, 1] = height - points[:, 1]
    vor = scipyVoronoi(points)
    output_img = img_bgr
    # drawing Voronoi vertices
    vertex_size = int(3)
    vertex_color = (255, 0, 0)
    for x_pos, y_pos in zip(vor.vertices[:, 0], vor.vertices[:, 1]):
        output_img = cv.circle(output_img, (int(x_pos), int(y_pos)), vertex_size, vertex_color)

    # drawing Voronoi edges
    edge_color = (0, 255, 0)
    edge_thickness = int(2)
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            output_img = cv.line(output_img, (int(vor.vertices[simplex[0], 0]), int(vor.vertices[simplex[0], 1])),
                                 (int(vor.vertices[simplex[1], 0]), int(vor.vertices[simplex[1], 1])), edge_color,
                                 edge_thickness)

    center = points.mean(axis=0)
    for point_idx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = points[point_idx[1]] - points[point_idx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = points[point_idx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 200
            output_img = cv.line(output_img, (int(vor.vertices[i, 0]), int(vor.vertices[i, 1])),
                                 (int(far_point[0]), int(far_point[1])), edge_color, edge_thickness)
    return output_img


def fft_general(sampling_rate, signal):
    """
    given sampling rate and signal,
    output frequency vector and one-sided power spectrum
    :param numpy array signal: the input signal in 1D array
    :param float sampling_rate: sampling rate in Hz
    :return: frequencies, one-sided power spectrum, both numpy array
    """
    #    sampling_interval = 1/sampling_rate # unit s
    #    times = np.linspace(0,sampling_length*sampling_interval, sampling_length)
    sampling_length = len(signal)  # total number of frames
    fft = np.fft.fft(signal)
    p2 = np.abs(fft / sampling_length)
    p1 = p2[0:int(sampling_length / 2) + 1]
    p1[1:-1] = 2 * p1[1:-1]  # one-sided power spectrum
    frequencies = sampling_rate / sampling_length * np.arange(0, int(sampling_length / 2) + 1)

    return frequencies, p1


def adjust_phases(phases_input):
    """
    adjust the phases to get rid of the jump of 360
    when it crosses from -180 to 180, or the reverse
    adjust single point anomaly.
    :param numpy array phases_input: initial phases, in deg
    :return: ajusted phases, in deg
    """
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
