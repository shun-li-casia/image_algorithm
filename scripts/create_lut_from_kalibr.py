#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 CASIA. All rights reserved.
#
# @Filename: create_lut_from_kalibr.py
#
# @Author: shun li
#
# @Email: shun.li.at.casia@outlook.com
#
# @Date: 15/01/2024
#
# @Description:

import os
import sys
import numpy as np
import cv2
import yaml
import argparse


def parse_calib_file(calib_file_path):
    with open(calib_file_path, 'r') as file:
        calibr = yaml.safe_load(file)

    left = calibr['cam0']
    right = calibr['cam1']

    intrinsics1 = np.array(left['intrinsics'])
    intrinsics2 = np.array(right['intrinsics'])

    l_fx = float(intrinsics1[0])
    l_fy = float(intrinsics1[1])
    l_cx = float(intrinsics1[2])
    l_cy = float(intrinsics1[3])

    r_fx = float(intrinsics2[0])
    r_fy = float(intrinsics2[1])
    r_cx = float(intrinsics2[2])
    r_cy = float(intrinsics2[3])

    # left camera params
    K1 = np.array([[l_fx, 0.,   l_cx],
                   [0.,   l_fy, l_cy],
                   [0.,   0.,   1.]])
    # right camera params
    K2 = np.array([[r_fx,  0.,   r_cx],
                   [0.,    r_fy, r_cy],
                   [0.,    0.,   1.]])

    D1 = np.array(left['distortion_coeffs'])
    D2 = np.array(right['distortion_coeffs'])

    transform = np.array(right['T_cn_cnm1'])

    image_size = tuple(left['resolution'])

    R = transform[:3, :3]
    T = transform[:3, 3].reshape(-1, 1)

    return K1, D1, K2, D2, image_size, R, T


def save_LDC_lut(map_x, map_y, file_path):
    """ save LDC LUT to given file_path
    """
    LDC_DS_FACTOR = 4

    width = map_x.shape[1]
    height = map_x.shape[0]
    sWidth = int(width / LDC_DS_FACTOR + 1)
    sHeight = int(height / LDC_DS_FACTOR + 1)
    lineOffset = ((sWidth + 15) & (~15))

    ldcLUT_x = np.zeros((sHeight, lineOffset), dtype=np.int16)
    ldcLUT_y = np.zeros((sHeight, lineOffset), dtype=np.int16)

    for y in range(0, sHeight):
        j = y * LDC_DS_FACTOR
        if j > height - 1:
            j = height - 1

        for x in range(0, sWidth):
            i = x * LDC_DS_FACTOR
            if i > width - 1:
                i = width - 1

            dx = np.floor(map_x[j, i]*8. + 0.5).astype(int) - i*8
            dy = np.floor(map_y[j, i]*8. + 0.5).astype(int) - j*8

            ldcLUT_x[y, x] = dx & 0xFFFF
            ldcLUT_y[y, x] = dy & 0xFFFF

        remain = ((sWidth + 15) & (~15)) - sWidth
        while (remain > 0):
            ldcLUT_x[y, sWidth - 1 + remain] = 0
            ldcLUT_y[y, sWidth - 1 + remain] = 0
            remain = remain - 1

    #  y offset comes first
    col0 = np.floor(ldcLUT_y.flatten().reshape(-1, 1)).astype(np.int16)
    col1 = np.floor(ldcLUT_x.flatten().reshape(-1, 1)).astype(np.int16)
    ldcLUT = np.concatenate((col0, col1), axis=1)

    ldcLUT.tofile(file_path)
    print("LDC LUT saved into {}".format(file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ZED camera camera_info & LCD LUT generation tool')

    parser.add_argument('--input', '-i', type=str, default='.yaml',
                        help='the stereo camera calibration file genreate from kalibr.')

    args = parser.parse_args()
    calib_file = args.input

    if not os.path.exists(calib_file):
        sys.exit(f'{calib_file} does not exist.')

    K1, D1, K2, D2, image_size, R, T = parse_calib_file(calib_file)

    # cv2.stereoRectify
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=K1, distCoeffs1=D1,
                                                      cameraMatrix2=K2, distCoeffs2=D2,
                                                      imageSize=image_size, R=R, T=T,
                                                      flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    #  create remap table for left
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    # create remap table for right
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    # write LDC undist/rect LUT
    save_LDC_lut(map_left_x, map_left_y, "LUT_left.bin")
    save_LDC_lut(map_right_x, map_right_y, "LUT_right.bin")
