/*******************************************************************************
 *   Copyright (C) 2023 CASIA. All rights reserved.
 *
 *   @Filename: stereo_rectifier.h
 *
 *   @Author: shun li
 *
 *   @Email: shun.li.at.casia@outlook.com
 *
 *   @Date: 28/09/2023
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef IMAGE_PREPROCESSOR_STEREO_RECTIFIER_H_
#define IMAGE_PREPROCESSOR_STEREO_RECTIFIER_H_

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <utility>

namespace image_algorithm {
class StereoRectifier {
 public:
  static void RectStereoParam(const Eigen::Matrix3d& Rrl,
                              const Eigen::Vector3d& trl,
                              Eigen::Matrix3d* rect_Rrl,
                              Eigen::Vector3d* rect_trl,
                              camodocal::PinholeCamera::Parameters* cam_param_l,
                              camodocal::PinholeCamera::Parameters* cam_param_r,
                              std::pair<cv::Mat, cv::Mat>* cam_l_maps,
                              std::pair<cv::Mat, cv::Mat>* cam_r_maps);
};
}  // namespace image_preprocessor

#endif
