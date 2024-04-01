/*******************************************************************************
 *   Copyright (C) 2023 CASIA. All rights reserved.
 *
 *   @Filename: disparity_calculator.h
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

#include "image_algorithm/disparity_calculator.h"
#include <opencv2/stereo.hpp>

namespace image_algorithm {

void DisparityCalculator::CalcuDisparitySGBM(const cv::Mat& left,
                                             const cv::Mat& right,
                                             const Param& param, cv::Mat* disp,
                                             cv::Mat* rgb_disparity) {
  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
  sgbm->setPreFilterCap(param.pre_filter_cap);
  sgbm->setBlockSize(param.block_size);
  sgbm->setP1(param.p1);
  sgbm->setP2(param.p2);
  sgbm->setMinDisparity(param.min_disp);
  sgbm->setNumDisparities(param.num_disparities);
  sgbm->setUniquenessRatio(param.uniqueness_ratio);
  sgbm->setSpeckleWindowSize(param.speckle_window_size);
  sgbm->setSpeckleRange(param.speckle_range);
  sgbm->setDisp12MaxDiff(param.disp12_max_diff);
  sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

  cv::Mat disparity_sgbm;
  sgbm->compute(left, right, disparity_sgbm);

  disparity_sgbm.convertTo(*disp, CV_32F, 1.0 / 16.0f);

  // colored disparity image
  cv::Mat disp8;
  disparity_sgbm.convertTo(disp8, CV_8U, 255 / (128 * 16.));
  cv::applyColorMap(disp8, *rgb_disparity, cv::COLORMAP_JET);

  // cv::Ptr<cv::ximgproc::DisparityWLSFilter> filter =
  //     cv::ximgproc::createDisparityWLSFilter(sgbm);
  // filter->filter(disparity_sgbm, left, disparity_sgbm);
}

void DisparityCalculator::CalcuDisparityBM(const cv::Mat& left,
                                           const cv::Mat& right,
                                           const Param& param, cv::Mat* disp,
                                           cv::Mat* rgb_disparity) {
  cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
  bm->setBlockSize(param.block_size);
  bm->setMinDisparity(param.min_disp);
  bm->setNumDisparities(param.num_disparities);

  cv::Mat disparity_sgbm;
  bm->compute(left, right, disparity_sgbm);
  disparity_sgbm.convertTo(*disp, CV_32F, 1.0 / 16.0f);

  // colored disparity image
  cv::Mat disp8;
  disparity_sgbm.convertTo(disp8, CV_8U, 255 / (128 * 16.));
  cv::applyColorMap(disp8, *rgb_disparity, cv::COLORMAP_JET);

  // cv::Ptr<cv::ximgproc::DisparityWLSFilter> filter =
  //     cv::ximgproc::createDisparityWLSFilter(sgbm);
  // filter->filter(disparity_sgbm, left, disparity_sgbm);
}

void DisparityCalculator::CalcuDisparityQuasiDense(const cv::Mat& left,
                                                   const cv::Mat& right,
                                                   const Param& param,
                                                   cv::Mat* disp,
                                                   cv::Mat* rgb_disparity) {
  cv::Size frameSize = left.size();
  cv::Ptr<cv::stereo::QuasiDenseStereo> stereo =
      cv::stereo::QuasiDenseStereo::create(frameSize);
  stereo->process(left, right);

  cv::Mat disparity_sgbm = stereo->getDisparity();
  disparity_sgbm.convertTo(*disp, CV_32F, 1.0 / 16.0f);

  // colored disparity image
  cv::Mat disp8;
  disparity_sgbm.convertTo(disp8, CV_8U, 255 / (128 * 16.));
  cv::applyColorMap(disp8, *rgb_disparity, cv::COLORMAP_JET);
}
}  // namespace image_algorithm
