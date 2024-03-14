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

#include "image_algorithm/disparity_calculator_cuda.h"
#include "libsgm.h"

namespace image_algorithm {

// FIXME(shun li): there are still no good results!
void DisparityCalculatorCuda::CalcuDisparityCuda(const cv::Mat& left,
                                                 const cv::Mat& right,
                                                 const Param& param,
                                                 cv::Mat* disp,
                                                 cv::Mat* rgb_disparity) {
  const int width = left.cols;
  const int height = left.rows;

  const int disp_size = 256;
  const int input_depth = left.type() == CV_8U ? 8 : 16;
  const int input_bytes = input_depth * width * height / 8;
  const int output_depth = disp_size < 256 ? 8 : 16;  // should be 16
  const int output_bytes = output_depth * width * height / 8;

  sgm::StereoSGM::Parameters cuda_param;
  cuda_param.P1 = param.p1;
  cuda_param.P2 = param.p2;
  cuda_param.min_disp = param.min_disp;
  cuda_param.LR_max_diff = param.disp12_max_diff;
  cuda_param.subpixel = true;
  cuda_param.uniqueness = param.uniqueness_ratio;
  cuda_param.path_type = sgm::PathType::SCAN_4PATH;
  sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth,
                     sgm::EXECUTE_INOUT_CUDA2CUDA, cuda_param);

  cv::Mat disparity(height, width, CV_16S);

  DeviceBuffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);

  cudaMemcpy(d_I1.data, left.data, input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_I2.data, right.data, input_bytes, cudaMemcpyHostToDevice);

  sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
  cudaDeviceSynchronize();

  cudaMemcpy(disparity.data, d_disparity.data, output_bytes,
             cudaMemcpyDeviceToHost);

  disparity.convertTo(*disp, CV_32F, 1.0 / 16.0f);

  // colored disparity image
  cv::Mat disp8;
  disparity.convertTo(disp8, CV_8U, 255 / (256 * 16.));
  cv::applyColorMap(disp8, *rgb_disparity, cv::COLORMAP_JET);
}

}  // namespace image_algorithm
