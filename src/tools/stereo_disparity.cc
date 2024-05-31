/*******************************************************************************
 *   Copyright (C) 2023 CASIA. All rights reserved.
 *
 *   @Filename: stereo_disparity.cc
 *
 *   @Author: shun li
 *
 *   @Email: shun.li.at.casia@outlook.com
 *
 *   @Date: 29/05/2024
 *
 *   @Description: calculate the stereo disparity from the rect image.
 *
 *******************************************************************************/

#include "image_algorithm/disparity_calculator.h"
#include "utility_tool/pcm_debug_helper.h"
#include "utility_tool/cmdline.h"
#include "stereo_msgs/DisparityImage.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

float resize_scale = 1;
image_algorithm::DisparityCalculator calculator;
image_algorithm::DisparityCalculator::Param param;

void imgCallback(const sensor_msgs::ImageConstPtr& msg) {
  utility_tool::Timer timer;
  try {
    cv::Mat rect_img = cv_bridge::toCvCopy(msg, "bgr8")->image;

    if (rect_img.empty()) return;

    cv::Mat left_img =
        rect_img(cv::Rect(0, 0, rect_img.cols / 2, rect_img.rows));
    cv::Mat right_img = rect_img(
        cv::Rect(rect_img.cols / 2, 0, rect_img.cols / 2, rect_img.rows));

    cv::resize(left_img, left_img, cv::Size(), resize_scale, resize_scale);
    cv::resize(right_img, right_img, cv::Size(), resize_scale, resize_scale);

    cv::Mat disp, rgb_disp;
    calculator.CalcuDisparitySGBM(left_img, right_img, &disp, &rgb_disp);

    cv::imshow("rgb_disp", rgb_disp);
    cv::waitKey(1);

  } catch (cv_bridge::Exception& e) {
    PCM_PRINT_ERROR("Could not convert from '%s' to 'bgr8'. \n",
                    msg->encoding.c_str());
  }

  PCM_PRINT_DEBUG("time cost: %f, fps: %f\n", timer.End(),
                  1000.0 / timer.End());
}

int main(int argc, char** argv) {
  cmdline::parser par;
  par.add<float>("resize_scale", 's', "resize scale", true);
  par.parse_check(argc, argv);
  resize_scale = par.get<float>("resize_scale");

  ros::init(argc, argv, "stereo_disparity");
  ros::NodeHandle nh;
  param.block_size = 3;
  param.num_disparities = 64;

  calculator.SetParam(param);
  ros::Subscriber sub = nh.subscribe("/hconcate_image_cam_0", 1, imgCallback);

  ros::spin();
}
