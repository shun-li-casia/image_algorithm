/*******************************************************************************
 *   Copyright (C) 2023 CASIA. All rights reserved.
 *
 *   @Filename: stereo_optical_matcher.h
 *
 *   @Author: shun li
 *
 *   @Email: shun.li.at.casia@outlook.com
 *
 *   @Date: 19/10/2023
 *
 *   @Description:
 *
 *******************************************************************************/

#include <opencv2/opencv.hpp>
#include <utility_tool/print_ctrl_macro.h>

#include <vector>

namespace image_algorithm {
class StereoOpticalMatcher {
 public:
  static void MatchLR(const cv::Mat& left, const cv::Mat& right, const int num,
                      const float dist, cv::Mat* flow_img,
                      cv::Mat* flow_back_img) {
    if (left.empty()) {
      PCM_PRINT_ERROR("left image is empty!\n");
      return;
    }
    if (right.empty()) {
      PCM_PRINT_ERROR("right image is empty!\n");
      return;
    }

    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> left_pts, right_pts;

    cv::goodFeaturesToTrack(left, left_pts, num, 0.01, dist);

    cv::calcOpticalFlowPyrLK(
        left, right, left_pts, right_pts, status, err, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01));

    // floaw back check
    std::vector<uchar> reverse_status;
    std::vector<float> reverse_err;
    std::vector<cv::Point2f> reverse_pts = left_pts;

    cv::calcOpticalFlowPyrLK(
        right, left, right_pts, reverse_pts, reverse_status, reverse_err,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    for (size_t i = 0; i < status.size(); i++) {
      if (status[i] && reverse_status[i] &&
          Distance(left_pts[i], reverse_pts[i]) <= 0.5) {
        status[i] = 1;
      } else {
        status[i] = 0;
      }
    }

    ReduceVector(status, &left_pts);
    ReduceVector(status, &right_pts);

    cv::hconcat(left, right, *flow_back_img);
    cv::cvtColor(*flow_back_img, *flow_back_img, cv::COLOR_GRAY2RGB);

    assert(left_pts.size() == right_pts.size());

    cv::RNG rng(1);
    for (size_t i = 0; i < left_pts.size(); ++i) {
      int b = rng.uniform(0, 255);
      int g = rng.uniform(0, 255);
      int r = rng.uniform(0, 255);

      // left pts
      cv::circle(*flow_back_img, left_pts[i], 1, cv::Scalar(b, g, r), 2);
      // right pts
      cv::Point2f r_pt = right_pts[i];
      r_pt.x += left.cols;
      cv::circle(*flow_back_img, r_pt, 1, cv::Scalar(b, g, r), 2);

      // line
      cv::line(*flow_back_img, left_pts[i], r_pt, cv::Scalar(b, g, r));
    }
  }

 private:
  template <typename T>
  static void ReduceVector(std::vector<uchar> status, std::vector<T>* vec) {
    int j = 0;
    for (size_t i = 0; i < vec->size(); ++i) {
      if (status[i]) (*vec)[j++] = (*vec)[i];
    }
    vec->resize(j);
  }
  static double Distance(cv::Point2f pt1, cv::Point2f pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
  }
};
}  // namespace image_imu_processor
