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

#ifndef IMAGE_PREPROCESSOR_DISPARITY_CALULATOR_H_
#define IMAGE_PREPROCESSOR_DISPARITY_CALULATOR_H_

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>

namespace image_algorithm {
class DisparityCalculator {
 public:
  struct Param {
    int block_size{5};
    int p1{400};
    int p2{1600};
    int min_disp{0};
    int num_disparities{128};
    int speckle_window_size{60};
    int speckle_range{2};
    int uniqueness_ratio{6};
    int disp12_max_diff{200};
    int pre_filter_cap{1};

    Param() = default;

    bool SaveToYaml(const std::string& filename) const {
      YAML::Node node;
      node["block_size"] = block_size;
      node["p1"] = p1;
      node["p2"] = p2;
      node["min_disp"] = min_disp;
      node["num_disparities"] = num_disparities;
      node["speckle_window_size"] = speckle_window_size;
      node["speckle_range"] = speckle_range;
      node["uniqueness_ratio"] = uniqueness_ratio;
      node["disp12_max_diff"] = disp12_max_diff;
      node["pre_filter_cap"] = pre_filter_cap;
      YAML::Emitter out;
      out << node;
      std::ofstream fout(filename);
      fout << out.c_str();
      fout.close();
      return true;
    }

    bool ReadFromYaml(const std::string& filename) {
      YAML::Node node = YAML::LoadFile(filename);
      block_size = node["block_size"].as<int>();
      p1 = node["p1"].as<int>();
      p2 = node["p2"].as<int>();
      min_disp = node["min_disp"].as<int>();
      num_disparities = node["num_disparities"].as<int>();
      speckle_window_size = node["speckle_window_size"].as<int>();
      speckle_range = node["speckle_range"].as<int>();
      uniqueness_ratio = node["uniqueness_ratio"].as<int>();
      disp12_max_diff = node["disp12_max_diff"].as<int>();
      pre_filter_cap = node["pre_filter_cap"].as<int>();
      return true;
    }

    void Print() {
      std::cout << "block_size: " << block_size << std::endl;
      std::cout << "p1: " << p1 << std::endl;
      std::cout << "p2: " << p2 << std::endl;
      std::cout << "min_disp: " << min_disp << std::endl;
      std::cout << "num_disparities: " << num_disparities << std::endl;
      std::cout << "speckle_window_size: " << speckle_window_size << std::endl;
      std::cout << "speckle_range: " << speckle_range << std::endl;
      std::cout << "uniqueness_ratio: " << uniqueness_ratio << std::endl;
      std::cout << "disp12_max_diff: " << disp12_max_diff << std::endl;
      std::cout << "pre_filter_cap: " << pre_filter_cap << std::endl;
    }

    Param& operator=(const Param& param) {
      block_size = param.block_size;
      p1 = param.p1;
      p2 = param.p2;
      min_disp = param.min_disp;
      num_disparities = param.num_disparities;
      speckle_window_size = param.speckle_window_size;
      speckle_range = param.speckle_range;
      uniqueness_ratio = param.uniqueness_ratio;
      disp12_max_diff = param.disp12_max_diff;
      pre_filter_cap = param.pre_filter_cap;
      return *this;
    }

    Param(const Param& param) {
      block_size = param.block_size;
      p1 = param.p1;
      p2 = param.p2;
      min_disp = param.min_disp;
      num_disparities = param.num_disparities;
      speckle_window_size = param.speckle_window_size;
      speckle_range = param.speckle_range;
      uniqueness_ratio = param.uniqueness_ratio;
      disp12_max_diff = param.disp12_max_diff;
      pre_filter_cap = param.pre_filter_cap;
    }
  };

  DisparityCalculator();

  DisparityCalculator(const Param& param);

  void SetParam(const Param& param);

  void CalcuDisparitySGBM(const cv::Mat& left, const cv::Mat& right,
                          cv::Mat* disparity, cv::Mat* rgb_disparity);

  static void CalcuDisparitySGBM(const cv::Mat& left, const cv::Mat& right,
                                 const Param& param, cv::Mat* disparity,
                                 cv::Mat* rgb_disparity);
  static void CalcuDisparityBM(const cv::Mat& left, const cv::Mat& right,
                               const Param& param, cv::Mat* disparity,
                               cv::Mat* rgb_disparity);

  static void CalcuDisparityQuasiDense(const cv::Mat& left,
                                       const cv::Mat& right, const Param& param,
                                       cv::Mat* disparity,
                                       cv::Mat* rgb_disparity);

 private:
  Param param_;
  cv::Ptr<cv::StereoSGBM> sgbm_;
};
}  // namespace image_algorithm

#endif
