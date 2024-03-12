#include "utility_tool/cmdline.h"
#include "utility_tool/print_ctrl_macro.h"
#include "image_algorithm/disparity_calculator.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  cmdline::parser par;
  // read the image and remap them, then save to the rect_ dir
  par.add<std::string>("l_img", 0, "left image", true);
  par.add<std::string>("r_img", 0, "right image", true);
  par.parse_check(argc, argv);

  const string lfile = par.get<string>("l_img");
  const string rfile = par.get<string>("r_img");

  string lpath = lfile.substr(0, lfile.find_last_of("/"));

  cv::Mat left = imread(lfile, IMREAD_GRAYSCALE);
  cv::Mat right = imread(rfile, IMREAD_GRAYSCALE);

  // cv::resize(left, left, cv::Size(), 0.5, 0.5);
  // cv::resize(right, right, cv::Size(), 0.5, 0.5);

  cv::Mat disp;

  image_algorithm::DisparityCalculator::Param param;

  param.min_disp = 0;
  param.num_disparities = 128;
  // 匹配块大小，大于1的奇数
  param.block_size = 3;
  // P1, P2控制视差图的光滑度
  // 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，P2 = 4 * P1
  param.p1 = 16 * param.block_size * param.block_size;
  // p1控制视差平滑度，p2值越大，差异越平滑
  param.p2 = 64 * param.block_size * param.block_size;
  // 左右视差图的最大容许差异（超过将被清零），默认为 -
  // 1，即不执行左右视差检查。
  param.disp12_max_diff = 200;
  param.pre_filter_cap = 0;
  // 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio /
  // 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为
  // 0，通常为5~15.
  param.uniqueness_ratio = 6;
  // 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。
  // 否则，将其设置在50
  // - 200的范围内。
  param.speckle_window_size = 60;
  // 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为
  // 正值，它将被隐式乘以16.通常，1或2就足够好了
  param.speckle_range = 2;

  cv::namedWindow("SGBM_disparity");
  cv::createTrackbar("blockSize", "SGBM_disparity", NULL, 11);
  cv::setTrackbarPos("blockSize", "SGBM_disparity", param.block_size);

  cv::createTrackbar("p1", "SGBM_disparity", NULL, 32);
  cv::setTrackbarPos("p1", "SGBM_disparity", param.p1);

  cv::createTrackbar("p2", "SGBM_disparity", NULL, 128);
  cv::setTrackbarPos("p2", "SGBM_disparity", param.p2);

  cv::createTrackbar("numDisparities", "SGBM_disparity", NULL, 256);
  cv::setTrackbarPos("numDisparities", "SGBM_disparity", param.num_disparities);

  cv::createTrackbar("speckleWindowSize", "SGBM_disparity", NULL, 200);
  cv::setTrackbarPos("speckleWindowSize", "SGBM_disparity",
                     param.speckle_window_size);

  cv::createTrackbar("speckleRange", "SGBM_disparity", NULL, 50);
  cv::setTrackbarPos("speckleRange", "SGBM_disparity", param.speckle_range);

  cv::createTrackbar("uniquenessRatio", "SGBM_disparity", NULL, 50);
  cv::setTrackbarPos("uniquenessRatio", "SGBM_disparity",
                     param.uniqueness_ratio);

  cv::createTrackbar("disp12MaxDiff", "SGBM_disparity", NULL, 21);
  cv::setTrackbarPos("disp12MaxDiff", "SGBM_disparity", param.disp12_max_diff);

  cv::createTrackbar("preFilterCap", "SGBM_disparity", NULL, 10);
  cv::setTrackbarPos("preFilterCap", "SGBM_disparity", param.pre_filter_cap);

  fstream f;
  int update_cnt = 0;
  while (true) {
    param.block_size = cv::getTrackbarPos("blockSize", "SGBM_disparity");
    param.p1 = cv::getTrackbarPos("p1", "SGBM_disparity") * param.block_size * param.block_size;
    param.p2 = cv::getTrackbarPos("p2", "SGBM_disparity") * param.block_size * param.block_size;

    param.num_disparities =
        cv::getTrackbarPos("numDisparities", "SGBM_disparity");
    param.speckle_window_size =
        cv::getTrackbarPos("speckleWindowSize", "SGBM_disparity");
    param.speckle_range = cv::getTrackbarPos("speckleRange", "SGBM_disparity");
    param.uniqueness_ratio =
        cv::getTrackbarPos("uniquenessRatio", "SGBM_disparity");
    param.disp12_max_diff =
        cv::getTrackbarPos("disp12MaxDiff", "SGBM_disparity");
    param.pre_filter_cap = cv::getTrackbarPos("preFilterCap", "SGBM_disparity");

    cv::Mat disp, rgb_disparity;
    image_algorithm::DisparityCalculator::CalcuDisparity(
        left, right, param, &disp, &rgb_disparity);

    // cv::resize(rgb_disparity, rgb_disparity, cv::Size(), 0.5, 0.5);
    cv::imshow("SGBM_disparity", rgb_disparity);
    cv::waitKey(0);

    // save the parameters and disparity image
    cv::imwrite(to_string(update_cnt) + "_sgbm_param.png", rgb_disparity);
    param.SaveToYaml(to_string(update_cnt) + "_sgbm_param.yaml");

    PCM_PRINT_DEBUG("update at update_cnt is %d!\n", update_cnt);
    update_cnt++;
  }
  return 0;
}
