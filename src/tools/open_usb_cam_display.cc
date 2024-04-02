#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <sstream>
using namespace std;

// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

#include "ros/ros.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "open_usb_cam_node");
  ros::NodeHandle nh("~");

  // 1.创建视频采集对象;
  VideoCapture cap;
  printf("set the camera!\n");

  // 2.打开默认相机;
  cap.open(std::stoi(argv[1]), CAP_GSTREAMER);

  printf("open the camera!\n");

  // 3.判断相机是否打开成功;
  if (!cap.isOpened()) return -1;

  // 4.显示窗口命名;
  namedWindow("Video", 1);
  for (int i = 0; ros::ok(); ++i) {
    // 获取新的一帧;
    Mat frame;
    cap >> frame;
    if (frame.empty()) return 0;
    // 显示新的帧;
    /* cv::imwrite(to_string(i) + ".png", frame); */
    imshow("Video", frame);

    // 按键退出显示;
    if (waitKey(30) >= 0) break;
    ros::Rate(30).sleep();
  }

  // 5.释放视频采集对象;
  cap.release();

  return 0;
}
