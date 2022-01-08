#include <iostream>
#include <opencv4/opencv2/highgui.hpp>

#include "underground_detector.h"

// ### TODO: includes to be deleted in final version ###
#include <opencv4/opencv2/imgproc.hpp>
// ################

int main(int argc, char *argv[]) {
  auto image = cv::imread("./images/40_40px_circle_test.png");
  detector::LogoDetector detector(image);
  cv::Mat thresh(image.size(), CV_8UC1);
  cv::cvtColor(image, thresh, cv::COLOR_BGR2GRAY);

  thresh = detector.invertImage(thresh);

  cv::imshow("window", thresh);
  cv::waitKey(0);

  // cv::imwrite("images/m_fill_test_ret.jpg", res);

  auto w7 = detector.calcW7coeff(thresh);
  auto w9 = detector.calcW9coeff(thresh);

  std::cout << w7 << " " << w9 << "\n";

  

  // for (int i = 1; i < argc; i++) {
  //   auto image = cv::imread(argv[i]);
  //   if (image.data == NULL) {
  //     std::cout << "\033[1;31m"
  //               << "[ERROR] Cannot find the \"" << argv[i] << "\" image."
  //               << "\033[0m" << std::endl;
  //   } else { // normal operation
  //     detector::LogoDetector detector(image);

  //     cv::imshow("window", image);
  //     cv::waitKey(0);
  //   }
  // }
}
