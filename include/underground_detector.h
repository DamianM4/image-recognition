#pragma once
#include <opencv4/opencv2/core.hpp>
#include <queue>

// ### TODO: includes to be deleted in final version ###
#include <opencv4/opencv2/highgui.hpp>
// ################

namespace detector {
namespace CONSTS {
constexpr int CHANNEL_NUM = 3;

const int UPPER_RED = 250;
const int LOWER_RED = 180;

const int UPPER_GREEN = 230;
const int LOWER_GREEN = 135;

const int UPPER_BLUE = 140;
const int LOWER_BLUE = 0;

const double W7_MIN = 0.7;
const double W9_MIN = 0.5;
}  // namespace CONSTS

struct ColorSpan {
  struct Span {
    int lower;
    int upper;
    Span() : lower(0), upper(0) {}
    Span(int l, int u) : lower(l), upper(u) {}
  };
  // array stores BGR spans
  std::array<Span, CONSTS::CHANNEL_NUM> spans;
  ColorSpan() {
    for (int i = 0; i < CONSTS::CHANNEL_NUM; i++) {
      spans[i] = Span();
    }
  }
  ColorSpan(int up_r, int low_r, int up_g, int low_g, int up_b, int low_b) {
    spans[0] = Span(low_b, up_b);
    spans[1] = Span(low_g, up_g);
    spans[2] = Span(low_r, up_r);
  }
};

struct Point {
  int x, y;
  Point() : x(0), y(0) {}
  Point(int x_coor, int y_coor) : x(x_coor), y(y_coor) {}
};

struct BoundingBox {
  Point up_corner;
  int width;
  int height;
  BoundingBox(Point start_point, int w, int h)
      : up_corner(start_point), width(w), height(h) {}
};

class LogoDetector {
  // ### TODO: change to private in final version ###
 public:
  // ################

  cv::Mat img_;
  cv::Mat bin_img_;
  ColorSpan metro_yellow_;

  // image preparation and morphology methods
  cv::Mat binarizeByColor(cv::Mat img, ColorSpan color);
  cv::Mat dilate(const cv::Mat &img, int kernel_size = 3);
  cv::Mat erode(const cv::Mat &img, int kernel_size = 3);
  cv::Mat cropImage(const cv::Mat &img, BoundingBox b_box);

  // high level recognition methods
  std::vector<BoundingBox> getCirclesCandidates(const cv::Mat &img);
  std::vector<BoundingBox> findCircles(const cv::Mat &img);
  bool isShapeACircle(const cv::Mat &cropped_img);
  bool isMLetterFound(const cv::Mat &cropped_img);

  // fillEnclosedAreas methods
  cv::Mat fillEnclosedAreas();
  cv::Mat invertImage(const cv::Mat &img);
  cv::Mat logicOr(const cv::Mat &img1, const cv::Mat &img2);
  cv::Mat logicAnd(const cv::Mat &img1, const cv::Mat &img2);

  // scanFill methods
  cv::Mat scanFill(const cv::Mat &img, Point fill_point, int fill_value = 255);
  bool isInsideOfContour(const cv::Mat &img, Point point, int inside_color = 0);
  void addScan(const cv::Mat &img, int up_x, int down_x, int y,
               std::queue<Point> &q);

  // shape recognition methods
  double calculateArea(const cv::Mat &cropped_img);
  double calculatePerimeter(const cv::Mat &cropped_img);
  Point calculateMidpoint(const cv::Mat &cropped_img);
  double calcW7coeff(const cv::Mat &cropped_img);
  double calcW9coeff(const cv::Mat &cropped_img);

  double m(const cv::Mat &img, int p = 0, int q = 0);

 public:
  LogoDetector(cv::Mat &img);

  std::vector<BoundingBox> findMetroSigns();
};
}  // namespace detector
