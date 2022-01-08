#include "underground_detector.h"

#include <cmath>
#include <iostream>

using namespace detector;

LogoDetector::LogoDetector(cv::Mat &img) : img_(img.clone()) {
  bin_img_ = cv::Mat(img_.size(), CV_8UC1);
  metro_yellow_ =
      ColorSpan(CONSTS::UPPER_RED, CONSTS::LOWER_RED, CONSTS::UPPER_GREEN,
                CONSTS::LOWER_GREEN, CONSTS::UPPER_BLUE, CONSTS::LOWER_BLUE);
}

auto LogoDetector::findMetroSigns() -> std::vector<BoundingBox> {
  bin_img_ = binarizeByColor(img_, metro_yellow_);
  bin_img_ = dilate(bin_img_);
  bin_img_ = erode(bin_img_);
  auto b_boxes = findCircles(bin_img_);
}

cv::Mat LogoDetector::binarizeByColor(cv::Mat img, ColorSpan color) {
  cv::Mat color_bin_img = cv::Mat(img.size(), CV_8UC1);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      for (int channel = 0; channel < CONSTS::CHANNEL_NUM; channel++) {
        if (img.at<uchar>(i, j, channel) < color.spans[channel].upper &&
            img.at<uchar>(i, j, channel) > color.spans[channel].lower) {
          color_bin_img.at<uchar>(i, j) = 255;
        } else {
          color_bin_img.at<uchar>(i, j) = 0;
        }
      }
    }
  }
  return color_bin_img;
}

cv::Mat LogoDetector::dilate(const cv::Mat &img, int kernel_size) {
  cv::Mat ret_img(img.clone());
  for (int i = kernel_size / 2; i < ret_img.rows - kernel_size / 2; i++) {
    for (int j = kernel_size / 2; j < ret_img.cols - kernel_size / 2; j++) {
      if (i < kernel_size / 2 || i >= ret_img.rows - kernel_size / 2 ||
          j < kernel_size / 2 || j >= ret_img.cols - kernel_size / 2) {
        // adding black frame on border of the image of width = kernel_size/2
        ret_img.at<uchar>(i, j) = 0;
      } else {
        // performing dilation
        bool paint_pixel_white = 0;
        for (int ii = -kernel_size / 2; ii < kernel_size / 2; ii++) {
          for (int jj = -kernel_size / 2; jj < kernel_size / 2; jj++) {
            if (ret_img.at<uchar>(i + ii, j + jj) == 255) {
              paint_pixel_white = true;
              break;
            }
          }
          if (paint_pixel_white) {
            break;
          }
        }
        if (paint_pixel_white) {
          ret_img.at<uchar>(i, j) = 255;
        }
      }
    }
  }
  return ret_img;
}

cv::Mat LogoDetector::erode(const cv::Mat &img, int kernel_size) {
  cv::Mat ret_img(img.clone());
  for (int i = kernel_size / 2; i < ret_img.rows - kernel_size / 2; i++) {
    for (int j = kernel_size / 2; j < ret_img.cols - kernel_size / 2; j++) {
      if (i < kernel_size / 2 || i >= ret_img.rows - kernel_size / 2 ||
          j < kernel_size / 2 || j >= ret_img.cols - kernel_size / 2) {
        // adding black frame on border of the image of width = kernel_size/2
        ret_img.at<uchar>(i, j) = 0;
      } else {
        // performing erosion
        bool paint_pixel_black = 0;
        for (int ii = -kernel_size / 2; ii < kernel_size / 2; ii++) {
          for (int jj = -kernel_size / 2; jj < kernel_size / 2; jj++) {
            if (ret_img.at<uchar>(i + ii, j + jj) == 0) {
              paint_pixel_black = true;
              break;
            }
          }
          if (paint_pixel_black) {
            break;
          }
        }
        if (paint_pixel_black) {
          ret_img.at<uchar>(i, j) = 0;
        }
      }
    }
  }
  return ret_img;
}

cv::Mat LogoDetector::cropImage(const cv::Mat &img, BoundingBox b_box) {
  cv::Mat cropped(b_box.width, b_box.height, CV_8UC1);
  for (int i = b_box.up_corner.x, m = 0; i < cropped.rows, m < cropped.rows;
       i++, m++) {
    for (int j = b_box.up_corner.y, n = 0; j < cropped.cols, n < cropped.cols;
         j++, n++) {
      cropped.at<uchar>(m, n) = img.at<uchar>(i, j);
    }
  }
  return cropped;
}

auto LogoDetector::getCirclesCandidates(const cv::Mat &img)
    -> std::vector<BoundingBox> {
  std::vector<BoundingBox> b_boxes;
  cv::Mat base_img(img.clone());
  cv::Mat mask(img.clone());
  for (int i = 0; i < base_img.rows; i++) {
    for (int j = 0; j < base_img.cols; j++) {
      // if edge is encountered
      if (base_img.at<uchar>(i - 1, j) != base_img.at<uchar>(i + 1, j) ||
          base_img.at<uchar>(i, j - 1) != base_img.at<uchar>(i, j + 1)) {
        mask = scanFill(base_img, Point(i, j), 0);
        auto inv_mask = invertImage(mask);
        auto candidate_mask = logicAnd(inv_mask, base_img);
        int x0 = candidate_mask.rows - 1, y0 = candidate_mask.cols - 1, x1 = 0,
            y1 = 0;
        for (int m = 0; m < candidate_mask.rows; m++) {
          for (int n = 0; n < candidate_mask.cols; n++) {
            if (base_img.at<uchar>(i - 1, j) != base_img.at<uchar>(i + 1, j) ||
                base_img.at<uchar>(i, j - 1) != base_img.at<uchar>(i, j + 1)) {
              if (i < x0) {
                x0 = i;
              }
              if (i > x1) {
                x1 = i;
              }
              if (j < y0) {
                y0 = j;
              }
              if (j > y1) {
                y1 = j;
              }
            }
          }
        }
        BoundingBox bb(Point(x0, y0), std::abs(x1 - x0), std::abs(y1 - y0));
        b_boxes.push_back(bb);

        auto inv_candidate_mask = invertImage(candidate_mask);
        base_img = logicAnd(inv_mask, base_img);
      }
    }
  }
  return b_boxes;
}

auto LogoDetector::findCircles(const cv::Mat &img) -> std::vector<BoundingBox> {
  auto b_boxes = getCirclesCandidates(img);
  std::vector<BoundingBox> circular_shapes_boxes;
  for (auto b_box : b_boxes) {
    auto cropped_img = cropImage(img, b_box);
    if (isShapeACircle(cropped_img)) {
      circular_shapes_boxes.push_back(b_box);
    }
  }
  return circular_shapes_boxes;
}

bool LogoDetector::isShapeACircle(const cv::Mat &cropped_img) {
  auto w7 = calcW7coeff(cropped_img);
  auto w9 = calcW9coeff(cropped_img);
  if (w7 > CONSTS::W7_MIN && w9 > CONSTS::W9_MIN) {
    return true;
  }
  return false;
}

bool LogoDetector::isMLetterFound(const cv::Mat &cropped_img) {}

cv::Mat LogoDetector::fillEnclosedAreas() {
  auto filled_img = scanFill(bin_img_, Point(1, 1));
  filled_img = invertImage(filled_img);
  filled_img = logicOr(filled_img, bin_img_);
  return filled_img;
}

cv::Mat LogoDetector::invertImage(const cv::Mat &img) {
  cv::Mat inverted(img.size(), CV_8UC1);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<uchar>(i, j) == 255) {
        inverted.at<uchar>(i, j) = 0;
      } else {
        inverted.at<uchar>(i, j) = 255;
      }
    }
  }
  return inverted;
}

cv::Mat LogoDetector::logicOr(const cv::Mat &img1, const cv::Mat &img2) {
  if (img1.rows == img2.rows && img1.cols == img2.cols) {
    std::cout << "Images' sizes don't match in line: " << __LINE__ << "\n";
  }
  cv::Mat or_img(img1.size(), CV_8UC1);
  for (int i = 0; i < or_img.rows; i++) {
    for (int j = 0; j < or_img.cols; j++) {
      if (img1.at<uchar>(i, j) == 255 || img2.at<uchar>(i, j) == 255) {
        or_img.at<uchar>(i, j) = 255;
      } else {
        or_img.at<uchar>(i, j) = 0;
      }
    }
  }
  return or_img;
}

cv::Mat LogoDetector::logicAnd(const cv::Mat &img1, const cv::Mat &img2) {
  if (img1.rows == img2.rows && img1.cols == img2.cols) {
    std::cout << "Images' sizes don't match in line: " << __LINE__ << "\n";
  }
  cv::Mat and_img(img1.size(), CV_8UC1);
  for (int i = 0; i < and_img.rows; i++) {
    for (int j = 0; j < and_img.cols; j++) {
      if (img1.at<uchar>(i, j) == 255 && img2.at<uchar>(i, j) == 255) {
        and_img.at<uchar>(i, j) = 255;
      } else {
        and_img.at<uchar>(i, j) = 0;
      }
    }
  }
  return and_img;
}

cv::Mat LogoDetector::scanFill(const cv::Mat &img, Point fill_point,
                               int fill_value) {
  cv::Mat filled_img(img.clone());
  if (!isInsideOfContour(img, fill_point)) {
    std::cout << "Selcted fill point: (" << fill_point.x << ", " << fill_point.y
              << ") is outside of contour.\n";
    return filled_img;
  }
  std::queue<Point> q;
  q.push(fill_point);
  while (!q.empty()) {
    Point current_point = q.front();
    q.pop();
    int up_x = current_point.x;
    int y = current_point.y;
    std::cout << "working... " << up_x << " " << y << "\n";
    while (isInsideOfContour(filled_img, Point(up_x, y))) {
      filled_img.at<uchar>(up_x, y) = fill_value;
      up_x--;
    }
    int x = current_point.x + 1;
    while (isInsideOfContour(filled_img, Point(x, y))) {
      filled_img.at<uchar>(x, y) = fill_value;
      x++;
    }
    addScan(filled_img, up_x + 1, x, y + 1, q);
    addScan(filled_img, up_x + 1, x, y - 1, q);
  }
  return filled_img;
}

bool LogoDetector::isInsideOfContour(const cv::Mat &filled_img, Point point,
                                     int inside_color) {
  if (point.x < 0 || point.x >= filled_img.rows || point.y < 0 ||
      point.y >= filled_img.cols) {
    return false;
  }
  if (filled_img.at<uchar>(point.x, point.y) == inside_color) {
    return true;
  }
  return false;
}

void LogoDetector::addScan(const cv::Mat &img, int up_x, int down_x, int y,
                           std::queue<Point> &q) {
  bool added = false;
  for (int x = up_x; x < down_x; x++) {
    if (!isInsideOfContour(img, Point(x, y))) {
      added = false;
    } else if (!added) {
      q.push(Point(x, y));
      added = true;
    }
  }
}

double LogoDetector::calculateArea(const cv::Mat &cropped_img) {
  double area = 0.;
  for (int i = 0; i < cropped_img.rows; i++) {
    for (int j = 0; j < cropped_img.cols; j++) {
      if (cropped_img.at<uchar>(i, j) == 255) {
        area++;
      }
    }
  }
  return area;
}

double LogoDetector::calculatePerimeter(const cv::Mat &cropped_img) {
  double perimeter = 0.;
  for (int i = 1; i < cropped_img.rows - 1; i++) {
    for (int j = 1; j < cropped_img.cols - 1; j++) {
      if (cropped_img.at<uchar>(i - 1, j) != cropped_img.at<uchar>(i + 1, j) ||
          cropped_img.at<uchar>(i, j - 1) != cropped_img.at<uchar>(i, j + 1)) {
        perimeter++;
      }
    }
  }
  return perimeter;
}

Point LogoDetector::calculateMidpoint(const cv::Mat &cropped_img) {
  Point midpoint;
  midpoint.x = m(cropped_img, 1, 0) / m(cropped_img, 0, 0);
  midpoint.y = m(cropped_img, 0, 1) / m(cropped_img, 0, 0);
  return midpoint;
}

double LogoDetector::calcW7coeff(const cv::Mat &cropped_img) {
  double min_dist = std::hypot(cropped_img.cols, cropped_img.rows);
  double max_dist = 0.;
  double dist = 0.;
  Point midpoint = calculateMidpoint(cropped_img);
  std::cout << "midpoint " << midpoint.x << " " << midpoint.y << "\n";
  for (int i = 1; i < cropped_img.rows - 1; i++) {
    for (int j = 1; j < cropped_img.cols - 1; j++) {
      if (cropped_img.at<uchar>(i - 1, j) != cropped_img.at<uchar>(i + 1, j) ||
          cropped_img.at<uchar>(i, j - 1) != cropped_img.at<uchar>(i, j + 1)) {
        dist = hypot(midpoint.x - i, midpoint.y - j);
        if (dist > max_dist) {
          max_dist = dist;
        }
        if (dist < min_dist) {
          min_dist = dist;
        }
      }
    }
  }
  std::cout << "dists " << min_dist << " " << max_dist << "\n";
  double W7 = min_dist / max_dist;
  return W7;
}

double LogoDetector::calcW9coeff(const cv::Mat &cropped_img) {
  const double PI = atan(1) * 4;
  auto S = calculateArea(cropped_img);
  auto L = calculatePerimeter(cropped_img);
  return 2 * sqrt(PI * S) / L;
}

double LogoDetector::m(const cv::Mat &img, int p, int q) {
  double sum = 0.;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<uchar>(i, j) == 255) {
        sum += pow(static_cast<double>(i), static_cast<double>(p)) *
               pow(static_cast<double>(j), static_cast<double>(q)) * 1.;
      }
    }
  }
  return sum;
}
