/*
 Author: Bi Sai
 Date: 2014/06/18
 This program is a reimplementation of algorithms in "Face Alignment by Explicit
 Shape Regression" by Cao et al.
 If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com
 Copyright (c) 2014 Bi Sai
 The MIT License (MIT)
 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef JC_UTILITY_H
#define JC_UTILITY_H

#include "opencv2/imgproc/imgproc.hpp"

#include <QDebug>
#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>
#include <utility>

namespace Detection
{

struct BoundingBox
{
    BoundingBox() {
        start_x = start_y = 0;
        width = height = 0;
        centroid_x = centroid_y = 0;
    };

    double start_x, start_y;
    double width, height;
    double centroid_x, centroid_y;
};

cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
                              const std::vector<BoundingBox>& bounding_box);

void GetShapeResidual(const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                      const std::vector<cv::Mat_<double> >& current_shapes,
                      const std::vector<BoundingBox>& bounding_boxs,
                      const cv::Mat_<double>& mean_shape,
                      std::vector<cv::Mat_<double> >& shape_residuals);

cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2, cv::Mat_<double>& rotation,double& scale);
double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape);

} // namespace Detection

#endif // JC_UTILITY_H
