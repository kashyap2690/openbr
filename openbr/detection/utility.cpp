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

#include "utility.h"

using namespace std;
using namespace cv;

Mat_<double> Detection::GetMeanShape(const vector<Mat_<double> > &shapes, const vector<BoundingBox> &bounding_box)
{
    Mat_<double> result = Mat::zeros(shapes[0].rows,2,CV_64FC1);
    for(int i = 0; i < shapes.size(); i++)
        result = result + ProjectShape(shapes[i], bounding_box[i]);
    result = 1.0 / shapes.size() * result;
    return result;
}

void Detection::GetShapeResidual(const vector<Mat_<double> > &ground_truth_shapes,
                                 const vector<Mat_<double> > &current_shapes,
                                 const vector<BoundingBox> &bounding_boxs,
                                 const Mat_<double> &mean_shape,
                                 vector<Mat_<double> > &shape_residuals)
{
    Mat_<double> rotation;
    double scale;
    shape_residuals.resize(bounding_boxs.size());
    for (int i = 0; i < bounding_boxs.size(); i++) {
        shape_residuals[i] = ProjectShape(ground_truth_shapes[i], bounding_boxs[i]) - ProjectShape(current_shapes[i], bounding_boxs[i]);
        SimilarityTransform(mean_shape, ProjectShape(current_shapes[i], bounding_boxs[i]), rotation, scale);
        transpose(rotation, rotation);
        shape_residuals[i] = scale * shape_residuals[i] * rotation;
    }
}

Mat_<double> Detection::ProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box)
{
    Mat_<double> temp(shape.rows, 2);
    for(int j = 0; j < shape.rows; j++) {
        temp(j,0) = (shape(j,0) - bounding_box.centroid_x) / (bounding_box.width / 2.0);
        temp(j,1) = (shape(j,1) - bounding_box.centroid_y) / (bounding_box.height / 2.0);
    }
    return temp;
}

Mat_<double> Detection::ReProjectShape(const Mat_<double>& shape, const BoundingBox& bounding_box){
    Mat_<double> temp(shape.rows,2);
    for(int j = 0; j < shape.rows; j++) {
        temp(j,0) = (shape(j,0) * (bounding_box.width / 2.0) + bounding_box.centroid_x);
        temp(j,1) = (shape(j,1) * (bounding_box.height / 2.0) + bounding_box.centroid_y);
    }
    return temp;
}


void Detection::SimilarityTransform(const Mat_<double> &shape1, const Mat_<double> &shape2, Mat_<double> &rotation, double &scale)
{
    rotation = Mat::zeros(2,2,CV_64FC1);
    scale = 0;

    // center the data
    double center_x_1 = 0;
    double center_y_1 = 0;
    double center_x_2 = 0;
    double center_y_2 = 0;
    for (int i = 0; i < shape1.rows; i++) {
        center_x_1 += shape1(i,0);
        center_y_1 += shape1(i,1);
        center_x_2 += shape2(i,0);
        center_y_2 += shape2(i,1);
    }
    center_x_1 /= shape1.rows;
    center_y_1 /= shape1.rows;
    center_x_2 /= shape2.rows;
    center_y_2 /= shape2.rows;

    Mat_<double> temp1 = shape1.clone();
    Mat_<double> temp2 = shape2.clone();
    for (int i = 0; i < shape1.rows;i++) {
        temp1(i,0) -= center_x_1;
        temp1(i,1) -= center_y_1;
        temp2(i,0) -= center_x_2;
        temp2(i,1) -= center_y_2;
    }

    // calculate covariance matrix
    Mat_<double> covariance1, covariance2;
    Mat_<double> mean1, mean2;
    calcCovarMatrix(temp1, covariance1, mean1, CV_COVAR_COLS);
    calcCovarMatrix(temp2, covariance2, mean2, CV_COVAR_COLS);

    double s1 = sqrt(norm(covariance1));
    double s2 = sqrt(norm(covariance2));
    scale = s1 / s2;
    temp1 = 1.0 / s1 * temp1;
    temp2 = 1.0 / s2 * temp2;

    double num = 0;
    double den = 0;
    for (int i = 0; i < shape1.rows; i++) {
        num = num + temp1(i,1) * temp2(i,0) - temp1(i,0) * temp2(i,1);
        den = den + temp1(i,0) * temp2(i,0) + temp1(i,1) * temp2(i,1);
    }

    double norm = sqrt(num*num + den*den);
    double sin_theta = num / norm;
    double cos_theta = den / norm;
    rotation(0,0) = cos_theta;
    rotation(0,1) = -sin_theta;
    rotation(1,0) = sin_theta;
    rotation(1,1) = cos_theta;
}

double Detection::CalculateError(Mat_<double>& ground_truth_shape, Mat_<double>& predicted_shape)
{
    Mat_<double> temp;
    temp = ground_truth_shape.rowRange(36, 41) - ground_truth_shape.rowRange(42, 47);
    double x = mean(temp.col(0))[0];
    double y = mean(temp.col(1))[1];
    double interocular_distance = sqrt(x * x + y * y);
    double sum = 0;
    for (int i = 0; i < ground_truth_shape.rows; i++)
        sum += norm(ground_truth_shape.row(i) - predicted_shape.row(i));
    return sum / (ground_truth_shape.rows * interocular_distance);
}
