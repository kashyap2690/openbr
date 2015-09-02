//
//  LBFRegressor.h
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef LBFREGRESSOR_H
#define LBFREGRESSOR_H

#include "randomforest.h"
#include <linear.h>

namespace Detection
{

struct LBFParams
{
    double bagging_overlap;
    int max_numtrees;
    int max_depth;
    int landmark_num;// to be decided
    int initial_num;

    int max_numstage;
    double max_radio_radius[10];
    int max_numfeats[10]; // number of pixel pairs
    int max_numthreshs;
};

class LBFRegressor
{
public:
    LBFRegressor(LBFParams &params) {
        params_ = params;
        forest_.resize(params.max_numstage, RandomForest(params.max_numtrees, params.landmark_num, params.max_depth, params.bagging_overlap));
        models_.resize(params.max_numstage);
    }
    ~LBFRegressor() {}

    void Train(const std::vector<cv::Mat_<uchar> >& images,
               const std::vector<cv::Mat_<double> >& ground_truth_shapes,
               const std::vector<BoundingBox> & bounding_boxs);

    void                           Predict();
    cv::Mat_<double>               Predict(const cv::Mat_<uchar>& image, const BoundingBox& bounding_box);
    std::vector<cv::Mat_<double> > Predict(const std::vector<cv::Mat_<uchar> >& images, const std::vector<BoundingBox>& bounding_boxs);

    void Load(std::string path);
    void Save(std::string path);

private:
    struct feature_node **DeriveBinaryFeat(const RandomForest &randf,
                                            const std::vector<cv::Mat_<uchar> > &images,
                                            const std::vector<cv::Mat_<double> > &current_shapes,
                                            const std::vector<BoundingBox> &bounding_boxs,
                                            const cv::Mat_<double> &mean_shape);
    void ReleaseFeatureSpace(struct feature_node **binfeatures, int num_train_sample);

    int   GetCodefromTree(const Tree &tree,
                          const cv::Mat_<uchar> &image,
                          const cv::Mat_<double> &shapes,
                          const BoundingBox &bounding_box,
                          const cv::Mat_<double> &rotation,
                          const double scale);

    void GlobalRegression(struct feature_node **binfeatures,
                          const std::vector<cv::Mat_<double> > &shapes_residual,
                          std::vector<cv::Mat_<double> > &current_shapes,
                          const std::vector<BoundingBox> &bounding_boxs,
                          const cv::Mat_<double> &mean_shape,
                          std::vector<struct model*> &models,
                          int num_feature,
                          int num_train_sample
                         );

    void GlobalPrediction(struct feature_node**,
                          const std::vector<struct model*> &models,
                          std::vector<cv::Mat_<double> > &current_shapes,
                          const std::vector<BoundingBox> &bounding_boxs
                         );

    void saveModel(std::ofstream &fout, struct model *model);
    struct model *loadModel(std::ifstream &fin);

    std::vector<RandomForest> forest_;
    std::vector<std::vector<struct model*> > models_;
    cv::Mat_<double> mean_shape_;
    std::vector<cv::Mat_<double> > shapes_residual_;
    LBFParams params_;
};

} // namespace Detection

#endif
