//
//  RandomForest.h
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef JC_RANDOMFOREST_H
#define JC_RANDOMFOREST_H

#include <QtConcurrent>

#include "tree.h"

namespace Detection
{

class RandomForest
{
public:
    std::vector<std::vector<Tree> > rfs_;
    int max_numtrees_;
    int num_landmark_;
    int max_depth_;
    double overlap_ratio_;

    RandomForest(int max_numtrees, int num_landmark, int max_depth, double overlap_ratio) {
        max_numtrees_  = max_numtrees;
        num_landmark_  = num_landmark;
        max_depth_     = max_depth;
        overlap_ratio_ = overlap_ratio;

        // resize the trees
        rfs_.resize(num_landmark_);
        for (int i=0;i<num_landmark_;i++)
            rfs_[i].resize(max_numtrees_, Tree(max_depth));
    }

    void Train(const std::vector<cv::Mat_<uchar> >& images,
               const std::vector<cv::Mat_<double> >& ground_truth_shapes,
               const std::vector<cv::Mat_<double> >& current_shapes,
               const std::vector<BoundingBox> & bounding_boxs,
               const cv::Mat_<double>& mean_shape,
               const std::vector<cv::Mat_<double> >& shapes_residual,
               int max_numfeats,
               double max_radio_radius
               );

    void Read(std::ifstream& fin);
    void Write(std::ofstream& fout);
};

} // namespace detection

#endif // JC_RANDOMFOREST_H
