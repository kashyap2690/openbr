//
//  LBFRegressor.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "regressor.h"

using namespace std;
using namespace cv;

struct feature_node **Detection::LBFRegressor::DeriveBinaryFeat (
                                    const RandomForest &randf,
                                    const vector<Mat_<uchar> > &images,
                                    const vector<Mat_<double> > &current_shapes,
                                    const vector<BoundingBox> &bounding_boxs,
                                    const Mat_<double> &mean_shape)
{
    // initilaize the memory for binfeatures
    struct feature_node **binfeatures = new struct feature_node* [images.size()];
    for (int i = 0; i < (int)images.size(); i++)
        binfeatures[i] = new struct feature_node[params_.max_numtrees * params_.landmark_num + 1];

    int bincode;
    int ind;
    int leafnode_per_tree = pow(2,(params_.max_depth - 1));
    Mat_<double> rotation;
    double scale;

    // extract feature for each samples
    for (int i = 0; i < (int)images.size(); i++) {
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]), mean_shape, rotation, scale);
        for (int j = 0; j < params_.landmark_num; j++) {
            for (int k = 0; k < params_.max_numtrees; k++) {
                bincode = GetCodefromTree(randf.rfs_[j][k], images[i], current_shapes[i], bounding_boxs[i], rotation, scale);
                ind = j * params_.max_numtrees + k;
                binfeatures[i][ind].index = leafnode_per_tree * ind + bincode;
                binfeatures[i][ind].value = 1;
            }
        }
        binfeatures[i][params_.landmark_num * params_.max_numtrees].index = -1;
        binfeatures[i][params_.landmark_num * params_.max_numtrees].value = -1;
    }
    return binfeatures;
}

int Detection::LBFRegressor::GetCodefromTree(const Tree &tree,
                                             const Mat_<uchar> &image,
                                             const Mat_<double> &shape,
                                             const BoundingBox &bounding_box,
                                             const Mat_<double> &rotation,
                                             const double scale)
{
    int currnode = 0;
    int bincode = 0;
    while(1) {
        double x1 = tree.nodes_[currnode].feat[0];
        double y1 = tree.nodes_[currnode].feat[1];
        double x2 = tree.nodes_[currnode].feat[2];
        double y2 = tree.nodes_[currnode].feat[3];

        double project_x1 = rotation(0,0) * x1 + rotation(0,1) * y1;
        double project_y1 = rotation(1,0) * x1 + rotation(1,1) * y1;
        project_x1 = scale * project_x1 * bounding_box.width / 2.0;
        project_y1 = scale * project_y1 * bounding_box.height / 2.0;
        int real_x1 = project_x1 + shape(tree.landmarkID_,0);
        int real_y1 = project_y1 + shape(tree.landmarkID_,1);
        real_x1 = max(0.0, min((double)real_x1, image.cols - 1.0));
        real_y1 = max(0.0, min((double)real_y1, image.rows - 1.0));

        double project_x2 = rotation(0,0) * x2 + rotation(0,1) * y2;
        double project_y2 = rotation(1,0) * x2 + rotation(1,1) * y2;
        project_x2 = scale * project_x2 * bounding_box.width / 2.0;
        project_y2 = scale * project_y2 * bounding_box.height / 2.0;
        int real_x2 = project_x2 + shape(tree.landmarkID_,0);
        int real_y2 = project_y2 + shape(tree.landmarkID_,1);
        real_x2 = max(0.0,min((double)real_x2,image.cols-1.0));
        real_y2 = max(0.0,min((double)real_y2,image.rows-1.0));

        double pdf = ((int)(image(real_y1, real_x1)) - (int)(image(real_y2, real_x2)));
        if (pdf < tree.nodes_[currnode].thresh)
            currnode = tree.nodes_[currnode].cnodes[0];
        else
            currnode = tree.nodes_[currnode].cnodes[1];

        if (tree.nodes_[currnode].isleafnode == 1) {
            bincode = 1;
            for (vector<int>::const_iterator citer = tree.id_leafnodes_.begin(); citer != tree.id_leafnodes_.end(); citer++) {
                if (*citer == currnode)
                    return bincode;
                bincode++;
            }
            return bincode;
        }
    }
    return bincode;
};

void Detection::LBFRegressor::GlobalRegression(struct feature_node **binfeatures,
                                               const vector<Mat_<double> > &shapes_residual,
                                               vector<Mat_<double> > &current_shapes,
                                               const vector<BoundingBox> &bounding_boxs,
                                               const Mat_<double> &mean_shape,
                                               vector<struct model*> &models,
                                               int num_feature,
                                               int num_train_sample
                                              )
{
    // shapes_residual: n*(l*2)
    // construct the problem(expect y)
    struct problem *prob = new struct problem;
    prob->l = num_train_sample;
    prob->n = num_feature;
    prob->x = binfeatures;
    prob->bias = -1;

    // construct the parameter
    struct parameter *param = new struct parameter;
    param->solver_type = L2R_L2LOSS_SVR_DUAL;
    // param-> solver_type = L2R_L2LOSS_SVR;
    param->C = 1.0 / num_train_sample;
    param->p = 0;

    // initialize the y
    int num_residual = shapes_residual[0].rows*2;
    double **yy = new double*[num_residual];

    for (int i = 0; i < num_residual; i++)
        yy[i] = new double[num_train_sample];

    for (int i = 0; i < num_train_sample; i++) {
        for (int j = 0; j < num_residual; j++) {
            if (j < num_residual/2)
                yy[j][i] = shapes_residual[i](j, 0);
            else
                yy[j][i] = shapes_residual[i](j - num_residual/2,1);
        }
    }

    //train
    models.clear();
    models.resize(num_residual);
    for (int i = 0; i < num_residual; i++) {
        clock_t t1 = clock();
        qDebug("Training %dth landmark", i);
        prob->y = yy[i];
        check_parameter(prob, param);
        struct model* lbfmodel  = train(prob, param);
        models[i] = lbfmodel;
        double time =double(clock() - t1) / CLOCKS_PER_SEC;
        qDebug("Linear regression of landmark %d: %f", i, time);
    }

    // update the current shape and shapes_residual
    double tmp;
    double scale;
    Mat_<double>rotation;
    Mat_<double> deltashape_bar(num_residual / 2, 2);
    Mat_<double> deltashape_bar1(num_residual / 2, 2);
    for (int i = 0; i < num_train_sample; i++) {
        for (int j = 0; j < num_residual; j++) {
            tmp = predict(models[j], binfeatures[i]);
            if (j < num_residual / 2)
                deltashape_bar(j, 0) = tmp;
            else
                deltashape_bar(j - num_residual / 2, 1) = tmp;
        }
        // transfer or not to be decided
        // now transfer
        SimilarityTransform(ProjectShape(current_shapes[i], bounding_boxs[i]), mean_shape, rotation, scale);
        transpose(rotation, rotation);
        deltashape_bar1 = scale * deltashape_bar * rotation;
        current_shapes[i] = ReProjectShape((ProjectShape(current_shapes[i], bounding_boxs[i]) + deltashape_bar1), bounding_boxs[i]);

        // update shapes_residual
        // shapes_residual[i] = shapes_residual[i] - deltashape_bar;
    }
}

void Detection::LBFRegressor::GlobalPrediction(struct feature_node **binfeatures,
                                               const vector<struct model*> &models,
                                               vector<Mat_<double> > &current_shapes,
                                               const vector<BoundingBox> &bounding_boxs
                                              )
{
    int num_train_sample = (int)current_shapes.size();
    int num_residual = current_shapes[0].rows*2;
    double tmp;
    double scale;
    Mat_<double>rotation;
    Mat_<double> deltashape_bar(num_residual/2,2);
    Mat_<double> deltashape_bar1(num_residual/2,2);
    for (int i = 0; i < num_train_sample; i++) {
        for (int j = 0; j < num_residual; j++) {
            tmp = predict(models[j], binfeatures[i]);
            if (j < num_residual / 2)
                deltashape_bar(j, 0) = tmp;
            else
                deltashape_bar(j - num_residual / 2, 1) = tmp;
        }

        // transfer or not to be decided
        // now transfer
        SimilarityTransform(ProjectShape(current_shapes[i], bounding_boxs[i]), mean_shape_, rotation, scale);
        transpose(rotation, rotation);
        deltashape_bar1 = scale * deltashape_bar * rotation;
        current_shapes[i] = ReProjectShape((ProjectShape(current_shapes[i], bounding_boxs[i]) + deltashape_bar1), bounding_boxs[i]);
    }
}

void Detection::LBFRegressor::Train(const vector<Mat_<uchar> > &images,
                         const vector<Mat_<double> > &ground_truth_shapes,
                         const vector<BoundingBox> &bounding_boxs)
{
    // data augmentation and multiple initialization
    vector<Mat_<uchar> > augmented_images;
    vector<BoundingBox> augmented_bounding_boxs;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vector<Mat_<double> > current_shapes;

    RNG random_generator(getTickCount());
    for (int i = 0; i < (int)images.size(); i++) {
        for (int j = 0; j < params_.initial_num; j++) {
            int index = 0;
            do {
                index = random_generator.uniform(0, (int)images.size());
            } while(index == i);

            // 1. Select ground truth shapes of other images as initial shapes
            augmented_images.push_back(images[i]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
            augmented_bounding_boxs.push_back(bounding_boxs[i]);

            // 2. Project current shape to bounding box of ground truth shapes
            Mat_<double> temp = ProjectShape(ground_truth_shapes[index], bounding_boxs[index]);
            temp = ReProjectShape(temp, bounding_boxs[i]);
            current_shapes.push_back(temp);
        }
    }

    // get mean shape from training shapes(only origin train images)
    mean_shape_ = GetMeanShape(ground_truth_shapes,bounding_boxs);

    // train random forest
    int num_feature = params_.landmark_num * params_.max_numtrees * pow(2, (params_.max_depth - 1));
    int num_train_sample = (int)augmented_images.size();
    for (int stage = 0; stage < params_.max_numstage; stage++) {
        clock_t t = clock();
        GetShapeResidual(augmented_ground_truth_shapes, current_shapes, augmented_bounding_boxs, mean_shape_, shapes_residual_);
        qDebug("Learning random forest for stage %d...", stage);
        forest_[stage].Train(augmented_images, augmented_ground_truth_shapes, current_shapes, augmented_bounding_boxs, mean_shape_, shapes_residual_, params_.max_numfeats[stage], params_.max_radio_radius[stage]);

        qDebug("Deriving bianry codes from learned forest...");
        struct feature_node ** binfeatures ;
        binfeatures = DeriveBinaryFeat(forest_[stage], augmented_images, current_shapes, augmented_bounding_boxs, mean_shape_);

        qDebug("Learning global regression from binary codes...");
        GlobalRegression(binfeatures, shapes_residual_, current_shapes, augmented_bounding_boxs, mean_shape_, models_[stage], num_feature, num_train_sample);
        ReleaseFeatureSpace(binfeatures,(int)augmented_images.size());
        double time = double(clock() - t) / CLOCKS_PER_SEC;
        qDebug("Finished training stage %d. Time: %f secs", stage, time);
    }
}

void Detection::LBFRegressor::ReleaseFeatureSpace(struct feature_node ** binfeatures, int num_train_sample)
{
    for (int i = 0;i < num_train_sample;i++)
        delete[] binfeatures[i];
    delete[] binfeatures;
}

vector<Mat_<double> > Detection::LBFRegressor::Predict(const vector<Mat_<uchar> >& images, const vector<BoundingBox>& bounding_boxs)
{
    vector<Mat_<double> > current_shapes;
    for (int i=0; i < (int)images.size();i++){
        Mat_<double> current_shape = ReProjectShape(mean_shape_, bounding_boxs[i]);
        current_shapes.push_back(current_shape);
    }
    for ( int stage = 0; stage < params_.max_numstage; stage++){
        struct feature_node ** binfeatures ;
        binfeatures = DeriveBinaryFeat(forest_[stage],images,current_shapes,bounding_boxs, mean_shape_);
        GlobalPrediction(binfeatures, models_[stage], current_shapes, bounding_boxs);
    }
    return current_shapes;
}

Mat_<double> Detection::LBFRegressor::Predict(const cv::Mat_<uchar> &image, const BoundingBox &bounding_box)
{
    vector<Mat_<uchar> > images;
    vector<Mat_<double> > current_shapes;
    vector<BoundingBox>  bounding_boxs;

    images.push_back(image);
    bounding_boxs.push_back(bounding_box);
    current_shapes.push_back(ReProjectShape(mean_shape_, bounding_box));

    for ( int stage = 0; stage < params_.max_numstage; stage++){
        struct feature_node ** binfeatures ;
        binfeatures = DeriveBinaryFeat(forest_[stage],images,current_shapes,bounding_boxs, mean_shape_);
        GlobalPrediction(binfeatures, models_[stage], current_shapes, bounding_boxs);
    }
    return current_shapes[0];
}

void Detection::LBFRegressor::saveModel(ofstream &fout, model *model)
{
    int nr_feature = model->nr_feature;
    int n;
    const parameter& param = model->param;

    if (model->bias>=0)
        n=nr_feature+1;
    else
        n=nr_feature;
    int w_size = n;

    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    int nr_w;
    if (model->nr_class == 2 && model->param.solver_type != MCSVM_CS)
        nr_w=1;
    else
        nr_w = model->nr_class;

    fout.write((char*)&param.solver_type, sizeof(int));
    fout.write((char*)&model->nr_class, sizeof(int));

    int haslabel = 0;
    if (model->label)
        haslabel = 1;
    else
        haslabel = 0;

    fout.write((char*)&haslabel, sizeof(int));
    if(model->label)
        fout.write((char*)model->label, sizeof(int)*model->nr_class);

    fout.write((char*)&nr_feature, sizeof(int));
    fout.write((char*)&model->bias, sizeof(double));
    fout.write((char*)model->w, sizeof(double)*w_size*nr_w);

    setlocale(LC_ALL, old_locale);
    free(old_locale);
}

void Detection::LBFRegressor::Save(string path)
{
    qDebug() << "\nSaving model...";
    ofstream fout;
    fout.open(path.c_str());

    // Write parameters
    fout << params_.bagging_overlap << endl;
    fout << params_.max_numtrees << endl;
    fout << params_.max_depth << endl;
    fout << params_.max_numthreshs << endl;
    fout << params_.landmark_num << endl;
    fout << params_.initial_num << endl;
    fout << params_.max_numstage << endl;

    for (int i = 0; i < params_.max_numstage; i++)
        fout << params_.max_radio_radius[i] << " ";
    fout << endl;

    for (int i = 0; i < params_.max_numstage; i++)
        fout << params_.max_numfeats[i] << " ";
    fout << endl;

    // Write model
    for(int i = 0;i < params_.landmark_num; i++)
        fout << mean_shape_(i,0) << " " << mean_shape_(i,1) << " ";
    fout << endl;

    ofstream fout_reg; string modelPath = path + "/model/Regressor.model";
    fout_reg.open(modelPath.c_str(),ios::binary);
    for (int i=0; i < params_.max_numstage; i++ ){
        forest_[i].Write(fout);
        fout << models_[i].size() << endl;
        for (int j = 0; j < (int)models_[i].size(); j++)
            saveModel(fout_reg, models_[i][j]);
    }
    fout_reg.close();

    fout.close();
    qDebug() << "Model saved!\n";
}

struct model *Detection::LBFRegressor::loadModel(ifstream &fin)
{
    int nr_feature;
    int n;
    int nr_class;
    int haslabel;
    struct model *model = (struct model *)malloc(sizeof(struct model));
    parameter& param = model->param;

    model->label = NULL;

    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    fin.read((char*)&param.solver_type, sizeof(int));
    fin.read((char*)&model->nr_class, sizeof(int));
    nr_class = model->nr_class;
    fin.read((char*)&haslabel, sizeof(int));
    if (haslabel)
        fin.read((char*)model->label, sizeof(int)*model->nr_class);

    fin.read((char*)&model->nr_feature, sizeof(int));
    fin.read((char*)&model->bias, sizeof(double));

    nr_feature = model->nr_feature;
    if (model->bias >= 0)
        n = nr_feature + 1;
    else
        n = nr_feature;

    int w_size = n;
    int nr_w;
    if(nr_class==2 && param.solver_type != MCSVM_CS)
        nr_w = 1;
    else
        nr_w = nr_class;

    model->w = (double *)malloc(w_size * nr_w * sizeof(double));

    fin.read((char*)model->w, sizeof(double) * w_size * nr_w);
    setlocale(LC_ALL, old_locale);
    free(old_locale);

    return model;
}

void Detection::LBFRegressor::Load(string path)
{
    qDebug() << "Loading model...";
    ifstream fin;
    fin.open(path.c_str());

    // Read parameters
    fin >> params_.bagging_overlap;
    fin >> params_.max_numtrees;
    fin >> params_.max_depth;
    fin >> params_.max_numthreshs;
    fin >> params_.landmark_num;
    fin >> params_.initial_num;
    fin >> params_.max_numstage;

    for (int i = 0; i< params_.max_numstage; i++)
        fin >> params_.max_radio_radius[i];

    for (int i = 0; i < params_.max_numstage; i++)
        fin >> params_.max_numfeats[i];

    // Read model
    mean_shape_ = Mat::zeros(params_.landmark_num,2,CV_64FC1);
    for(int i = 0;i < params_.landmark_num;i++)
        fin >> mean_shape_(i,0) >> mean_shape_(i,1);

    ifstream fin_reg; string modelPath = path + "/model/Regressor.model";
    fin_reg.open(modelPath.c_str(),ios::binary);
    for (int i=0; i < params_.max_numstage; i++ ){
        forest_[i].Read(fin);
        int num =0;
        fin >> num;
        models_[i].resize(num);
        for (int j = 0; j < num; j++)
            models_[i][j] = loadModel(fin_reg);
    }
    fin_reg.close();

    fin.close();
    qDebug() << "End";
}
