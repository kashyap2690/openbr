#include <openbr/plugins/openbr_internal.h>
#include <openbr/detection/regressor.h>

using namespace cv;

namespace br
{

class JointCascadeTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(int maxTrees READ get_maxTrees WRITE set_maxTrees RESET reset_maxTrees STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int numLandmarks READ get_numLandmarks WRITE set_numLandmarks RESET reset_numLandmarks STORED false)
    BR_PROPERTY(int, maxTrees, 10)
    BR_PROPERTY(int, maxDepth, 5)
    BR_PROPERTY(int, numLandmarks, 68)

    void train(const TemplateList &data)
    {
        std::vector<Mat_<uchar> > images;
        std::vector<Mat_<double> > shapes;
        std::vector<Detection::BoundingBox> boxes;
        for (int i = 0; i < data.size(); i++) {
            if (data[i].file.points().size() != numLandmarks)
                continue;

            images.push_back((Mat_<uchar> &)data[i].m());

            QList<QPointF> points = data[i].file.points();
            Mat_<double> landmarks(points.size(), 2, CV_64F);
            for (int j = 0; j < points.size(); j++) {
                landmarks(j, 0) = points[j].x();
                landmarks(j, 1) = points[j].y();
            }
            shapes.push_back(landmarks);

            QRectF rect = data[i].file.rects().first(); // Assume 1 face per template

            Detection::BoundingBox box;
            box.start_x = rect.x();
            box.start_y = rect.y();
            box.width   = rect.width();
            box.height  = rect.height();
            box.centroid_x = box.start_x + box.width/2.0;
            box.centroid_y = box.start_y + box.height/2.0;
            boxes.push_back(box);
        }
        qDebug() << images.size();
        Detection::LBFParams params;
        params.bagging_overlap = 0.4;
        params.max_numtrees = maxTrees;
        params.max_depth = maxDepth;
        params.landmark_num = numLandmarks;
        params.initial_num = 5;

        params.max_numstage = 7;

        double m_max_radio_radius[10] = {0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05};
        for (int i = 0; i < 10; i++)
            params.max_radio_radius[i] = m_max_radio_radius[i];

        double m_max_numfeats[8] = {200, 200, 200, 100, 100, 80, 80};
        for (int i = 0; i < 8; i++)
            params.max_numfeats[i] = m_max_numfeats[i];

        params.max_numthreshs = 500;

        Detection::LBFRegressor regressor(params);
        regressor.Train(images, shapes, boxes);
        regressor.Save("/Users/m29803/Desktop/LBF.model");
    }

    void project(const Template &src, Template &dst) const
    {
        (void)src;
        (void)dst;
    }
};

BR_REGISTER(Transform, JointCascadeTransform)

} // namespace br

#include "classification/jointcascade.moc"
