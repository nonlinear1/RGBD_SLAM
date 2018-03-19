#include "local_graph.h"

using namespace std;

namespace daniel_slam{

    void LocalGraph::optimize() {
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*> (graph_.vertex(0));
        vertex->setFixed(true);
        graph_.initializeOptimization();
        graph_.optimize(50);
        for (int i = 0; i < transform_list_.size(); i++) {
            g2o::VertexSE3* v= (g2o::VertexSE3*)(graph_.vertex(i));
            transform_list_[i].translation() = v->estimate().cast<float>().translation();
            transform_list_[i].rotate(v->estimate().cast<float>().rotation());
        }
    }

    void LocalGraph::set_init(Eigen::Affine3f transform) {
        transform_list_[0] = transform;
    }

    void LocalGraph::show() {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rst(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        cloud->clear();
        png2pointcloud(rgb_list_[0], depth_list_[0], cloud, camera_);
        pcl::transformPointCloud(*cloud, *cloud, transform_list_[0].matrix());
        (*rst) += (*cloud);

        cloud->clear();
        png2pointcloud(rgb_list_[seq_-1], depth_list_[seq_-1], cloud, camera_);
        pcl::transformPointCloud(*cloud, *cloud, transform_list_[seq_-1].matrix());
        (*rst) += (*cloud);

        showPointcloud(rst);
    }

    void LocalGraph::add_vertex(Eigen::Affine3f transform) {
        Eigen::Affine3d trans = transform.cast<double>();
        Eigen::Isometry3d pos;
        pos.translation() = trans.translation();
        pos.rotate(trans.rotation());

        g2o::VertexSE3* vertex = new g2o::VertexSE3();
        vertex->setId(seq_++);
        vertex->setEstimate(pos);
        graph_.addVertex(vertex);
    }

    void LocalGraph::change_vertex(Eigen::Affine3f transform) {
        Eigen::Affine3d trans = transform.cast<double>();
        Eigen::Isometry3d pos;
        pos.translation() = trans.translation();
        pos.rotate(trans.rotation());

        g2o::VertexSE3* vertex =(g2o::VertexSE3*) graph_.vertex(seq_-1);
        vertex->setEstimate(pos);
    }

    void LocalGraph::add_edge(Eigen::Affine3f& transform, InformationMatrix& information, int from, int to) {
        Eigen::Affine3d trans = transform.cast<double>();
        Eigen::Isometry3d pos = Eigen::Isometry3d::Identity();
        pos.translation() = trans.translation();
        pos.rotate(trans.rotation());
        Eigen::Matrix<double, 6, 6> inf = information.cast<double>();

        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->vertices()[0] = graph_.vertex(from);
        edge->vertices()[1] = graph_.vertex(to);
        edge->setMeasurement(pos);
        edge->setInformation(inf);

        graph_.addEdge(edge);
    }

    void LocalGraph::output_file(std::string name) {
        graph_.save(name.data());
    }

    bool LocalGraph::add_new_frame(cv::Mat rgb, cv::Mat depth) {
        FrameNode* frame_current = new FrameNode(depth, rgb, 3, depth.cols, depth.rows);
        frame_current->init(camera_);

        if(seq_ == 0) {
            pre_frame_.reset(frame_current);
            key_frame_ = pre_frame_;
            rgb_list_.push_back(rgb);
            depth_list_.push_back(depth);
            add_vertex(transform_list_[0]);
            return true;
        }

        InformationMatrix information;
        Eigen::Affine3f transform_pre, transform_cur, transform_key, transform_cp;

        transform_pre = transform_list_[seq_-1];
        transform_cur = transform_pre;
        transform_key = transform_list_[0];

        float ratio;

        icp_transfrom(
                *pre_frame_,
                *frame_current,
                transform_cur,
                transform_pre,
                information,
                camera_,
                ratio);

        transform_cp = transform_pre.inverse()*transform_cur;
        add_vertex(transform_cur);
        add_edge(transform_cp, information, seq_-2, seq_-1);

        icp_transfrom(
                *key_frame_,
                *frame_current,
                transform_cur,
                transform_key,
                information,
                camera_,
                ratio);

        transform_cp = transform_pre.inverse()*transform_cur;

        change_vertex(transform_cur);
        transform_list_.push_back(transform_cur);
        add_edge(transform_cp, information, 0, seq_-1);

        pre_frame_.reset(frame_current);
        rgb_list_.push_back(rgb);
        depth_list_.push_back(depth);

        cout << "ratio: " << ratio << endl;

        return ratio >= 0.15;

    }
}

