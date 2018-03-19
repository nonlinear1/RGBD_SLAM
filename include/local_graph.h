#ifndef DANIEL_SLAM_LOCAL_GRAPH_H
#define DANIEL_SLAM_LOCAL_GRAPH_H

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "internal.h"
#include "rgbicp.h"

namespace daniel_slam {
    class LocalGraph {
    public:
        typedef boost::shared_ptr<LocalGraph> Ptr;
        typedef g2o::BlockSolver_6_3 BlockSolver;
        typedef g2o::LinearSolverEigen<BlockSolver::PoseMatrixType> LinearSolver;
        typedef Eigen::Matrix<float , 6, 6> InformationMatrix;

        LocalGraph() {
            graph_.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
                    new BlockSolver(
                            new LinearSolver())));
            seq_ = 0;
            transform_list_.push_back(Eigen::Affine3f::Identity());
        }

        void set_init(Eigen::Affine3f transform);

        void set_camera(CameraIntr camera) { camera_ = camera; }

        bool add_new_frame(cv::Mat rgb, cv::Mat depth);

        g2o::SparseOptimizer &get_graph() { return graph_; }

        void add_vertex(Eigen::Affine3f transform);

        void change_vertex(Eigen::Affine3f transform);

        void add_edge(Eigen::Affine3f& transform, InformationMatrix& information, int from, int to);

        void optimize();

        void output_file(std::string name = "results.g2o");

        void show();

        FrameNode::Ptr get_keyframe() { return key_frame_; }
        FrameNode::Ptr get_last() { return pre_frame_; }
        cv::Mat get_keyframe_rgb() { return rgb_list_[0]; }
        cv::Mat get_keyframe_depth() { return depth_list_[0]; }
        cv::Mat get_last_rgb() { return rgb_list_[seq_-1]; }
        cv::Mat get_last_depth() { return depth_list_[seq_-1]; }
        Eigen::Affine3f get_keyframe_transform(){ return transform_list_[0]; }
        Eigen::Affine3f get_last_transform(){ return transform_list_[seq_-1]; }

    private:
        unsigned long seq_;
        CameraIntr camera_;
        FrameNode::Ptr key_frame_;
        FrameNode::Ptr pre_frame_;
        std::vector<cv::Mat> rgb_list_;
        std::vector<cv::Mat> depth_list_;
        std::vector<Eigen::Affine3f> transform_list_;
        g2o::SparseOptimizer graph_;
    };
}

#endif //DANIEL_SLAM_LOCAL_GRAPH_H
