#ifndef DANIEL_SLAM_GLOBAL_GRAPH_H
#define DANIEL_SLAM_GLOBAL_GRAPH_H

#include "local_graph.h"

namespace daniel_slam{
    class GlobalGraph{
    public:
        typedef boost::shared_ptr<GlobalGraph> Ptr;
        typedef g2o::BlockSolver_6_3 BlockSolver;
        typedef g2o::LinearSolverEigen<BlockSolver::PoseMatrixType> LinearSolver;

        GlobalGraph() {
            graph_.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
                    new BlockSolver(
                            new LinearSolver())));
        }

        void add_local_graph(LocalGraph::Ptr local_graph);

        void set_camera(CameraIntr camera) { camera_ = camera; }

        void optimize();

        Eigen::Affine3f get_next(){
            return transform_list_.back();
        }

        void add_edge(Eigen::Affine3f& transform, Eigen::Matrix<float , 6, 6>& information, int from, int to);

        void add_keyframe();

        void show();

        void output_file(std::string name = "results.g2o");

        void add_graph(g2o::SparseOptimizer& graph, unsigned int num);
        
    private:
        CameraIntr camera_;
        std::vector<FrameNode::Ptr> key_frame_list_;
        std::vector<unsigned int> key_frame_index_;
        std::vector<cv::Mat> rgb_list_;
        std::vector<cv::Mat> depth_list_;
        std::vector<Eigen::Affine3f> transform_list_;
        g2o::SparseOptimizer graph_;
    };
}

#endif //DANIEL_SLAM_GLOBAL_GRAPH_H
