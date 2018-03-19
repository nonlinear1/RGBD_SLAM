#include "global_graph.h"

namespace daniel_slam {
    void GlobalGraph::add_local_graph(LocalGraph::Ptr local_graph) {
        local_graph->optimize();
        if (key_frame_index_.size() == 0) {
            key_frame_index_.push_back(0);
            key_frame_list_.push_back(local_graph->get_keyframe());
            key_frame_list_.push_back(local_graph->get_last());
            rgb_list_.push_back(local_graph->get_keyframe_rgb());
            depth_list_.push_back(local_graph->get_keyframe_depth());
            rgb_list_.push_back(local_graph->get_last_rgb());
            depth_list_.push_back(local_graph->get_last_depth());
            transform_list_.push_back(local_graph->get_keyframe_transform());
            transform_list_.push_back(local_graph->get_last_transform());
            add_graph(local_graph->get_graph(), 0);
            return;
        }
        unsigned int MAX = graph_.vertices().size();
        key_frame_list_.push_back(local_graph->get_last());
        rgb_list_.push_back(local_graph->get_last_rgb());
        transform_list_.push_back(local_graph->get_last_transform());
        depth_list_.push_back(local_graph->get_last_depth());
        add_graph(local_graph->get_graph(), MAX);
    }

    void GlobalGraph::add_keyframe() {
        if(key_frame_index_.size() < 3)
            return;

        Eigen::Affine3f det;
        Eigen::Matrix<float , 6, 6> information;

        int keyframe = key_frame_index_.size()-1;
        for (int i = 0; i < key_frame_index_.size()-1; ++i) {
            det = transform_list_[i].inverse()*transform_list_[keyframe];
            float dist = static_cast<float >(det.translation().norm());
            cout <<  dist << endl;
            if(dist < 0.1) {

                cout << "pair: " << key_frame_index_[i] << ", " << key_frame_index_[keyframe] << endl;

                float ratio;
                Eigen::Affine3f transform_pre, transform_cur, transform_cp;

                transform_pre = transform_list_[i];
                transform_cur = transform_list_[keyframe];
                icp_transfrom(
                        *(key_frame_list_[i]),
                        *(key_frame_list_[keyframe]),
                        transform_cur,
                        transform_pre,
                        information,
                        camera_,
                        ratio);

                transform_cp = transform_pre.inverse()*transform_cur;
                add_edge(transform_cp, information, key_frame_index_[i], key_frame_index_[keyframe]);
            }
        }
    }

    void GlobalGraph::add_edge(Eigen::Affine3f& transform, Eigen::Matrix<float , 6, 6>& information, int from, int to) {
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

    void GlobalGraph::add_graph(g2o::SparseOptimizer &graph, unsigned int MAX) {
        unsigned int pre = graph_.vertices().size();
        if(pre == 0) MAX++;
        for (auto v_it = graph.vertices().begin(); v_it != graph.vertices().end(); ++v_it) {
            g2o::VertexSE3* vertex_ref = dynamic_cast<g2o::VertexSE3*> (v_it->second);
            g2o::VertexSE3* vertex = new g2o::VertexSE3();
            if(vertex_ref->id() == 0 && pre != 0) continue;
            vertex->setId(vertex_ref->id()+MAX-1);
            vertex->setEstimate(vertex_ref->estimate());
            graph_.addVertex(vertex);
        }
        for (auto e_it = graph.edges().begin(); e_it != graph.edges().end(); ++e_it) {
            g2o::EdgeSE3* e = (g2o::EdgeSE3*) (*e_it);
            g2o::EdgeSE3* edge = new g2o::EdgeSE3();
            edge->vertices()[0] = graph_.vertex(e->vertices()[0]->id()+MAX-1);
            edge->vertices()[1] = graph_.vertex(e->vertices()[1]->id()+MAX-1);
            edge->setMeasurement(e->measurement());
            edge->setInformation(e->information());
            graph_.addEdge(edge);
        }
        key_frame_index_.push_back(graph_.vertices().size()-1);
        add_keyframe();
    }

    void GlobalGraph::optimize() {
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*> (graph_.vertex(0));
        vertex->setFixed(true);
        graph_.initializeOptimization();
        graph_.optimize(50);
        for (int i = 0; i < key_frame_index_.size(); i++) {
            int index = key_frame_index_[i];
            g2o::VertexSE3* v= (g2o::VertexSE3*)(graph_.vertex(index));
            transform_list_[i].translation() = v->estimate().cast<float>().translation();
            transform_list_[i].rotate(v->estimate().cast<float>().rotation());
            cout << index << endl;
        }
    }

    void GlobalGraph::show() {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rst(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (int i = 0; i < rgb_list_.size(); ++i) {
            cloud->clear();
            png2pointcloud(rgb_list_[i], depth_list_[i], cloud, camera_);
            pcl::transformPointCloud(*cloud, *cloud, transform_list_[i].matrix());
            (*rst) += (*cloud);
        }

        showPointcloud(rst);
    }

    void GlobalGraph::output_file(std::string name) {
        graph_.save(name.data());
    }
}

