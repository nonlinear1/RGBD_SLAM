#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <sophus/se3.hpp>

#include "device_array.h"
#include "internal.h"
#include "rgbicp.h"
#include "global_graph.h"

using namespace std;
using namespace daniel_slam;

int main() {
    string file_directory = "rgbd_dataset_freiburg1_desk";

    ifstream depth_txt;
    ifstream rgb_txt;
    depth_txt.open(file_directory+"/depth.txt");
    rgb_txt.open(file_directory+"/rgb.txt");

    string depth_line, rgb_line;
    int i = 41;
    while(i--) {
        getline(rgb_txt, rgb_line);
    }
    i = 24;
    while(i--) {
        getline(depth_txt, depth_line);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cur(new pcl::PointCloud<pcl::PointXYZRGB>);

    Eigen::Affine3f cur = Eigen::Affine3f::Identity();
    Eigen::Affine3f pre = Eigen::Affine3f::Identity();

    CameraIntr camera(525, 525, 319.5, 239.5);

    LocalGraph::Ptr local_graph(new LocalGraph);
    GlobalGraph global_graph;

    local_graph->set_camera(camera);
    global_graph.set_camera(camera);

    while (1) {
        FrameNode frame_cur;
        if(!getline(depth_txt, depth_line) || !getline(rgb_txt, rgb_line))
            break;

        string depth_name = file_directory+'/'+depth_line.substr(18);
        string rgb_name = file_directory+'/'+rgb_line.substr(18);

        cout << depth_name << endl << rgb_name << endl;

        cv::Mat rgb, depth;

        rgb = cv::imread(rgb_name, CV_LOAD_IMAGE_COLOR);
        depth = cv::imread(depth_name, CV_LOAD_IMAGE_ANYDEPTH);

        if(!local_graph->add_new_frame(rgb, depth)) {
            local_graph->output_file("local_graph");
            global_graph.add_local_graph(local_graph);
            local_graph.reset(new LocalGraph);
            local_graph->set_camera(camera);
            local_graph->set_init(global_graph.get_next());
            local_graph->add_new_frame(rgb, depth);
        }
    }
    global_graph.output_file();
    global_graph.optimize();
    global_graph.output_file("results_after.g2o");
    global_graph.show();
    return 0;
}

