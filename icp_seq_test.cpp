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

using namespace cv;
using namespace pcl::gpu;
using namespace daniel_slam;
using namespace std;

int main() {
    string file_directory = "rgbd_dataset_freiburg1_desk";

    ifstream depth_txt;
    ifstream rgb_txt;
    depth_txt.open(file_directory+"/depth.txt");
    rgb_txt.open(file_directory+"/rgb.txt");

    bool first_frame = true;

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

    FrameNode frame_pre;

    CameraIntr camera(525, 525, 319.5, 239.5);
    Eigen::Matrix<float , 6, 6> A;

    i = 10;
    while (i--) {
        FrameNode frame_cur;
        if(!getline(depth_txt, depth_line) || !getline(rgb_txt, rgb_line))
            break;

        string depth_name = file_directory+'/'+depth_line.substr(18);
        string rgb_name = file_directory+'/'+rgb_line.substr(18);

        cout << depth_name << endl << rgb_name << endl;

        cv::Mat rgb, depth;

        rgb = cv::imread(rgb_name, CV_LOAD_IMAGE_COLOR);
        depth = cv::imread(depth_name, CV_LOAD_IMAGE_ANYDEPTH);

        if(first_frame){
            first_frame = false;
            frame_pre.accept_data(depth, rgb, 3, 640, 480);
            frame_pre.init(camera);
            png2pointcloud(rgb, depth, cloud, camera);
            continue;
        }

        frame_cur.accept_data(depth, rgb, 3, 640, 480);
        frame_cur.init(camera);

        float ratio;
        icp_transfrom(frame_pre, frame_cur, cur, pre, A, camera, ratio);
        cloud_cur->clear();
        png2pointcloud(rgb, depth, cloud_cur, camera);
        pcl::transformPointCloud(*cloud_cur, *cloud_cur, cur.matrix());
        (*cloud) += (*cloud_cur);

        frame_pre = frame_cur;
        pre = cur;
    }
    showPointcloud(cloud);
    return 0;
}

