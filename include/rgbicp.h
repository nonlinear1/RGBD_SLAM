#ifndef DANIEL_SLAM_RGBICP_H
#define DANIEL_SLAM_RGBICP_H

#include "internal.h"

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

using namespace cv;

typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPtr;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

namespace daniel_slam{
    void icp_transfrom(
            FrameNode& frame_pre,
            FrameNode& frame_cur,
            Eigen::Affine3f& transform_cur,
            Eigen::Affine3f& transform_pre,
            Eigen::Matrix<float , 6, 6>& information,
            CameraIntr& camera,
            float& ratio
    );

    bool png2pointcloud(Mat rgb, Mat depth, PointCloudPtr cloud, CameraIntr camera);

    void showPointcloud(PointCloudPtr cloud, std::string name = "sample_cloud");

}

#endif //DANIEL_SLAM_RGBICP_H
