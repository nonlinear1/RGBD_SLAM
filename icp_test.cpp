#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <sophus/se3.hpp>

#include "device_array.h"
#include "internal.h"
#include "rgbicp.h"

using namespace cv;
using namespace pcl::gpu;
using namespace daniel_slam;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPtr;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

int main() {

    const unsigned short height = 480;
    const unsigned short width = 640;

    CameraIntr camera(525, 525, 319.5, 239.5);

    float ratio;

    Mat rgb[2];
    rgb[0] = imread("rgbd_dataset_freiburg1_desk/rgb/1305031459.391667.png", CV_LOAD_IMAGE_COLOR);
    rgb[1] = imread("rgbd_dataset_freiburg1_desk/rgb/1305031459.427646.png", CV_LOAD_IMAGE_COLOR);


    Mat depth[2];
    depth[0] = imread("rgbd_dataset_freiburg1_desk/depth/1305031459.408831.png", CV_LOAD_IMAGE_ANYDEPTH);
    depth[1] = imread("rgbd_dataset_freiburg1_desk/depth/1305031459.443181.png", CV_LOAD_IMAGE_ANYDEPTH);


    cout << "load..." << endl;

    FrameNode frame1(depth[0], rgb[0], 3, width, height);
    FrameNode frame2(depth[1], rgb[1], 3, width, height);


    frame1.init(camera);
    frame2.init(camera);


    cout << "init..." << endl;


    Eigen::Affine3f t1 = Eigen::Affine3f::Identity();
    Eigen::Affine3f t2 = Eigen::Affine3f::Identity();

    t1.translation() = Eigen::Vector3f(  -1.57869f, -0.0204611f, 0.60757);
    Eigen::Matrix3f r;

    r << 0.774449, -0.35595, 0.522999, 0.453407, 0.888822,-0.0664707, -0.441192, 0.28861, 0.849738;
    t1.rotate(r);

    Eigen::Matrix<float, 6, 6> A;

    t2 = t1;

    cout << "t1:" << endl << t1.translation() << endl << "r1:" << endl << t1.rotation() << endl;

    icp_transfrom(frame1, frame2, t2, t1, A, camera, ratio);
    cout << "t2:" << endl << t2.translation() << endl << "r2:" << endl << t2.rotation() << endl;
    cout << "ratio:" << endl << ratio << endl;

    Eigen::Affine3f pre = t1;
    t2 = t1.inverse()*t2;
    cout << "t2:" << endl << t2.translation() << endl << "r2:" << endl << t2.rotation() << endl;

    t1 = Eigen::Affine3f::Identity();

    icp_transfrom(frame1, frame2, t2, t1, A, camera, ratio);

    t1 = pre;
    t2 = t1*t2;

    cout << "t2:" << endl << t2.translation() << endl << "r2:" << endl << t2.rotation() << endl;
    cout << "ratio:" << endl << ratio << endl;

    PointCloudPtr cloud1(new PointCloud);
    PointCloudPtr cloud2(new PointCloud);

    png2pointcloud(rgb[0], depth[0], cloud1, camera);
    png2pointcloud(rgb[1], depth[1], cloud2, camera);

    pcl::transformPointCloud(*cloud1, *cloud1, t1.matrix());
    pcl::transformPointCloud(*cloud2, *cloud2, t2.matrix());

    (*cloud2) += (*cloud1);

    showPointcloud(cloud2);

    return 0;
}

