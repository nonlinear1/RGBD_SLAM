#include "rgbicp.h"

#include <sophus/se3.hpp>

namespace daniel_slam {

    bool png2pointcloud(Mat rgb, Mat depth, PointCloudPtr cloud, CameraIntr camera) {
        for (int i = 0; i < rgb.rows; ++i) {
            for (int j = 0; j < rgb.cols; ++j) {
                pcl::PointXYZRGB point;

                unsigned short dep = depth.at<unsigned short>(i, j);

                point.z = static_cast<float>(dep) / 5000;
                point.x = (j - camera.cx) * point.z / camera.fx;
                point.y = (i - camera.cy) * point.z / camera.fy;

                point.r = rgb.at<Vec3b>(i, j)[0];
                point.g = rgb.at<Vec3b>(i, j)[1];
                point.b = rgb.at<Vec3b>(i, j)[2];

                cloud->push_back(point);
            }
        }

        cloud->height = static_cast<unsigned int>(rgb.rows);
        cloud->width = static_cast<unsigned int>(rgb.cols);

        return true;
    }

    void showPointcloud(PointCloudPtr cloud, std::string name) {

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(name));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_cloud(cloud);
        viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb_cloud, name);

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
    }

    void icp_transfrom(
            FrameNode& frame_pre,
            FrameNode& frame_cur,
            Eigen::Affine3f& transform_cur,
            Eigen::Affine3f& transform_pre,
            Eigen::Matrix<float , 6, 6>& information,
            CameraIntr& camera,
            float& ratio){
        const unsigned int NUM = 3;
        const unsigned int iterations[3] = {30, 0, 0};

        Eigen::Affine3f transform = transform_pre.inverse()*transform_cur;
        Eigen::Matrix3f Rcur = transform.rotation();
        Eigen::Vector3f tcur = transform.translation();

        DeviceArray<JtJJtrSE3> sumDataSE3;
        DeviceArray<JtJJtrSE3> outDataSE3;

        sumDataSE3.create(1024);
        outDataSE3.create(1);

        Eigen::Matrix<float, 6, 6> A_icp;
        Eigen::Matrix<float, 6, 1> b_icp;
        float residual_icp[2];

        Eigen::Matrix<float, 6, 6> A_rgb;
        Eigen::Matrix<float, 6, 1> b_rgb;
        float residual_rgb[2];

        Eigen::Matrix<float, 6, 6> A;
        Eigen::Matrix<float, 6, 1> b;

        Eigen::Matrix<float, 6, 1> result;

        Eigen::Affine3f rgbOdom;
        Eigen::Affine3f currentT;

        currentT.setIdentity();
        currentT.rotate(Rcur);
        currentT.translation() = tcur;

        for (int i = NUM - 1; i >= 0; i--) {
            CameraIntr camera_current = camera(i);
            for (int j = 0; j < iterations[i]; ++j) {
                mat33 device_Rcur = Rcur;
                float3 device_tcur = *reinterpret_cast<float3 *>(tcur.data());

                icpStep(i,
                        device_Rcur,
                        device_tcur,
                        camera_current,
                        frame_pre,
                        frame_cur,
                        0.1f,
                        sin(30.f * 3.14159254f / 180.f),
                        sumDataSE3,
                        outDataSE3,
                        A_icp.data(),
                        b_icp.data(),
                        &residual_icp[0]
                );

                rgbStep(i,
                        device_Rcur,
                        device_tcur,
                        camera_current,
                        frame_pre,
                        frame_cur,
                        0.1f,
                        0.1f,
                        sumDataSE3,
                        outDataSE3,
                        A_rgb.data(),
                        b_rgb.data(),
                        &residual_rgb[0]
                );
                A = A_rgb.transpose() + A_icp.transpose();
                b = b_rgb + b_icp;
                result = A.ldlt().solve(b);
                Sophus::SE3<float > se3 = Sophus::SE3<float>::exp(result);
                rgbOdom = se3.matrix();
                currentT = currentT * rgbOdom;
                tcur = currentT.translation();
                Rcur = currentT.rotation();
            }
        }

        information = A;
        transform = Eigen::Affine3f::Identity();
        transform.translation() = tcur;
        transform.rotate(Rcur);
        transform_cur = transform_pre*transform;

        float N = frame_cur.cols * frame_cur.rows;
        ratio = residual_icp[1] / N;
    }
}