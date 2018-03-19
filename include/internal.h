#ifndef DANIEL_SLAM_INTERNAL_H
#define DANIEL_SLAM_INTERNAL_H

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include "device_array.h"
#include "cuda_type.cuh"

namespace daniel_slam {

    using pcl::gpu::DeviceArray2D;
    using pcl::gpu::DeviceArray;

    typedef DeviceArray2D<unsigned short> DepthDevice;
    typedef DeviceArray2D<float > ImageDevice;
    typedef DeviceArray2D<float> MapDevice;

    struct CameraIntr {
        float fx;
        float fy;
        float cx;
        float cy;

        CameraIntr() {}

        CameraIntr(float fx, float fy, float cx, float cy) :
                fx(fx), fy(fy), cx(cx), cy(cy) {}

        CameraIntr operator()(int level) const {
            int div = 1 << level;
            return (CameraIntr(fx / div, fy / div, cx / div, cy / div));
        }
    };

    void pyrDown(const DepthDevice &src, DepthDevice &dst);

    void pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float> & dst);

    void createVMap(const CameraIntr &camera, const DepthDevice &depth, MapDevice &vmap);

    void createNmap(const MapDevice &vmap, MapDevice &nmap);

    void computeDerivativeImages(ImageDevice &src, DeviceArray2D<float> &dx, DeviceArray2D<float> &dy);

    void imageBGRToIntensity(DeviceArray2D<unsigned char>& image, DeviceArray2D<float > & dst);

    struct FrameNode {
        typedef boost::shared_ptr<FrameNode> Ptr;

        cv::Mat depth;
        cv::Mat image;
        DeviceArray2D<unsigned char> color_device;
        std::vector<ImageDevice> image_device;
        std::vector<DepthDevice> depth_device;
        std::vector<MapDevice> vertex_map;
        std::vector<MapDevice> normal_map;
        std::vector<MapDevice> dIdx;
        std::vector<MapDevice> dIdy;
        unsigned short NUM;
        unsigned short cols;
        unsigned short rows;

        FrameNode() {}

        FrameNode(cv::Mat depth, cv::Mat image, unsigned short NUM, unsigned short cols, unsigned short rows) :
                depth(depth), image(image), NUM(NUM), cols(cols), rows(rows) {}

        void accept_data(cv::Mat depth_in, cv::Mat image_in, unsigned short NUM_in, unsigned short cols_in, unsigned short rows_in) {
            depth = depth_in;
            image = image_in;
            NUM = NUM_in;
            cols = cols_in;
            rows = rows_in;
        }

        void release() {
            color_device.release();
            for (int i = 0; i < NUM; ++i) {
                image_device[i].release();
                depth_device[i].release();
                vertex_map[i].release();
                normal_map[i].release();
                dIdx[i].release();
                dIdy[i].release();
            }
        }

        void init(CameraIntr camera) {
            image_device.resize(NUM);
            depth_device.resize(NUM);
            vertex_map.resize(NUM);
            normal_map.resize(NUM);
            dIdx.resize(NUM);
            dIdy.resize(NUM);

            color_device.upload((unsigned char*)image.data, image.cols*image.elemSize(), image.rows, image.cols*3);

            imageBGRToIntensity(color_device, image_device[0]);

            depth_device[0].upload(depth.data, depth.cols * depth.elemSize(), depth.rows, depth.cols);

            for (int j = 1; j < NUM; ++j) {
                pyrDown(depth_device[j-1], depth_device[j]);
                pyrDownGaussF(image_device[j-1], image_device[j]);
            }

            for (int i = 0; i < NUM; ++i) {
                createVMap(camera(i), depth_device[i], vertex_map[i]);
                createNmap(vertex_map[i], normal_map[i]);
                computeDerivativeImages(image_device[i], dIdx[i], dIdy[i]);
            }
        }
    };

    void icpStep(const int layer,
                 const mat33 &Rcurr,
                 const float3 &tcurr,
                 const CameraIntr &intr,
                 const FrameNode &frame_pre,
                 const FrameNode &frame_cur,
                 float distThres,
                 float angleThres,
                 DeviceArray<JtJJtrSE3> &sum,
                 DeviceArray<JtJJtrSE3> &out,
                 float *matrixA_host,
                 float *vectorB_host,
                 float *residual_host);

    void rgbStep(const int layer,
                 const mat33 &Rcurr,
                 const float3 &tcurr,
                 const CameraIntr &camera,
                 const FrameNode &frame_pre,
                 const FrameNode &frame_cur,
                 float minScale,
                 float distThres,
                 DeviceArray<JtJJtrSE3> &sum,
                 DeviceArray<JtJJtrSE3> &out,
                 float *matrixA_host,
                 float *vectorB_host,
                 float *residual_host);
}

#endif //DANIEL_SLAM_INTERNAL_H
