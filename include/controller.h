#ifndef DANIEL_SLAM_CONTROLLER_H
#define DANIEL_SLAM_CONTROLLER_H

#include "local_graph.h"

namespace daniel_slam{
    class Controller{
    public:

        Controller() {}

    private:
        LocalGraph::Ptr local_graph_;
    };
}

#endif //DANIEL_SLAM_CONTROLLER_H
