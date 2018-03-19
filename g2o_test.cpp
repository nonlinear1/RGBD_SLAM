#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

int main() {
    typedef g2o::BlockSolver_6_3 BlockSolver;
    typedef g2o::LinearSolverEigen<BlockSolver::PoseMatrixType> LinearSolver;

    g2o::SparseOptimizer mOptimizer;
    mOptimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
            new BlockSolver(
                    new LinearSolver())));
    // Add vertices
    g2o::VertexSE3 *v0 = new g2o::VertexSE3;
    v0->setEstimate(Eigen::Transform<double, 3, 1>(Eigen::Translation<double, 3>(0, 0, 0)));
    v0->setId(0);
    mOptimizer.addVertex(v0);

    g2o::VertexSE3 *v1 = new g2o::VertexSE3;
    v1->setEstimate(Eigen::Transform<double, 3, 1>(Eigen::Translation<double, 3>(0, 0, 0)));
    v1->setId(1);
    mOptimizer.addVertex(v1);

    g2o::VertexSE3 *v2 = new g2o::VertexSE3;
    v2->setEstimate(Eigen::Transform<double, 3, 1>(Eigen::Translation<double, 3>(0, 0, 0)));
    v2->setId(2);
    mOptimizer.addVertex(v2);

    // Add edges
    g2o::EdgeSE3 *e1 = new g2o::EdgeSE3();
    e1->vertices()[0] = mOptimizer.vertex(0);
    e1->vertices()[1] = mOptimizer.vertex(1);
    e1->setMeasurement(Eigen::Isometry3d(Eigen::Translation<double, 3>(1, 0, 0)));
    e1->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    mOptimizer.addEdge(e1);

    g2o::EdgeSE3 *e2 = new g2o::EdgeSE3();
    e2->vertices()[0] = mOptimizer.vertex(1);
    e2->vertices()[1] = mOptimizer.vertex(2);
    e2->setMeasurement(Eigen::Isometry3d(Eigen::Translation<double, 3>(0, 1, 0)));
    e2->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    mOptimizer.addEdge(e2);

    g2o::EdgeSE3 *e3 = new g2o::EdgeSE3();
    e3->vertices()[0] = mOptimizer.vertex(1);
    e3->vertices()[1] = mOptimizer.vertex(2);
    e3->setMeasurement(Eigen::Isometry3d(Eigen::Translation<double, 3>(0.1, 0.8, 0.1)));
    e3->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    mOptimizer.addEdge(e3);

    g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*> (mOptimizer.vertex(0));
    vertex->setFixed(true);
    mOptimizer.initializeOptimization();
    mOptimizer.optimize(50);

    mOptimizer.save("g2o_test.g2o");

    return 0;
}

