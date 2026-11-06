// Checks for the infinite-line landmark (VertexLine3D + EdgeSE3Line3D) with no ROS involved.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include <memory>
#include <vector>

#include "g2o/g2o_edge_types.hpp"

namespace
{

std::unique_ptr<g2o::SparseOptimizer> makeOptimizer()
{
  auto optimizer = std::make_unique<g2o::SparseOptimizer>();
  using BlockSolverX = g2o::BlockSolverX;
  auto linear_solver =
    std::make_unique<g2o::LinearSolverEigen<BlockSolverX::PoseMatrixType>>();
  auto block_solver = std::make_unique<BlockSolverX>(std::move(linear_solver));
  optimizer->setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver)));
  optimizer->setVerbose(false);
  return optimizer;
}

Eigen::Isometry3d makePose(double x, double y, double z, double yaw, double roll = 0.0)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() << x, y, z;
  pose.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
  return pose;
}

// A point of the world line at height t along it, and the direction, seen from `pose`.
Eigen::Matrix<double, 6, 1> measureLine(
  const Eigen::Isometry3d & pose, const Eigen::Vector3d & point_w, const Eigen::Vector3d & dir_w,
  double t)
{
  Eigen::Matrix<double, 6, 1> meas;
  meas.head<3>() = pose.inverse() * (point_w + t * dir_w.normalized());
  meas.tail<3>() = (pose.rotation().transpose() * dir_w).normalized();
  return meas;
}

g2o_custom::EdgeSE3Line3D * makeEdge(
  g2o::VertexSE3 * pose, g2o_custom::VertexLine3D * line, const Eigen::Matrix<double, 6, 1> & meas)
{
  auto * edge = new g2o_custom::EdgeSE3Line3D();
  edge->vertices()[0] = pose;
  edge->vertices()[1] = line;
  edge->setMeasurement(meas);
  edge->setInformation(Eigen::Matrix4d::Identity());
  return edge;
}

double lineDistance(const g2o_custom::VertexLine3D & line, const Eigen::Vector3d & x)
{
  const Eigen::Vector3d v = x - line.point();
  return (v - v.dot(line.direction()) * line.direction()).norm();
}

}  // namespace

TEST(VertexLine3D, OplusKeepsDirectionUnitAndPointClosestToOrigin)
{
  g2o_custom::VertexLine3D v;
  v.setEstimate(g2o_custom::VertexLine3D::fromPointDirection({1.0, 2.0, 3.0}, {0.0, 0.0, 1.0}));
  EXPECT_NEAR(v.point().z(), 0.0, 1e-12);

  const double update[4] = {0.05, -0.02, 0.3, -0.1};
  v.oplus(update);

  EXPECT_NEAR(v.direction().norm(), 1.0, 1e-12);
  EXPECT_NEAR(v.point().dot(v.direction()), 0.0, 1e-12);
  EXPECT_GT((v.direction() - Eigen::Vector3d::UnitZ()).norm(), 1e-3);
  EXPECT_GT((v.point().head<2>() - Eigen::Vector2d(1.0, 2.0)).norm(), 0.1);
}

TEST(EdgeSE3Line3D, ZeroErrorAnywhereAlongTheLineAndForEitherSign)
{
  const Eigen::Vector3d point_w(4.0, -2.0, 0.0);
  const Eigen::Vector3d dir_w = Eigen::Vector3d(0.05, -0.02, 1.0).normalized();
  const Eigen::Isometry3d pose = makePose(0.5, -3.0, 1.2, 0.7, 0.05);

  g2o::VertexSE3 v_pose;
  v_pose.setEstimate(pose);
  g2o_custom::VertexLine3D v_line;
  v_line.setEstimate(g2o_custom::VertexLine3D::fromPointDirection(point_w, dir_w));

  for (const double t : {-5.0, 0.0, 1.3, 40.0}) {
    for (const double sign : {1.0, -1.0}) {
      Eigen::Matrix<double, 6, 1> meas = measureLine(pose, point_w, dir_w, t);
      meas.tail<3>() *= sign;
      std::unique_ptr<g2o_custom::EdgeSE3Line3D> edge(makeEdge(&v_pose, &v_line, meas));
      edge->computeError();
      EXPECT_LT(edge->error().norm(), 1e-9) << "t " << t << " sign " << sign;
    }
  }
}

TEST(EdgeSE3Line3D, OffsetAcrossTheLineIsMeasured)
{
  const Eigen::Vector3d point_w(4.0, -2.0, 0.0);
  const Eigen::Vector3d dir_w = Eigen::Vector3d::UnitZ();
  const Eigen::Isometry3d pose = makePose(0.0, 0.0, 1.0, 0.3);

  g2o::VertexSE3 v_pose;
  v_pose.setEstimate(pose);
  g2o_custom::VertexLine3D v_line;
  v_line.setEstimate(g2o_custom::VertexLine3D::fromPointDirection(point_w, dir_w));

  Eigen::Matrix<double, 6, 1> meas = measureLine(pose, point_w + Eigen::Vector3d(0.2, 0.0, 0.0),
      dir_w, 1.0);
  std::unique_ptr<g2o_custom::EdgeSE3Line3D> edge(makeEdge(&v_pose, &v_line, meas));
  edge->computeError();
  EXPECT_NEAR(edge->error().head<2>().norm(), 0.0, 1e-12);
  EXPECT_NEAR(edge->error().tail<2>().norm(), 0.2, 1e-9);
}

TEST(RotateLineCovariance, MatchesRotatedTangentPerturbations)
{
  const Eigen::Vector3d d = Eigen::Vector3d(0.1, 0.2, 1.0).normalized();
  const Eigen::Matrix3d R =
    (Eigen::AngleAxisd(0.8, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY())).toRotationMatrix();
  Eigen::Vector3d b1, b2, c1, c2;
  g2o_custom::directionTangentBasis(d, b1, b2);
  g2o_custom::directionTangentBasis((R * d).normalized(), c1, c2);

  // A small tilt of d along b1 shows up in the rotated basis as the first column of J.
  const double eps = 1e-6;
  const Eigen::Vector3d tilted = R * (d + eps * b1).normalized();
  Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
  cov(0, 0) = 1.0;
  const Eigen::Matrix4d rotated = g2o_custom::rotateLineCovariance(cov, d, R);
  const Eigen::Vector2d j1((tilted - R * d).dot(c1) / eps, (tilted - R * d).dot(c2) / eps);
  EXPECT_NEAR(rotated(0, 0), j1.x() * j1.x(), 1e-5);
  EXPECT_NEAR(rotated(1, 1), j1.y() * j1.y(), 1e-5);
  EXPECT_NEAR(rotated(0, 1), j1.x() * j1.y(), 1e-5);

  Eigen::Matrix4d full = Eigen::Matrix4d::Identity();
  full(0, 1) = full(1, 0) = 0.3;
  full(2, 3) = full(3, 2) = -0.2;
  const Eigen::Matrix4d back = g2o_custom::rotateLineCovariance(
    g2o_custom::rotateLineCovariance(full, d, R), (R * d).normalized(), R.transpose());
  EXPECT_LT((back - full).norm(), 1e-9);
}

TEST(EdgeSE3Line3D, RecoversLineFromFixedPoses)
{
  const Eigen::Vector3d point_gt(4.0, -2.0, 0.0);
  const Eigen::Vector3d dir_gt = Eigen::Vector3d(0.03, -0.02, 1.0).normalized();
  auto optimizer = makeOptimizer();

  const std::vector<Eigen::Isometry3d> poses = {
    makePose(0.0, -3.0, 1.0, 0.0), makePose(2.0, 1.0, 1.2, 1.2), makePose(6.0, 0.0, 0.8, 2.1)};
  int id = 0;
  std::vector<g2o::VertexSE3 *> vertices;
  for (const auto & pose : poses) {
    auto * v = new g2o::VertexSE3();
    v->setId(id++);
    v->setEstimate(pose);
    v->setFixed(true);
    optimizer->addVertex(v);
    vertices.push_back(v);
  }
  auto * line = new g2o_custom::VertexLine3D();
  line->setId(id++);
  line->setEstimate(g2o_custom::VertexLine3D::fromPointDirection(
      point_gt + Eigen::Vector3d(0.4, -0.3, 0.0), Eigen::Vector3d(0.2, 0.1, 1.0)));
  optimizer->addVertex(line);
  const std::vector<double> heights = {0.5, 1.5, 2.7};
  for (size_t k = 0; k < poses.size(); ++k) {
    optimizer->addEdge(makeEdge(vertices[k], line,
        measureLine(poses[k], point_gt, dir_gt, heights[k])));
  }

  optimizer->initializeOptimization();
  optimizer->optimize(30);

  EXPECT_GT(std::abs(line->direction().dot(dir_gt)), 1.0 - 1e-9);
  EXPECT_LT(lineDistance(*line, point_gt), 1e-6);
  EXPECT_LT(lineDistance(*line, point_gt + 3.0 * dir_gt), 1e-6);
}

TEST(EdgeSE3Line3D, TwoCrossingLinesFixAPose)
{
  auto optimizer = makeOptimizer();
  const Eigen::Isometry3d pose_gt = makePose(1.0, -2.0, 1.5, 0.6, 0.04);
  const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lines = {
    {{3.0, 1.0, 0.0}, Eigen::Vector3d::UnitZ()}, {{0.0, 4.0, 2.5}, Eigen::Vector3d::UnitX()}};

  auto * pose = new g2o::VertexSE3();
  pose->setId(0);
  pose->setEstimate(pose_gt * makePose(0.2, -0.15, 0.1, 0.08, -0.03));
  optimizer->addVertex(pose);
  int id = 1;
  for (const auto & l : lines) {
    auto * v = new g2o_custom::VertexLine3D();
    v->setId(id++);
    v->setEstimate(g2o_custom::VertexLine3D::fromPointDirection(l.first, l.second));
    v->setFixed(true);
    optimizer->addVertex(v);
    optimizer->addEdge(makeEdge(pose, v, measureLine(pose_gt, l.first, l.second, 1.0)));
  }

  optimizer->initializeOptimization();
  optimizer->optimize(30);

  const Eigen::Isometry3d delta = pose_gt.inverse() * pose->estimate();
  EXPECT_LT(delta.translation().norm(), 1e-6);
  EXPECT_LT(Eigen::AngleAxisd(delta.rotation()).angle(), 1e-6);
}
