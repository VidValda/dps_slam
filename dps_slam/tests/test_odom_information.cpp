// The odometry covariance comes in rad^2; a g2o::EdgeSE3 weighted with se3InformationFromCovariance must then give
// a chi2 of (error / std)^2 for a rotation error, as it does for a translation error.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include "g2o/g2o_edge_types.hpp"

namespace
{

double chi2(const Eigen::Isometry3d & error, const Eigen::Matrix<double, 6, 6> & covariance)
{
  g2o::VertexSE3 from, to;
  from.setEstimate(Eigen::Isometry3d::Identity());
  to.setEstimate(error);
  g2o::EdgeSE3 edge;
  edge.vertices()[0] = &from;
  edge.vertices()[1] = &to;
  edge.setMeasurement(Eigen::Isometry3d::Identity());
  edge.setInformation(g2o_custom::se3InformationFromCovariance(covariance));
  edge.computeError();
  return edge.chi2();
}

TEST(OdomInformation, YawErrorOfOneStdGivesChi2One)
{
  const double std = 0.01;
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  covariance(5, 5) = std * std;
  Eigen::Isometry3d error = Eigen::Isometry3d::Identity();
  error.linear() = Eigen::AngleAxisd(std, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  EXPECT_NEAR(chi2(error, covariance), 1.0, 1e-4);
}

TEST(OdomInformation, TranslationIsUnchanged)
{
  const double std = 0.05;
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
  covariance(0, 0) = std * std;
  Eigen::Isometry3d error = Eigen::Isometry3d::Identity();
  error.translation().x() = 2 * std;
  EXPECT_NEAR(chi2(error, covariance), 4.0, 1e-9);
}

}  // namespace
