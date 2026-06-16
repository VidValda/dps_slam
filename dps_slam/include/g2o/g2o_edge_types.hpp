// Copyright 2024 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


/********************************************************************************************
 *  \file       edge_types.hpp
 *  \brief      Custom g2o edges for SemanticSlam
 *  \authors    David Pérez Saura
 *              Miguel Fernández Cortizas
 *
 *  \copyright  Copyright (c) 2024 Universidad Politécnica de Madrid
 *              All Rights Reserved
 ********************************************************************************/

#ifndef G2O__EDGE_TYPES_HPP_
#define G2O__EDGE_TYPES_HPP_

#include <cmath>
#include <istream>
#include <ostream>

#include <Eigen/Core>
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d_addons/vertex_plane.h"
#include "g2o/types/slam3d_addons/plane3d.h"

// Function to compute the skew-symmetric matrix of a 3D vector
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d & v);

namespace g2o_custom
{

// Unit-vector landmark living on the 2-sphere S^2. The estimate is a 3D unit
// vector (a direction, e.g. a cylinder's axis); the internal DOF is 2 (the
// tangent plane at the current estimate). oplus retracts a 2D tangent update
// back onto the sphere, keeping the estimate normalized.
class VertexUnitVector3 : public g2o::BaseVertex<2, Eigen::Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexUnitVector3() {}

  void setToOriginImpl() override
  {
    _estimate = Eigen::Vector3d(0.0, 0.0, 1.0);
  }

  void oplusImpl(const double * update) override
  {
    const Eigen::Vector3d v = _estimate.normalized();
    // Build an orthonormal basis of the tangent plane at v.
    const Eigen::Vector3d ref =
      (std::abs(v.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
    const Eigen::Vector3d b1 = (ref - ref.dot(v) * v).normalized();
    const Eigen::Vector3d b2 = v.cross(b1);
    const Eigen::Vector3d updated = v + update[0] * b1 + update[1] * b2;
    _estimate = updated.normalized();
  }

  bool read(std::istream & is) override
  {
    for (int i = 0; i < 3; ++i) {is >> _estimate[i];}
    _estimate.normalize();
    return true;
  }

  bool write(std::ostream & os) const override
  {
    for (int i = 0; i < 3; ++i) {os << _estimate[i] << " ";}
    return os.good();
  }
};

class EdgeSE3Point3D : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSE3,
    g2o::VertexPointXYZ>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3Point3D() {}

  static EdgeSE3Point3D * create()
  {
    return new EdgeSE3Point3D();
  }

  void computeError() override
  {
    const g2o::VertexSE3 * se3 = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
    const g2o::VertexPointXYZ * point = static_cast<const g2o::VertexPointXYZ *>(_vertices[1]);

    // Transform the point from the global frame to the SE3 frame
    Eigen::Vector3d transformedPoint = se3->estimate().inverse() * point->estimate();

    // Compute the error as the difference between the transformed point and the measurement
    _error = transformedPoint - _measurement;
    std::cout << "Edge: " << _id << std::endl;
    std::cout << "SE3 translation" << std::endl;
    std::cout << se3->estimate().translation().transpose() << std::endl;
    std::cout << "SE3 rotation" << std::endl;
    std::cout << se3->estimate().rotation().transpose() << std::endl;
    std::cout << "Point translation" << std::endl;
    std::cout << point->estimate().transpose() << std::endl;
    std::cout << "Computed error" << std::endl;
    std::cout << _error.transpose() << std::endl;
    std::cout << "---" << std::endl;
  }

  void linearizeOplus() override
  {
    const g2o::VertexSE3 * se3 = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
    const g2o::VertexPointXYZ * point = static_cast<const g2o::VertexPointXYZ *>(_vertices[1]);

    Eigen::Matrix3d Ri = se3->estimate().rotation().transpose();
    Eigen::Vector3d transformedPoint = se3->estimate().inverse() * point->estimate();

    Eigen::Matrix<double, 3, 6> jacobian_pose;
    jacobian_pose.block<3, 3>(0, 0) = Ri * skewSymmetric(transformedPoint);
    jacobian_pose.block<3, 3>(0, 3) = -Ri;

    Eigen::Matrix<double, 3, 3> jacobian_point = -Ri;

    _jacobianOplusXi = jacobian_pose;
    _jacobianOplusXj = jacobian_point;
  }


  // Read method for deserialization
  bool read(std::istream & is) override
  {
    // Read the measurement (3D vector) from the input stream
    for (int i = 0; i < 3; ++i) {
      is >> _measurement[i];
    }
    return true;
  }

  // Write method for serialization
  bool write(std::ostream & os) const override
  {
    // Write the measurement (3D vector) to the output stream
    for (int i = 0; i < 3; ++i) {
      os << _measurement[i] << " ";
    }
    return os.good();
  }
};

// Binary edge constraining a robot SE3 pose to a plane landmark.
// The error is the 3-DOF ominus (azimuth, elevation, distance) between the plane
// predicted in the robot frame and the measured plane. Jacobians are numeric.
class EdgeSE3Plane3D : public g2o::BaseBinaryEdge<3, g2o::Plane3D, g2o::VertexSE3,
    g2o::VertexPlane>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3Plane3D() {}

  void computeError() override
  {
    const g2o::VertexSE3 * se3 = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
    const g2o::VertexPlane * plane = static_cast<const g2o::VertexPlane *>(_vertices[1]);

    // Predict the plane in the robot frame and compare with the measurement.
    g2o::Plane3D predicted_plane = se3->estimate().inverse() * plane->estimate();
    _error = predicted_plane.ominus(_measurement);
  }

  bool read(std::istream & is) override
  {
    Eigen::Vector4d v;
    for (int i = 0; i < 4; ++i) {is >> v[i];}
    setMeasurement(g2o::Plane3D(v));
    return true;
  }

  bool write(std::ostream & os) const override
  {
    Eigen::Vector4d v = _measurement.toVector();
    for (int i = 0; i < 4; ++i) {os << v[i] << " ";}
    return os.good();
  }
};

// Binary edge constraining a robot SE3 pose to a direction (unit-vector) landmark.
// The measurement is the direction observed in the robot frame; the landmark stores
// the direction in the map frame. The error is the difference between the landmark
// direction predicted in the robot frame and the measured direction. Jacobians are
// left to g2o's numeric differentiation.
class EdgeSE3Direction : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSE3,
    VertexUnitVector3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3Direction() {}

  void computeError() override
  {
    const g2o::VertexSE3 * se3 = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
    const VertexUnitVector3 * dir = static_cast<const VertexUnitVector3 *>(_vertices[1]);

    // Predict the landmark direction in the robot frame (rotation only).
    Eigen::Vector3d predicted = se3->estimate().rotation().transpose() * dir->estimate();
    _error = predicted.normalized() - _measurement.normalized();
  }

  bool read(std::istream & is) override
  {
    for (int i = 0; i < 3; ++i) {is >> _measurement[i];}
    _measurement.normalize();
    return true;
  }

  bool write(std::ostream & os) const override
  {
    for (int i = 0; i < 3; ++i) {os << _measurement[i] << " ";}
    return os.good();
  }
};

}  // namespace g2o_custom

#endif  // G2O__EDGE_TYPES_HPP_
