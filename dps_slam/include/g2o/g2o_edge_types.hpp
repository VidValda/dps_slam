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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include <cmath>
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d_addons/vertex_plane.h"
#include "g2o/types/slam3d_addons/plane3d.h"

// Function to compute the skew-symmetric matrix of a 3D vector
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d & v);

namespace g2o_custom
{

// Information of a g2o::EdgeSE3 from a (x, y, z, roll, pitch, yaw) covariance in m^2 and rad^2: the edge's rotation
// error is the quaternion vector part, half the angle.
inline Eigen::Matrix<double, 6, 6> se3InformationFromCovariance(const Eigen::Matrix<double, 6, 6> & _covariance)
{
  Eigen::Matrix<double, 6, 6> covariance = _covariance;
  covariance.bottomRows<3>() *= 0.5;
  covariance.rightCols<3>() *= 0.5;
  return covariance.inverse();
}

// Orthonormal basis (b1, b2) of the tangent plane at unit direction d, so that
// (d, b1, b2) is a right-handed frame. Used both for the minimal 2-DOF update
// of a cylinder axis direction and for projecting direction residuals.
inline void directionTangentBasis(
  const Eigen::Vector3d & d, Eigen::Vector3d & b1, Eigen::Vector3d & b2)
{
  const Eigen::Vector3d ref = (std::abs(d.z()) < 0.9)
    ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  b1 = d.cross(ref).normalized();
  b2 = d.cross(b1).normalized();
}

// Cylinder-axis landmark: a 5-DOF state [anchor(3); direction(3, unit)].
// The anchor is a distinguished point ON the axis (the cylinder TOP in this
// pipeline -- finite cylinders make the along-axis position observable, unlike
// an infinite Line3D), and the direction is the axis unit vector. The internal
// estimate stores 6 numbers but the vertex exposes the minimal 5 DOF: 3 for the
// anchor plus a 2-DOF tangent-plane update that keeps the direction on S^2.
class VertexCylinderAxis : public g2o::BaseVertex<5, Eigen::Matrix<double, 6, 1>>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCylinderAxis() {}

  void setToOriginImpl() override
  {
    _estimate.setZero();
    _estimate[5] = 1.0;  // direction = +Z
  }

  void oplusImpl(const double * update) override
  {
    _estimate[0] += update[0];
    _estimate[1] += update[1];
    _estimate[2] += update[2];
    Eigen::Vector3d d = _estimate.tail<3>().normalized();
    Eigen::Vector3d b1, b2;
    directionTangentBasis(d, b1, b2);
    _estimate.tail<3>() = (d + b1 * update[3] + b2 * update[4]).normalized();
  }

  Eigen::Vector3d anchor() const {return _estimate.head<3>();}
  Eigen::Vector3d direction() const {return _estimate.tail<3>().normalized();}

  bool read(std::istream & is) override
  {
    for (int i = 0; i < 6; ++i) {is >> _estimate[i];}
    return true;
  }

  bool write(std::ostream & os) const override
  {
    for (int i = 0; i < 6; ++i) {os << _estimate[i] << " ";}
    return os.good();
  }
};

// Binary edge constraining a robot SE3 pose to a cylinder-axis landmark.
// Measurement: [anchor(3); direction(3, unit)] observed in the robot frame.
// Error (5-DOF): the world landmark predicted in the robot frame minus the
// measurement -- 3 anchor residuals plus the direction residual projected onto
// the 2-DOF tangent plane of the measured direction. Jacobians are numeric
// (same approach as the plane edge above).
class EdgeSE3CylinderAxis : public g2o::BaseBinaryEdge<5, Eigen::Matrix<double, 6, 1>,
    g2o::VertexSE3, VertexCylinderAxis>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3CylinderAxis() {}

  void computeError() override
  {
    const g2o::VertexSE3 * se3 = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
    const VertexCylinderAxis * cyl = static_cast<const VertexCylinderAxis *>(_vertices[1]);

    const Eigen::Isometry3d T_inv = se3->estimate().inverse();
    const Eigen::Vector3d anchor_robot = T_inv * cyl->anchor();
    const Eigen::Vector3d dir_robot = (T_inv.rotation() * cyl->direction()).normalized();

    const Eigen::Vector3d meas_anchor = _measurement.head<3>();
    const Eigen::Vector3d meas_dir = _measurement.tail<3>().normalized();

    Eigen::Vector3d b1, b2;
    directionTangentBasis(meas_dir, b1, b2);
    const Eigen::Vector3d dir_diff = dir_robot - meas_dir;

    _error.head<3>() = anchor_robot - meas_anchor;
    _error[3] = b1.dot(dir_diff);
    _error[4] = b2.dot(dir_diff);
  }

  bool read(std::istream & is) override
  {
    for (int i = 0; i < 6; ++i) {is >> _measurement[i];}
    return true;
  }

  bool write(std::ostream & os) const override
  {
    for (int i = 0; i < 6; ++i) {os << _measurement[i] << " ";}
    return os.good();
  }
};

// Re-expresses a 4x4 line covariance over [dir_b1, dir_b2, offset_b1, offset_b2], given in
// the tangent basis of `direction`, in the tangent basis of R * direction.
inline Eigen::Matrix4d rotateLineCovariance(
  const Eigen::Matrix4d & cov, const Eigen::Vector3d & direction, const Eigen::Matrix3d & R)
{
  Eigen::Vector3d b1, b2, c1, c2;
  directionTangentBasis(direction.normalized(), b1, b2);
  directionTangentBasis((R * direction).normalized(), c1, c2);
  Eigen::Matrix2d J;
  J << c1.dot(R * b1), c1.dot(R * b2), c2.dot(R * b1), c2.dot(R * b2);
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  A.block<2, 2>(0, 0) = J;
  A.block<2, 2>(2, 2) = J;
  return A * cov * A.transpose();
}

// Infinite 3D line landmark (4-DOF). The estimate stores [p(3); d(3, unit)], p the point of
// the line closest to the origin (p . d = 0). An update tilts d in its tangent plane (2) and
// shifts p across the line (2); nothing moves the line along itself.
class VertexLine3D : public g2o::BaseVertex<4, Eigen::Matrix<double, 6, 1>>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexLine3D() {}

  static Eigen::Matrix<double, 6, 1> fromPointDirection(
    const Eigen::Vector3d & point, const Eigen::Vector3d & direction)
  {
    Eigen::Matrix<double, 6, 1> state;
    const Eigen::Vector3d d = direction.normalized();
    state.head<3>() = point - point.dot(d) * d;
    state.tail<3>() = d;
    return state;
  }

  void setToOriginImpl() override
  {
    _estimate.setZero();
    _estimate[5] = 1.0;
  }

  void oplusImpl(const double * update) override
  {
    Eigen::Vector3d d = direction();
    Eigen::Vector3d b1, b2;
    directionTangentBasis(d, b1, b2);
    const Eigen::Vector3d p = point() + b1 * update[2] + b2 * update[3];
    d = (d + b1 * update[0] + b2 * update[1]).normalized();
    _estimate = fromPointDirection(p, d);
  }

  Eigen::Vector3d point() const {return _estimate.head<3>();}
  Eigen::Vector3d direction() const {return _estimate.tail<3>().normalized();}

  bool read(std::istream & is) override
  {
    for (int i = 0; i < 6; ++i) {is >> _estimate[i];}
    return true;
  }

  bool write(std::ostream & os) const override
  {
    for (int i = 0; i < 6; ++i) {os << _estimate[i] << " ";}
    return os.good();
  }
};

// Binary edge constraining a robot SE3 pose to an infinite line. Measurement: [q(3); m(3)],
// a point on the line and its direction in the robot frame. Error (4-DOF), predicted minus
// measured in the tangent basis of m: the direction difference, then the offset from q to the
// predicted line. A shift of q along the line changes nothing.
class EdgeSE3Line3D : public g2o::BaseBinaryEdge<4, Eigen::Matrix<double, 6, 1>,
    g2o::VertexSE3, VertexLine3D>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3Line3D() {}

  void computeError() override
  {
    const g2o::VertexSE3 * se3 = static_cast<const g2o::VertexSE3 *>(_vertices[0]);
    const VertexLine3D * line = static_cast<const VertexLine3D *>(_vertices[1]);

    const Eigen::Isometry3d T_inv = se3->estimate().inverse();
    const Eigen::Vector3d p_robot = T_inv * line->point();
    const Eigen::Vector3d q = _measurement.head<3>();
    const Eigen::Vector3d m = _measurement.tail<3>().normalized();
    Eigen::Vector3d d_robot = (T_inv.rotation() * line->direction()).normalized();
    if (d_robot.dot(m) < 0.0) {d_robot = -d_robot;}

    Eigen::Vector3d b1, b2;
    directionTangentBasis(m, b1, b2);
    const Eigen::Vector3d v = p_robot - q;
    const Eigen::Vector3d offset = v - v.dot(d_robot) * d_robot;

    _error[0] = b1.dot(d_robot - m);
    _error[1] = b2.dot(d_robot - m);
    _error[2] = b1.dot(offset);
    _error[3] = b2.dot(offset);
  }

  bool read(std::istream & is) override
  {
    for (int i = 0; i < 6; ++i) {is >> _measurement[i];}
    return true;
  }

  bool write(std::ostream & os) const override
  {
    for (int i = 0; i < 6; ++i) {os << _measurement[i] << " ";}
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

// Unary prior keeping a plane vertical: the error is the elevation of its normal (0 for a wall).
// Wall landmarks come from 2D lines, so they are vertical by construction; without this the
// optimizer can tip one over (seen: a wall drawn 11 m up, its plane turned near horizontal).
class EdgePlaneVertical : public g2o::BaseUnaryEdge<1, double, g2o::VertexPlane>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgePlaneVertical() {setMeasurement(0.0);}

  void computeError() override
  {
    const g2o::VertexPlane * plane = static_cast<const g2o::VertexPlane *>(_vertices[0]);
    const Eigen::Vector3d n = plane->estimate().normal();
    _error[0] = std::atan2(n.z(), n.head<2>().norm()) - _measurement;
  }

  bool read(std::istream & is) override {is >> _measurement; return true;}
  bool write(std::ostream & os) const override {os << _measurement << " "; return os.good();}
};

// Unary prior keeping a wall on the building axes: the error is the azimuth of its normal from
// the nearest multiple of 90 deg around the reference yaw held in the measurement.
class EdgePlaneManhattan : public g2o::BaseUnaryEdge<1, double, g2o::VertexPlane>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgePlaneManhattan() {setMeasurement(0.0);}

  static double deviation(const Eigen::Vector3d & _normal, const double _reference_yaw)
  {
    const double delta = std::atan2(_normal.y(), _normal.x()) - _reference_yaw;
    return delta - (M_PI / 2.0) * std::round(delta / (M_PI / 2.0));
  }

  void computeError() override
  {
    const g2o::VertexPlane * plane = static_cast<const g2o::VertexPlane *>(_vertices[0]);
    _error[0] = deviation(plane->estimate().normal(), _measurement);
  }

  bool read(std::istream & is) override {is >> _measurement; return true;}
  bool write(std::ostream & os) const override {os << _measurement << " "; return os.good();}
};

}  // namespace g2o_custom

#endif  // G2O__EDGE_TYPES_HPP_
