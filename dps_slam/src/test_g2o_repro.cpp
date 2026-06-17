// Minimal, deterministic reproduction of dps_slam's pose-graph optimization, in
// isolation from the live ROS stack. Same solver ("lm_var_cholmod") and the same
// g2o vertex/edge types graph_g2o.cpp uses (VertexSE3, VertexPointXYZ, EdgeSE3,
// EdgeSE3PointXYZ, ParameterSE3Offset, one fixed anchor).
//
// Scenario: the drone flies a straight GT line (yaw = 0). The odometry has a small
// constant yaw bias per step, so the *integrated* odom trajectory curves away
// (accumulated heading drift). Landmarks sit at ground truth and are seen from
// every keyframe with GT-accurate relative measurements. Keyframes are initialized
// at the drifty integrated poses; landmarks at GT.
//
// Question: does a clean batch optimize() pull the keyframes back onto the GT line?
//   - YES  -> g2o + this graph structure descend fine; dps_slam's problem is its
//             incremental per-keyframe strategy, not the optimizer/observability.
//   - NO   -> the structure itself can't recover heading drift here (deeper issue).

#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/edge_se3_pointxyz.h>
#include <g2o/types/slam3d/parameter_se3_offset.h>

#include <string>
#include "dps_slam/optimizer_g2o.hpp"          // the REAL dps_slam driver
#include "dps_slam/object_detection_types.hpp"
#include "dps_slam/graph_node_types.hpp"
#include "utils/conversions.hpp"

G2O_USE_OPTIMIZATION_LIBRARY(cholmod)

static Eigen::Isometry3d se2(double x, double y, double yaw)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(x, y, 1.0);
  T.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return T;
}
static double yaw_of(const Eigen::Isometry3d & T)
{
  return std::atan2(T.linear()(1, 0), T.linear()(0, 0));
}

static const int N = 15;                  // keyframes (0 is the fixed anchor)
static const double step = 0.5;           // m per step along +x (GT)
static const double yaw_bias = 3.0 * M_PI / 180.0;  // odom over-rotates 3 deg/step

struct Scene
{
  std::vector<Eigen::Isometry3d> gt, drift, odom_inc;
  std::vector<Eigen::Vector3d> lm;
};

static Scene make_scene()
{
  Scene s;
  s.gt.resize(N); s.drift.resize(N); s.odom_inc.resize(N);
  for (int i = 0; i < N; ++i) {s.gt[i] = se2(i * step, 0.0, 0.0);}
  s.lm = {{2.0, 3.0, 1.0}, {2.0, -3.0, 1.0}, {5.0, 3.0, 1.0}, {5.0, -3.0, 1.0}};
  s.drift[0] = s.gt[0];
  for (int i = 1; i < N; ++i) {
    Eigen::Isometry3d true_inc = s.gt[i - 1].inverse() * s.gt[i];
    s.odom_inc[i] = true_inc *
      Eigen::Isometry3d(Eigen::AngleAxisd(yaw_bias, Eigen::Vector3d::UnitZ()));
    s.drift[i] = s.drift[i - 1] * s.odom_inc[i];
  }
  return s;
}

static g2o::SparseOptimizer * make_optimizer()
{
  auto * o = new g2o::SparseOptimizer();
  g2o::OptimizationAlgorithmProperty prop;
  o->setAlgorithm(
    g2o::OptimizationAlgorithmFactory::instance()->construct("lm_var_cholmod", prop));
  auto * off = new g2o::ParameterSE3Offset();
  off->setOffset(Eigen::Isometry3d::Identity());
  off->setId(0);
  o->addParameter(off);
  return o;
}

static double last_yaw_err(const std::vector<g2o::VertexSE3 *> & kf, const Scene & s)
{
  return std::fabs(yaw_of(kf[N - 1]->estimate()) - yaw_of(s.gt[N - 1])) * 180.0 / M_PI;
}

// gt_landmark_init: true => landmarks start at GT; false => at the drifty pose of
//   the first keyframe that sees them (what dps_slam effectively does).
// incremental: false => add everything then optimize once (batch);
//   true => add keyframe i + its detections, optimize, repeat (like dps_slam).
static void run(const char * name, bool gt_landmark_init, bool incremental)
{
  Scene s = make_scene();
  const int M = static_cast<int>(s.lm.size());
  auto * o = make_optimizer();

  std::vector<g2o::VertexSE3 *> kf(N, nullptr);
  std::vector<g2o::VertexPointXYZ *> pt(M, nullptr);

  auto add_kf = [&](int i) {
    kf[i] = new g2o::VertexSE3();
    kf[i]->setId(i);
    kf[i]->setEstimate(s.drift[i]);
    if (i == 0) {kf[i]->setFixed(true);}
    o->addVertex(kf[i]);
    if (i > 0) {
      auto * e = new g2o::EdgeSE3();
      e->setVertex(0, kf[i - 1]); e->setVertex(1, kf[i]);
      e->setMeasurement(s.odom_inc[i]);
      e->setInformation(Eigen::MatrixXd::Identity(6, 6) * 10.0);
      o->addEdge(e);
    }
  };
  auto add_det = [&](int i, int j) {
    if (pt[j] == nullptr) {                 // first sighting: create the landmark
      pt[j] = new g2o::VertexPointXYZ();
      pt[j]->setId(N + j);
      // GT, or placed consistently with the (drifty) keyframe that first sees it
      pt[j]->setEstimate(gt_landmark_init ? s.lm[j] : (kf[i]->estimate() * (s.gt[i].inverse() * s.lm[j])));
      o->addVertex(pt[j]);
    }
    auto * e = new g2o::EdgeSE3PointXYZ();
    e->setVertex(0, kf[i]); e->setVertex(1, pt[j]);
    e->setMeasurement(s.gt[i].inverse() * s.lm[j]);   // GT-accurate measurement
    e->setInformation(Eigen::MatrixXd::Identity(3, 3) * 10.0);
    e->setParameterId(0, 0);
    o->addEdge(e);
  };

  if (!incremental) {
    for (int i = 0; i < N; ++i) {add_kf(i);}
    for (int i = 0; i < N; ++i) {for (int j = 0; j < M; ++j) {add_det(i, j);}}
    o->initializeOptimization();
    o->optimize(100);
  } else {
    for (int i = 0; i < N; ++i) {
      add_kf(i);
      for (int j = 0; j < M; ++j) {add_det(i, j);}
      o->initializeOptimization();
      o->optimize(100);   // re-optimize the whole graph every keyframe, like dps_slam
    }
  }

  double ye = last_yaw_err(kf, s);
  double pe = (kf[N - 1]->estimate().translation() - s.gt[N - 1].translation()).norm();
  printf("  [%-26s] last-kf yaw err=%6.2f deg  pos err=%6.3f m  -> %s\n",
    name, ye, pe, ye < 2.0 ? "RECOVERS GT" : "FAILS");
  delete o;
}

// E: drive dps_slam's ACTUAL OptimizerG2O (handleNewOdom / handleNewObjectDetection /
// optimizeGraph with rollback / increment math / map->odom), single-threaded, with the
// same drift scenario. If THIS recovers GT, the optimizer code is correct and the live
// non-convergence is ROS plumbing/concurrency; if it FAILS, the bug is in the driver code.
static void run_real_driver()
{
  Scene s = make_scene();
  const int M = static_cast<int>(s.lm.size());

  OptimizerG2O opt;
  OptimizerG2OParameters p{};
  p.main_graph_odometry_distance_threshold = 0.1;
  p.main_graph_odometry_orientation_threshold = 5.0;
  p.temp_graph_odometry_distance_threshold = 0.1;
  p.temp_graph_odometry_orientation_threshold = 0.1;
  p.main_graph_odometry_distance_threshold_if_detections = 0.1;
  p.map_odom_security_threshold = 5.0;
  p.map_odom_transform_alpha = 1.0;
  p.odometry_is_relative = false;
  p.generate_odom_map_transform = true;
  p.calculate_odom_covariance_ = false;
  p.throttle_detections = false;
  p.use_dual_graph = false;
  p.earth_to_map_transform = Eigen::Isometry3d::Identity();
  opt.setParameters(p);

  Eigen::MatrixXd cov6 = Eigen::MatrixXd::Identity(6, 6) * 0.1;
  Eigen::MatrixXd cov3 = Eigen::MatrixXd::Identity(3, 3) * 0.1;

  for (int i = 0; i < N; ++i) {
    OdometryWithCovariance odom; odom.odometry = s.drift[i]; odom.covariance = cov6;
    opt.handleNewOdom(odom);                       // create keyframe + optimize
    for (int j = 0; j < M; ++j) {                  // then its detections (like the live node)
      OdometryWithCovariance dodom; dodom.odometry = s.drift[i]; dodom.covariance = cov6;
      OdometryInfo dinfo;
      if (!opt.generateDetectionOdometryInfo(dodom, dinfo)) {continue;}
      Eigen::Vector3d measured = s.gt[i].inverse() * s.lm[j];   // true observation
      GateDetection det(std::to_string(j), measured, cov3, false);
      opt.handleNewObjectDetection(&det, dinfo);
    }
  }

  // Compare each optimized keyframe (SE3 node) to the nearest GT pose.
  double yaw_sum = 0, worst = 0; int nkf = 0;
  for (auto * gn : opt.main_graph->getNodes()) {
    auto * se3 = dynamic_cast<GraphNodeSE3 *>(gn);
    if (!se3) {continue;}
    Eigen::Isometry3d est = se3->getPose();
    double best = 1e9;
    for (int i = 0; i < N; ++i) {
      double d = (est.translation() - s.gt[i].translation()).norm();
      if (d < best) {best = d;}
    }
    // nearest-GT yaw error
    double ny = 1e9, nd = 1e9;
    for (int i = 0; i < N; ++i) {
      double d = (est.translation() - s.gt[i].translation()).norm();
      if (d < nd) {nd = d; ny = std::fabs(yaw_of(est) - yaw_of(s.gt[i])) * 180.0 / M_PI;}
    }
    yaw_sum += ny; worst = std::max(worst, ny); nkf++;
  }
  Eigen::Isometry3d corr = opt.getMapOdomTransform() * s.drift[N - 1];  // live corrected pose
  double corr_yaw = std::fabs(yaw_of(corr) - yaw_of(s.gt[N - 1])) * 180.0 / M_PI;
  double corr_pos = (corr.translation() - s.gt[N - 1].translation()).norm();
  printf("  [E: REAL OptimizerG2O driver] keyframes=%d  mean yaw err=%.2f deg  worst=%.2f deg\n",
    nkf, nkf ? yaw_sum / nkf : 0.0, worst);
  printf("       live corrected (map->odom * last_odom) vs GT: pos=%.3f m yaw=%.2f deg  -> %s\n",
    corr_pos, corr_yaw,
    (worst < 3.0) ? "RECOVERS GT (driver code OK -> live bug is plumbing/concurrency)"
                  : "FAILS (bug is in the driver code: optimizeGraph/increment/rollback/map->odom)");
}

// ---- replay of recorded LIVE inputs through the isolated driver ----
#include <fstream>
#include <sstream>
#include <algorithm>

static Eigen::Isometry3d iso(double x, double y, double z,
  double qx, double qy, double qz, double qw)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(x, y, z);
  T.linear() = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
  return T;
}
struct ORow {double t; Eigen::Isometry3d pose;};
struct DRow {double t; std::string id; Eigen::Vector3d pt;};
struct GRow {double t, x, y, yaw;};

static void run_replay()
{
  std::vector<ORow> odom; std::vector<DRow> det; std::vector<GRow> gt;
  {
    std::ifstream f("/tmp/rec_odom.csv"); std::string ln; std::getline(f, ln);
    while (std::getline(f, ln)) {std::stringstream s(ln); std::string c; double v[8]; int i=0;
      while (std::getline(s, c, ',')) {v[i++]=std::stod(c);}
      odom.push_back({v[0], iso(v[1],v[2],v[3],v[4],v[5],v[6],v[7])});}
  }
  {
    std::ifstream f("/tmp/rec_det.csv"); std::string ln; std::getline(f, ln);
    while (std::getline(f, ln)) {std::stringstream s(ln); std::string c; std::vector<std::string> col;
      while (std::getline(s, c, ',')) {col.push_back(c);}
      det.push_back({std::stod(col[0]), col[1],
        Eigen::Vector3d(std::stod(col[3]), std::stod(col[4]), std::stod(col[5]))});}
  }
  {
    std::ifstream f("/tmp/rec_gt.csv"); std::string ln; std::getline(f, ln);
    while (std::getline(f, ln)) {std::stringstream s(ln); std::string c; double v[5]; int i=0;
      while (std::getline(s, c, ',')) {v[i++]=std::stod(c);}
      gt.push_back({v[0], v[1], v[2], v[4]});}
  }
  printf("  replay: %zu odom, %zu det, %zu gt rows\n", odom.size(), det.size(), gt.size());
  if (odom.empty() || det.empty() || gt.empty()) {printf("  (missing recorded data)\n"); return;}

  auto gt_at = [&](double t) {
    size_t b = 0; double bd = 1e18;
    for (size_t i = 0; i < gt.size(); ++i) {double d = std::fabs(gt[i].t - t); if (d < bd) {bd = d; b = i;}}
    return gt[b];
  };
  auto odom_at = [&](double t) {
    size_t b = 0; double bd = 1e18;
    for (size_t i = 0; i < odom.size(); ++i) {double d = std::fabs(odom[i].t - t); if (d < bd) {bd = d; b = i;}}
    return odom[b].pose;
  };

  OptimizerG2O opt;
  OptimizerG2OParameters p{};
  p.main_graph_odometry_distance_threshold = 0.1;
  p.main_graph_odometry_orientation_threshold = 5.0;
  p.temp_graph_odometry_distance_threshold = 0.1;
  p.temp_graph_odometry_orientation_threshold = 0.1;
  p.main_graph_odometry_distance_threshold_if_detections = 0.1;
  p.map_odom_security_threshold = 5.0; p.map_odom_transform_alpha = 1.0;
  p.odometry_is_relative = false; p.generate_odom_map_transform = true;
  p.calculate_odom_covariance_ = false; p.throttle_detections = false;
  p.use_dual_graph = false; p.earth_to_map_transform = Eigen::Isometry3d::Identity();
  opt.setParameters(p);
  Eigen::MatrixXd cov6 = Eigen::MatrixXd::Identity(6, 6) * 0.5;
  Eigen::MatrixXd cov3 = Eigen::MatrixXd::Identity(3, 3) * 0.1;

  // merge odom + detection events in timestamp order, like the live executor
  size_t oi = 0, di = 0;
  std::vector<double> errp, erry; double t0 = odom.front().t, tend = odom.back().t;
  while (oi < odom.size() || di < det.size()) {
    bool do_odom = (di >= det.size()) || (oi < odom.size() && odom[oi].t <= det[di].t);
    if (do_odom) {
      OdometryWithCovariance o; o.odometry = odom[oi].pose; o.covariance = cov6;
      opt.handleNewOdom(o);
      Eigen::Isometry3d corr = opt.getMapOdomTransform() * odom[oi].pose;
      GRow g = gt_at(odom[oi].t);
      if (odom[oi].t - t0 > (tend - t0) * 0.4) {  // second 60%
        errp.push_back((corr.translation().head<2>() - Eigen::Vector2d(g.x, g.y)).norm());
        erry.push_back(std::fabs(std::fabs(yaw_of(corr) - g.yaw) > M_PI ?
          2 * M_PI - std::fabs(yaw_of(corr) - g.yaw) : yaw_of(corr) - g.yaw) * 180.0 / M_PI);
      }
      oi++;
    } else {
      double bt = det[di].t;
      OdometryWithCovariance d; d.odometry = odom_at(bt); d.covariance = cov6;
      OdometryInfo dinfo;
      if (opt.generateDetectionOdometryInfo(d, dinfo)) {
        while (di < det.size() && det[di].t == bt) {
          GateDetection gd(det[di].id, det[di].pt, cov3, false);
          opt.handleNewObjectDetection(&gd, dinfo);
          di++;
        }
      } else {while (di < det.size() && det[di].t == bt) {di++;}}
    }
  }
  std::sort(errp.begin(), errp.end()); std::sort(erry.begin(), erry.end());
  double mp = errp.empty() ? -1 : errp[errp.size()/2];
  double my = erry.empty() ? -1 : erry[erry.size()/2];
  printf("  [REPLAY of live data] corrected vs GT: median pos=%.0f cm  yaw=%.0f deg  -> %s\n",
    mp * 100, my,
    (mp < 0.5) ? "CONVERGES (so the live divergence is the ROS I/O layer, not the data/driver)"
               : "DIVERGES (the recorded data/sequence reproduces it -> debuggable here)");
}

int main()
{
  printf("Minimal g2o reproduction of dps_slam's pose-graph (drift = %.0f deg total):\n",
    yaw_bias * (N - 1) * 180.0 / M_PI);
  run("A: batch, GT lm init",        true,  false);
  run("B: batch, drifty lm init",    false, false);
  run("C: incremental, GT lm init",  true,  true);
  run("D: incremental, drifty lm",   false, true);
  printf("--- now the REAL dps_slam driver on the same scenario ---\n");
  run_real_driver();
  printf("--- REPLAY of recorded LIVE inputs through the isolated driver ---\n");
  run_replay();
  return 0;
}
