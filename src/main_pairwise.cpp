/** @author Adrian Haarbach
 *
 * Comparison of pairwise ICP.
 * variants:
 * - point to point
 * - point to plane
 * minimizers:
 * - closed form solution/1st order approximation
 * - g2o with SO3 vertices and GICP edges
 * - ceres with angle axis
 * - ceres with eigen quaternion
 */

#include "common.h"
#include "gflags/gflags.h"
#include "CPUTimer.h"

#include "icp-closedform.h"
#include "icp-g2o.h"
#include "icp-ceres.h"

using namespace std;

DEFINE_bool(pointToPlane, false, "pointToPlane");
DEFINE_bool(sophusSE3_autodiff, false,
            "weather to use automatic or analytic differentiation on local parameterizaiton");
DEFINE_bool(g2o, false, "Also run with g2o");
DEFINE_bool(ceres, true, "Also run with ceres");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  vector<Vector3d> pts, nor;
  //    loadXYZ("../samples/scene.xyz",pts,nor);
  loadXYZ("../samples/Bunny_RealData/cloudXYZ_0.xyz", pts, nor);

  for (int i = 0; i < 10; ++i) {
    cout << pts[i].transpose() << "\t";
    cout << nor[i].transpose() << endl;
  }

  Matrix3Xd ptsMat = vec2mat(pts);
  Matrix3Xd norMat = vec2mat(nor);

  Quaterniond q = Quaterniond::Identity();
  q = q * AngleAxisd(M_PI_4, Vector3d::UnitX());
  q = q * AngleAxisd(1, Vector3d::UnitY());
  q = q * AngleAxisd(-0.2, Vector3d::UnitZ());

  Vector4d rot(q.x(), q.y(), q.z(), q.w());
  // Vector4d rot(.2,.2,.2,.4);
  // Vector3d tra(0,0,0);//
  Vector3d tra(.01, -0.01, -0.005);  //,0.5,0.01);

  Isometry3d Pclean = Translation3d(tra) * Quaterniond(rot);

  Isometry3d P = addNoise(Pclean, 0.1, 0.1);

  //    for (int i = 0; i < 5; ++i) {

  Matrix3Xd ptsTra = P * ptsMat;
  vector<Vector3d> ptsTraVec = mat2vec(ptsTra);

  Isometry3d Ptest;
  Isometry3d PtestG2O;
  Isometry3d PtestCeres;
  Isometry3d PtestCeres2;
  Isometry3d PtestCeres_Sophus;

  CPUTimer timer;

  if (FLAGS_pointToPlane) {
    Matrix3Xd norTra = P.linear() * norMat;
    vector<Vector3d> norTraVec = mat2vec(norTra);
    timer.tic();
    Ptest = ICP_Closedform::pointToPlane(pts, ptsTraVec, norTraVec);
    timer.toc("closed");
    timer.tic();
    if (FLAGS_g2o) {
      PtestG2O = ICP_G2O::pointToPlane(pts, ptsTraVec, norTraVec);
      timer.toc("g2o");
      timer.tic();
    }
    if (FLAGS_ceres) {
      PtestCeres = ICP_Ceres::pointToPlane_CeresAngleAxis(pts, ptsTraVec, norTraVec);
      timer.toc("ceres CeresAngleAxis");
      timer.tic();
      PtestCeres2 = ICP_Ceres::pointToPlane_EigenQuaternion(pts, ptsTraVec, norTraVec);
      timer.toc("ceres EigenQuaternion");
      timer.tic();
      PtestCeres_Sophus = ICP_Ceres::pointToPlane_SophusSE3(pts, ptsTraVec, norTraVec, FLAGS_sophusSE3_autodiff);
      timer.toc("ceres SophusSE3");
    }
  } else {
    timer.tic();
    Ptest = ICP_Closedform::pointToPoint(pts, ptsTraVec);
    timer.toc("closed");
    timer.tic();
    if (FLAGS_g2o) {
      PtestG2O = ICP_G2O::pointToPoint(pts, ptsTraVec);
      timer.toc("g2o");
      timer.tic();
    }
    if (FLAGS_ceres) {
      PtestCeres = ICP_Ceres::pointToPoint_CeresAngleAxis(pts, ptsTraVec);
      timer.toc("ceres CeresAngleAxis");
      timer.tic();
      PtestCeres2 = ICP_Ceres::pointToPoint_EigenQuaternion(pts, ptsTraVec);
      timer.toc("ceres EigenQuaternion");
      timer.tic();
      PtestCeres_Sophus = ICP_Ceres::pointToPoint_SophusSE3(pts, ptsTraVec, FLAGS_sophusSE3_autodiff);
      timer.toc("ceres SophusSE3");
    }
  }

  timer.printAllTimings();

  cout << endl << "=====  Accurracy ====" << endl;

  cout << "closed form      " << poseDiff(P, Ptest) << endl;

  if (FLAGS_g2o) {
    cout << PtestG2O.matrix() << endl;
    cout << P.matrix() << endl;
    cout << "g2o              " << poseDiff(P, PtestG2O) << endl;
  }

  if (FLAGS_ceres) {
    cout << "ceres CeresAngleAxis" << poseDiff(P, PtestCeres) << endl;

    cout << "ceres EigenQuaternion" << poseDiff(P, PtestCeres2) << endl;

    cout << "ceres SophusSE3    " << poseDiff(P, PtestCeres2) << endl;
  }
}
