/** @author Adrian Haarbach
 *
 * Example of multiview LM-ICP.
 * variants:
 * - point to point
 * - point to plane
 * minimizers:
 * - ceres with angle axis
 * - ceres with Eigen Quaternion
 * - ceres with SophusSE3
 * options:
 * - recompute normals using PCA
 * - dmax cutoff distance
 * - knn number of clostest frames
 */

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <random>

#include "my_utils.h"
#include "Visualize.h"
#include "common.h"
#include "gflags/gflags.h"
#include "CPUTimer.h"
#include "frame.h"
#include "icp-ceres.h"

using namespace std;

DEFINE_bool(pointToPlane, true, "use point to plane distance metric");
DEFINE_bool(sophusSE3, true, "");
DEFINE_bool(angleAxis, false, "");
DEFINE_double(cutoff, 0.5, "dmax/cutoff distance after which we prune correspondences");
DEFINE_int32(knn, 2, "number of knn nearest neigbhours to build up the graph");
DEFINE_bool(robust, true,
            "robust loss function. Currently uses the SoftL1Loss with scaling parameter set to 1.5*median of point "
            "correspondance distances");

// DEFINE_string(dir,"../samples/dinosaur","dir");
DEFINE_string(dir_clouds, "../samples/Bunny_RealData", "dir_clouds");
DEFINE_string(dir_results, "../samples/results", "dir_results");
DEFINE_string(pose_config, "../samples/init_poses.yaml", "pose_config");
DEFINE_double(voxel_size, 0.1, "voxel size for downsampling");
DEFINE_int32(nr_iter, 20, "number of iterations for optimization");
DEFINE_bool(add_noise, false, "add noise to the initial poses");
DEFINE_double(trans_noise, 0.0, "translation noise in meters");
DEFINE_double(rot_noise, 0.0, "rotation noise in degrees");

namespace ApproachComponents {

static void computePoseNeighbours(vector<std::shared_ptr<Frame> >& frames, int knn) {
  // compute closest points
  MatrixXi adjacencyMatrix = MatrixXi::Zero(frames.size(), frames.size());
  for (int src_id = 0; src_id < frames.size(); src_id++) {
    Frame& srcCloud = *frames[src_id];
    srcCloud.computePoseNeighboursKnn(&frames, src_id, knn);
    for (int j = 0; j < srcCloud.neighbours.size(); j++) {
      adjacencyMatrix(src_id, srcCloud.neighbours[j].neighbourIdx) = 1;
    }
  }
  cout << "graph adjacency matrix == block structure" << endl;
  cout << adjacencyMatrix << endl;
}

static void computeClosestPoints(vector<std::shared_ptr<Frame> >& frames, float cutoff) {
  // compute closest points
  for (int src_id = 0; src_id < frames.size(); src_id++) {
    Frame& srcCloud = *frames[src_id];
    cout << "cloud " << src_id << endl;
    srcCloud.computeClosestPointsToNeighbours(&frames, cutoff);
  }
}
}  // namespace ApproachComponents

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // iterate through data folder, save and sort filenames
  std::vector<std::string> scan_names;
  for (auto& entry : std::filesystem::directory_iterator(FLAGS_dir_clouds)) {
    if (entry.path().extension() != ".ply" && entry.path().extension() != ".pcd") continue;
    scan_names.push_back(entry.path().filename().stem().string());
  }
  std::sort(scan_names.begin(), scan_names.end());

  // load scan clouds and their initial poses
  std::vector<PointCloudTPtr> clouds;
  std::vector<Eigen::Affine3d> poses;
  YAML::Node cloud_poses = YAML::LoadFile(FLAGS_pose_config);
  std::cout << "Loading scan clouds from " << FLAGS_dir_clouds << std::endl;
  for (int i = 0; i < scan_names.size(); i++) {
    std::cout << "Loading cloud " << scan_names[i] << std::endl;
    std::string scan_path = FLAGS_dir_clouds + "/" + scan_names[i];
    PointCloudTPtr cloud(new PointCloudT);
    if (!LoadCloudFromFile(scan_path, cloud)) continue;
    // load the initial transformation from the YAML file
    YAML::Node tf = cloud_poses[scan_names[i]];
    Eigen::Affine3d pose = ParseTransformation(tf);
    // pcl::transformPointCloud(*cloud, *cloud, pose.matrix());
    clouds.push_back(cloud);
    poses.push_back(pose);
    std::cout << "Loaded " << cloud->size() << " points" << std::endl;
  }
  if (clouds.size() != scan_names.size()) {
    std::cout << "Failed to load all clouds, please check input folder" << std::endl;
    return -1;
  }

  // downsample the point clouds
  std::cout << "Downsampling point clouds..." << std::endl;
  for (auto& cloud : clouds) {
    DownsamplePointCloud(cloud, cloud, FLAGS_voxel_size);
    std::cout << "Downsampled to " << cloud->size() << " points" << std::endl;
  }

  // estimate normals for each point cloud
  std::cout << "Estimating normals for point clouds..." << std::endl;
  for (int i = 0; i < clouds.size(); i++) {
    std::cout << "Estimating normals for " << scan_names[i] << std::endl;
    EstimatePointNormals(clouds[i], Eigen::Affine3d::Identity());
  }

  // colorize the point clouds
  std::cout << "Colorizing point clouds..." << std::endl;
  for (int i = 0; i < clouds.size(); i++) {
    std::cout << "Colorizing " << scan_names[i] << std::endl;
    ColorizePointCloud(clouds[i], i);
    // ColorizePointCloudByNormals(clouds[i]);
  }

  vector<std::shared_ptr<Frame> > frames;

  CPUTimer timer = CPUTimer();

  // adapt to the new data structure
  for (int i = 0; i < clouds.size(); i++) {
    shared_ptr<Frame> f(new Frame());
    for (int j = 0; j < clouds[i]->size(); j++) {
      f->pts.push_back(clouds[i]->points[j].getVector3fMap().cast<double>());
      f->nor.push_back(clouds[i]->points[j].getNormalVector3fMap().cast<double>());
    }
    if (FLAGS_add_noise) {
      Eigen::Affine3d noise = GetRandomTransformation(FLAGS_trans_noise, FLAGS_rot_noise);
      f->pose = Eigen::Isometry3d(poses[i].matrix() * noise.matrix());
    } else
      f->pose = Eigen::Isometry3d(poses[i].matrix());
    f->poseGroundTruth = Eigen::Isometry3d(poses[i].matrix());
    frames.push_back(f);
  }

  frames[0]->fixed = true;
  ApproachComponents::computePoseNeighbours(frames, FLAGS_knn);

  for (int i = 0; i < FLAGS_nr_iter; i++) {
    timer.tic();
    ApproachComponents::computeClosestPoints(frames, FLAGS_cutoff);
    timer.toc(std::string("closest pts ") + std::to_string(i));

    timer.tic();
    // robust loss function: currently uses the SoftL1Loss with scaling parameter
    // set to 1.5*median of point correspondance distance
    if (FLAGS_sophusSE3)
      ICP_Ceres::ceresOptimizer_sophusSE3(frames, FLAGS_pointToPlane, FLAGS_robust);
    else if (FLAGS_angleAxis)
      ICP_Ceres::ceresOptimizer_ceresAngleAxis(frames, FLAGS_pointToPlane, FLAGS_robust);
    else
      ICP_Ceres::ceresOptimizer(frames, FLAGS_pointToPlane, FLAGS_robust);

    timer.toc(std::string("global ") + std::to_string(i));
    cout << "round: " << i << endl;
  }

  // Save optimized poses to the given path
  std::filesystem::create_directory(FLAGS_dir_results);
  std::string output_init_poses = FLAGS_dir_results + "/init_poses_mvicp.yaml";
  std::stringstream stream;
  for (int i = 0; i < clouds.size(); ++i) {
    stream << scan_names[i] << ": [" << std::endl;
    stream << Matrix4dToString(poses[i].matrix()) << std::endl;
    stream << "]" << std::endl << std::endl;
  }
  SaveDataStreamToFile(stream, output_init_poses);

  std::string output_opt_poses = FLAGS_dir_results + "/optimized_poses_mvicp.yaml";
  stream.str("");
  for (int i = 0; i < clouds.size(); ++i) {
    stream << scan_names[i] << ": [" << std::endl;
    stream << Matrix4dToString(frames[i]->pose.matrix()) << std::endl;
    stream << "]" << std::endl << std::endl;
  }
  SaveDataStreamToFile(stream, output_opt_poses);
  std::cout << "Optimization completed" << std::endl;

  return 0;
}
