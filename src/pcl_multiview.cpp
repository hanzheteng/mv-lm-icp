/** @author Adrian Haarbach
 *
 * Example of multiview LM-ICP.
 * variants:
 * - point to point
 * - point to plane
 * minimizers:
 * - g2o with SO3 vertices and GICP edges
 * - ceres with angle axis
 * - ceres with Eigen Quaternion
 * - ceres with SophusSE3
 * options:
 * - recompute normals using PCA
 * - dmax cutoff distance
 * - knn number of clostest frames
 */

#include <yaml-cpp/yaml.h>
#include "my_utils.h"

#include "Visualize.h"
#include "common.h"
#include "gflags/gflags.h"
#include "CPUTimer.h"
#include "frame.h"

#include "icp-g2o.h"
#include "icp-ceres.h"

using namespace std;

DEFINE_bool(pointToPlane, true, "use point to plane distance metric");
DEFINE_bool(sophusSE3, true, "");
DEFINE_bool(sophusSE3_autodiff, false, "");
DEFINE_bool(angleAxis, false, "");

DEFINE_bool(g2o, false, "use g2o");
DEFINE_double(cutoff, 0.05, "dmax/cutoff distance after which we prune correspondences");  // dmax
DEFINE_int32(knn, 2, "number of knn nearest neigbhours to build up the graph");

// DEFINE_string(dir,"../samples/dinosaur","dir");
DEFINE_string(file_config, "../config/params.yaml", "file_config");

DEFINE_double(sigma, 0.02, "rotation noise variance");
DEFINE_double(sigmat, 0.01, "translation noise variance");

DEFINE_bool(fake, false, "weather to load the first frame repeteadly, useful for testing");
DEFINE_int32(limit, 40, "limit");
DEFINE_int32(step, 2, "step");

DEFINE_bool(recomputeNormals, true, "weather to recompute normals using PCA of 10 neighbours");

DEFINE_bool(robust, true,
            "robust loss function. Currently uses the SoftL1Loss with scaling parameter set to 1.5*median of point "
            "correspondance distances");

static void loadFrames(vector<std::shared_ptr<Frame> >& frames, std::string dir) {
  vector<string> clouds = getAllTextFilesFromFolder(dir, "cloud");
  vector<string> poses = getAllTextFilesFromFolder(dir, "pose");
  vector<string> groundtruth = getAllTextFilesFromFolder(dir, "groundtruth");

  if (clouds.size() != poses.size()) {  // || clouds.size() != groundtruth.size()){
    cout << "unequal size" << endl;
  }

  for (int i = 0; i < clouds.size() && i < FLAGS_limit * FLAGS_step; i += FLAGS_step) {
    shared_ptr<Frame> f(new Frame());
    int j = i;
    if (FLAGS_fake) j = 0;
    loadXYZ(clouds[j], f->pts, f->nor);
    if (FLAGS_recomputeNormals) {
      f->recomputeNormals();
    }

    if (groundtruth.size() == clouds.size()) {
      f->pose = Isometry3d(loadMatrix4d(poses[i]));
      f->poseGroundTruth = Isometry3d(loadMatrix4d(groundtruth[i]));
    } else {
      f->poseGroundTruth = Isometry3d(loadMatrix4d(poses[i]));
      //            double sigma = 0.02;
      if (i == 0) {
        f->pose = f->poseGroundTruth;
      } else {
        f->pose = addNoise(f->poseGroundTruth, FLAGS_sigma, FLAGS_sigmat);
      }
    }

    frames.push_back(f);
  }
}

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
    //        srcCloud.computeClosestPointsToNeighbours(&frames,cutoff);
  }
  cout << "graph adjacency matrix == block structure" << endl;
  cout << adjacencyMatrix << endl;
}

static void computeClosestPoints(vector<std::shared_ptr<Frame> >& frames, float cutoff) {
  // compute closest points
  for (int src_id = 0; src_id < frames.size(); src_id++) {
    Frame& srcCloud = *frames[src_id];
    //        srcCloud.computePoseNeighboursKnn(&frames,src_id,knn);
    cout << "cloud " << src_id << endl;
    srcCloud.computeClosestPointsToNeighbours(&frames, cutoff);
  }
}
}  // namespace ApproachComponents

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // load parameters from config file
  YAML::Node params = YAML::LoadFile(FLAGS_file_config);
  float voxel_size = params["voxel_size"].as<float>();
  std::string cloud_folder(params["dir_clouds"].as<std::string>());
  std::string pose_config(params["pose_config"].as<std::string>());
  std::string results_folder(params["results_folder"].as<std::string>());
  std::filesystem::create_directory(results_folder);

  // iterate through data folder, save and sort filenames
  std::vector<std::string> scan_names;
  for (auto& entry : std::filesystem::directory_iterator(cloud_folder)) {
    if (entry.path().extension() != ".ply" && entry.path().extension() != ".pcd") continue;
    scan_names.push_back(entry.path().filename().stem().string());
  }
  std::sort(scan_names.begin(), scan_names.end());

  // load scan clouds and their initial poses
  std::vector<PointCloudTPtr> clouds;
  std::vector<Eigen::Affine3d> poses;
  YAML::Node cloud_poses = YAML::LoadFile(pose_config);
  std::cout << "Loading BLK scans from " << cloud_folder << std::endl;
  for (int i = 0; i < scan_names.size(); i++) {
    std::cout << "Loading BLK scan " << scan_names[i] << std::endl;
    std::string scan_path = cloud_folder + "/" + scan_names[i];
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
    DownsamplePointCloud(cloud, cloud, voxel_size);
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
    f->pose = Eigen::Isometry3d(poses[i].matrix());
    f->poseGroundTruth = Eigen::Isometry3d(poses[i].matrix());
    frames.push_back(f);
  }

  // loadFrames(frames, cloud_folder);

  frames[0]->fixed = true;
  ApproachComponents::computePoseNeighbours(frames, FLAGS_knn);

  for (int i = 0; i < 20; i++) {
    timer.tic();
    ApproachComponents::computeClosestPoints(frames, FLAGS_cutoff);
    timer.toc(std::string("closest pts ") + std::to_string(i));

    timer.tic();
    if (!FLAGS_g2o) {
      if (FLAGS_sophusSE3)
        ICP_Ceres::ceresOptimizer_sophusSE3(frames, FLAGS_pointToPlane, FLAGS_robust);
      else if (FLAGS_angleAxis)
        ICP_Ceres::ceresOptimizer_ceresAngleAxis(frames, FLAGS_pointToPlane, FLAGS_robust);
      else
        ICP_Ceres::ceresOptimizer(frames, FLAGS_pointToPlane, FLAGS_robust);
    } else {
      ICP_G2O::g2oOptimizer(frames, FLAGS_pointToPlane);
    }

    timer.toc(std::string("global ") + std::to_string(i));
    cout << "round: " << i << endl;
  }

  // Save optimized poses to the given path
  std::string output_init_poses = results_folder + "/mvicp_init_poses.yaml";
  std::stringstream stream;
  for (int i = 0; i < clouds.size(); ++i) {
    stream << scan_names[i] << ": [" << std::endl;
    stream << Matrix4dToString(poses[i].matrix()) << std::endl;
    stream << "]" << std::endl << std::endl;
  }
  SaveDataStreamToFile(stream, output_init_poses);

  std::string output_opt_poses = results_folder + "/mvicp_optimized_poses.yaml";
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
