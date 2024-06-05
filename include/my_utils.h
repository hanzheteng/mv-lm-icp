#pragma once

#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <yaml-cpp/yaml.h>

using PointT = pcl::PointXYZRGBNormal;
using PointCloudT = pcl::PointCloud<PointT>;
using PointCloudTPtr = pcl::PointCloud<PointT>::Ptr;

void DownsamplePointCloud(const PointCloudTPtr& cloud_in, const PointCloudTPtr& cloud_out, float resolution) {
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setLeafSize(resolution, resolution, resolution);
  voxel_filter.setInputCloud(cloud_in);
  voxel_filter.filter(*cloud_out);
}

void EstimatePointNormals(PointCloudTPtr& cloud, const Eigen::Affine3d& pose) {
  Eigen::Vector3f viewpoint = pose.translation().cast<float>();
  pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
  pcl::NormalEstimation<PointT, PointT> normal_estimator;
  normal_estimator.setSearchMethod(kdtree);
  normal_estimator.setKSearch(10);
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setViewPoint(viewpoint[0], viewpoint[1], viewpoint[2]);
  normal_estimator.compute(*cloud);
}

void ColorizePointCloudByNormals(PointCloudTPtr cloud) {
  for (size_t i = 0; i < cloud->size(); ++i) {
    // Map from normal vector [-1, 1] to color [0, 255]
    PointT& p = cloud->points[i];
    p.r = static_cast<uint8_t>((p.normal_x + 1.0) * 0.5 * 255);
    p.g = static_cast<uint8_t>((p.normal_y + 1.0) * 0.5 * 255);
    p.b = static_cast<uint8_t>((p.normal_z + 1.0) * 0.5 * 255);
  }
}

void ColorizePointCloud(PointCloudTPtr cloud, int idx) {
  // Define a set of distinguishable RGB colors
  std::vector<Eigen::Vector3i> colors = {
      {255, 0, 0},    // Red
      {0, 255, 0},    // Green
      {0, 0, 255},    // Blue
      {255, 255, 0},  // Yellow
      {255, 0, 255},  // Magenta
      {0, 255, 255},  // Cyan
      {128, 0, 128},  // Purple
      {255, 165, 0},  // Orange
      {0, 128, 0}     // Dark Green
                      // Add more if needed
  };

  // Compute the index into the color array, wrapping around if necessary
  int color_index = idx % colors.size();

  // Get the RGB color for this point cloud
  Eigen::Vector3i rgb = colors[color_index];

  // Apply the color to all the points in the cloud
  for (size_t i = 0; i < cloud->size(); ++i) {
    cloud->points[i].r = rgb[0];
    cloud->points[i].g = rgb[1];
    cloud->points[i].b = rgb[2];
  }
}

bool LoadCloudFromFile(const std::string& path, const PointCloudTPtr& cloud) {
  if (std::filesystem::exists(path + ".ply")) {
    if (pcl::io::loadPLYFile<PointT>(path + ".ply", *cloud) == -1) {
      std::cerr << "Couldn't read file " << path << ".ply";
      return false;
    }
  } else if (std::filesystem::exists(path + ".pcd")) {
    if (pcl::io::loadPCDFile<PointT>(path + ".pcd", *cloud) == -1) {
      std::cerr << "Couldn't read file " << path << ".pcd";
      return false;
    }
  } else {
    std::cerr << "Unsupported file type; couldn't read from path " << path;
    return false;
  }
  return true;
}

Eigen::Affine3d ParseTransformation(const YAML::Node& tf) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  std::string matrixAsString;

  if (tf.IsSequence()) {
    // If the matrix is a sequence, concatenate all elements into a string
    for (const auto& element : tf) {
      matrixAsString += element.as<std::string>() + " ";
    }
  } else if (tf.IsScalar()) {
    // If the matrix is given as a single string
    matrixAsString = tf.as<std::string>();
  } else {
    throw std::runtime_error("Unsupported format for transformation matrix.");
  }

  // Replace potential commas with spaces and then parse the string
  std::replace(matrixAsString.begin(), matrixAsString.end(), ',', ' ');
  std::stringstream ss(matrixAsString);
  double value;
  for (int i = 0; i < 16; ++i) {
    if (!(ss >> value)) {
      throw std::runtime_error("Failed to parse transformation matrix.");
    }
    transform(i / 4, i % 4) = value;
  }

  return Eigen::Affine3d(transform);
}

std::string Matrix4dToString(const Eigen::Matrix4d& matrix) {
  std::ostringstream stream;
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      stream << std::setw(12) << std::fixed << std::setprecision(6) << matrix(i, j);
      if (j < matrix.cols() - 1) stream << " ";
    }
    if (i < matrix.rows() - 1) stream << std::endl;
  }
  return stream.str();
}

void SaveDataStreamToFile(const std::stringstream& stream, const std::string& path) {
  std::ofstream file(path);
  if (!file.is_open()) {
    std::cerr << "Error opening file for writing: " << path << std::endl;
    return;
  }
  file << stream.str();
  file.close();
  std::cout << "Data stream saved to: " << path << std::endl;
}

Eigen::Affine3d GetRandomTransformation(double trans_nosie, double rot_noise) {
  // Prepare random seeds
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_trans(-trans_nosie, trans_nosie);
  std::uniform_real_distribution<> dis_rot(-rot_noise, rot_noise);  // degrees

  // Generate random translation and rotation
  double translation_x = dis_trans(gen);
  double translation_y = dis_trans(gen);
  double rotation_z = dis_rot(gen) * M_PI / 180.0;  // radians

  // Create transformation matrix
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation() << translation_x, translation_y, 0.0;
  transform.rotate(Eigen::AngleAxisd(rotation_z, Eigen::Vector3d::UnitZ()));
  return transform;
}
