#pragma once

#include "difusion/pcl_types.h"

#include "measurement.h"

namespace difusion {

struct PcdMeasurement : public Measurement {};

struct PcdXYZIRTMeasurement : public PcdMeasurement {
  pcl::PointCloud<pcl::PointXYZIRT> pcd;
};

struct PcdFeatXYZIRTMeasurement : public PcdMeasurement {
  pcl::PointCloud<pcl::PointXYZIRT> edge;
  pcl::PointCloud<pcl::PointXYZIRT> plane;
};

} // namespace difusion
