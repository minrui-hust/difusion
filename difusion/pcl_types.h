#define PCL_NO_PRECOMPILE

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace pcl{

struct EIGEN_ALIGN16 PointXYZIRT
{
  float x;
  float y;
  float z;
  std::uint8_t i;  // intensity 0~255
  std::uint8_t r;  // ring id 0~ 255
  std::uint16_t t;  // time offset in 2us
  PCL_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
};

}

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT, 
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (std::uint8_t, i, i)
                                  (std::uint8_t, r, r)
                                  (std::uint16_t, t, t))

