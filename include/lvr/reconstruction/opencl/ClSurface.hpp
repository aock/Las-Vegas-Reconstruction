#ifndef __ClSurface_H
#define __ClSurface_H

#include "lvr/reconstruction/QueryPoint.hpp"
#include "lvr/reconstruction/LBKdTree.hpp"
#include "lvr/geometry/LBPointArray.hpp"
#include "lvr/geometry/ColorVertex.hpp"

#include <boost/shared_array.hpp>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif
#include "lvr/reconstruction/opencl/cl_helper.h"

#define MAX_SOURCE_SIZE (0x100000)

namespace lvr {

typedef boost::shared_array<float> floatArr;
typedef lvr::ColorVertex<float, unsigned char> cVertex ;
typedef lvr::QueryPoint<cVertex> QueryPointC;

class ClSurface {
public:
    ClSurface(floatArr& points, size_t num_points, size_t dim = 3);
    ~ClSurface();

    /**
    * @brief Starts calculation the normals on GPU
    *
    */
    void calculateNormals();

    /**
	 * @brief Get the resulting normals of the normal calculation. After calling "start".
	 * 
	 * @param output_normals 	PointArray as return value
	 */
	void getNormals(floatArr output_normals);

    /**
	 * @brief Set the number of k nearest neighbors
	 *        k-neighborhood
	 *
	 * @param k             The size of the used k-neighborhood
	 *
	 */
	void setKn(int kn);

	/**
	 * @brief Set the number of k nearest neighbors
	 *        k-neighborhood for interpolation
	 *
	 * @param k             The size of the used k-neighborhood
	 *
	 */
	void setKi(int ki);

	/**
	 * @brief Set the number of k nearest neighbors
	 *        k-neighborhood for distance
	 *
	 * @param k             The size of the used k-neighborhood
	 *
	 */
	void setKd(int kd);
	
	/**
	 * @brief Set the viewpoint to orientate the normals
	 *
	 * @param v_x 	Coordinate X axis
	 * @param v_y 	Coordinate Y axis
	 * @param v_z 	Coordinate Z axis
	 *
	 */
	void setFlippoint(float v_x, float v_y, float v_z);
	
	/**
	 * @brief Set Method for normal calculation
	 *
	 * @param method   "PCA","RANSAC"
	 *
	 */
	void setMethod(std::string method);

	/**
	* Reconstuction Mode: 
	* Points stay in gpu until reconstruction is finished
	*/
	void setReconstructionMode(bool mode = true);

	/**
	* TODO:
	*	Implement
	*/
	void distances(std::vector<lvr::QueryPoint<cVertex> >& query_points, float voxel_size);
	
	void freeGPU();

private:

    void init();

	const char *getErrorString(cl_int error);

    void initKdTree();

    void getDeviceInformation();

	void loadKernel();

	void initCl();

	void finalizeCl();

    // V->points and normals
	LBPointArray<float> V;
	LBPointArray<float>* kd_tree_values;
	LBPointArray<unsigned char>* kd_tree_splits;

	LBPointArray<float> Result_Normals;
    boost::shared_ptr<LBKdTree> kd_tree_gen;

	float m_vx, m_vy, m_vz;
	int m_k, m_ki, m_kd;
	
	
	int m_calc_method;
	bool m_reconstruction_mode;

	// Device Information
	cl_platform_id m_platform_id;
	cl_device_id m_device_id;
	cl_uint m_mps;
	cl_uint m_threads_per_block;
	cl_ulong m_device_global_memory;
	cl_int m_ret;
	cl_context m_context;
	cl_command_queue m_command_queue;
	cl_program m_program;
	cl_kernel m_kernel_normal_estimation;
	cl_kernel m_kernel_normal_interpolation;

	size_t m_kernel_source_size;
	char *m_kernel_source_str;

	cl_mem D_V;
	cl_mem D_kd_tree_values;
	cl_mem D_kd_tree_splits;
	cl_mem D_Normals;
	

};

} /* namespace lvr */

#endif // !__ClSurface_H