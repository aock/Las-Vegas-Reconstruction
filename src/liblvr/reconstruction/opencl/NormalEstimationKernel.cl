unsigned int GetKdTreePosition(__global const float* D_kd_tree_values, const unsigned int num_values,
							__global const unsigned char* D_kd_tree_splits , const unsigned int num_splits,
	 						float x, float y, float z)
{
	unsigned int pos = 0;
	unsigned int current_dim = 0;
	
	while(pos < num_splits)
	{
		current_dim = (unsigned int)(D_kd_tree_splits[pos]);
		if(current_dim == 0)
		{
			if(x <= D_kd_tree_values[pos] )
			{
				pos = pos*2+1;
			} else {
				pos = pos*2+2;
			}
		} else if(current_dim == 1) {
			if(y <= D_kd_tree_values[pos] ){
				pos = pos*2+1;
			}else{
				pos = pos*2+2;
			}
		} else {
			if(z <= D_kd_tree_values[pos] ){
				pos = pos*2+1;
			}else{
				pos = pos*2+2;
			}
		}
	}
	
    return pos;
}

__kernel void NormalEstimationKernel(
		__global const float* D_V, const unsigned int num_points,
		__global const float* D_kd_tree_values, const unsigned int num_values,
		__global const unsigned char* D_kd_tree_splits , const unsigned int num_splits,
		__global float* D_Normals, const unsigned int num_pointnormals,
		const unsigned int k,
		const float flip_x, const float flip_y, const float flip_z
		)
{

	//use vectors for conciseness
    // uint2 globalId = (uint2)(get_global_id(0), get_global_id(1));
    // uint2 localId = (uint2)(get_local_id(0), get_local_id(1));
    // uint2 groupId = (uint2)(get_group_id (0), get_group_id (1));
    // uint2 globalSize = (uint2)(get_global_size(0), get_global_size(1));
    // uint2 locallSize = (uint2)(get_local_size(0), get_local_size(1));
    // uint2 numberOfGrp = (uint2)(get_num_groups (0), get_num_groups (1));

	// printf("globalId = (%u, %u)\n", (&(globalId))[0], (&(globalId))[1]);
	// printf("localId = (%u, %u)\n", (&(localId))[0], (&(localId))[1]);
	// printf("groupId = (%u, %u)\n", (&(groupId))[0], (&(groupId))[1]);
	// printf("globalSize = (%u, %u)\n", (&(globalSize))[0], (&(globalSize))[1]);
	// printf("locallSize = (%u, %u)\n", (&(locallSize))[0], (&(locallSize))[1]);
	// printf("numberOfGrp = (%u, %u)\n", (&(numberOfGrp))[0], (&(numberOfGrp))[1]);

	unsigned int loc_id = get_local_id(0);
	unsigned int loc_size = get_local_size(0);
	unsigned int glob_id = get_global_id(0);
	unsigned int glob_size = get_global_size(0);
	unsigned int group_id = get_group_id(0);
	unsigned int group_size = get_num_groups(0);
    //Read from global and store to local
    //temp[localId.x + localId.y * localSize.x] = in[globalId.x + globalId.y * globalSize.x];
    //Sync
    //barrier(CLK_LOCAL_MEM_FENCE);   

	unsigned int tid = glob_id;
	const unsigned int offset = glob_size;
	
	for(;tid < num_points; tid += offset)
	{
		// printf("num_points: %u\n", num_points);
		//printf("%u = %u + %u * %u\n", tid, glob_id, glob_size, group_id);
		//printf("tid = %u, offset = %u\n",tid, offset);
		
		unsigned int pos = GetKdTreePosition(D_kd_tree_values, num_values,
											 D_kd_tree_splits, num_splits,
											D_V[tid * 3], D_V[tid * 3 + 1], D_V[tid * 3 +2] );
		
		// instant leaf!
		unsigned int vertex_index = (unsigned int)(D_kd_tree_values[pos]+ 0.5);

		
		if(vertex_index < num_points)
		{
				
			float vertex_x = D_V[ vertex_index * 3 + 0 ];
			float vertex_y = D_V[ vertex_index * 3 + 1 ];
			float vertex_z = D_V[ vertex_index * 3 + 2 ];

			unsigned int nearest_index;

			int start = pos-(k/2);
			int end = pos+((k+1)/2);

			int correct = 0;

			if(start < num_splits)
			{
				correct = num_splits - start;
			}else if(end > num_values)
			{
				correct = num_values - end;
			}

			start += correct;
			end += correct;

			// start and end defined

			float result_x = 0.0;
			float result_y = 0.0;
			float result_z = 0.0;
			
			// PCA STUFF INIT

			//x
			float xx = 0.0;
			float xy = 0.0;
			float xz = 0.0;
			
			//y
			float yy = 0.0;
			float yz = 0.0;
			
			//z
			float zz = 0.0;

			for(unsigned int i = start; i < end && i<num_values; i++ )
			{
				if(i != pos)
				{
					nearest_index = (unsigned int)(D_kd_tree_values[i]+ 0.5);

					if(nearest_index < num_points)
					{
						//vector from query point to nearest neighbor
						float rx = D_V[ nearest_index * 3 + 0 ] - vertex_x;
						float ry = D_V[ nearest_index * 3 + 1 ] - vertex_y;
						float rz = D_V[ nearest_index * 3 + 2 ] - vertex_z;
						
						// instant PCA!
						xx += rx * rx;
						xy += rx * ry;
						xz += rx * rz;
						yy += ry * ry;
						yz += ry * rz;
						zz += rz * rz;

					}
					
				}
			}


			//determinante? 
			float det_x = yy * zz - yz * yz;
			float det_y = xx * zz - xz * xz;
			float det_z = xx * yy - xy * xy;
			
			float dir_x;
			float dir_y;
			float dir_z;
			// det X biggest
			if( det_x >= det_y && det_x >= det_z){
				
				if(det_x <= 0.0){
					//not a plane
				}
				
				dir_x = 1.0;
				dir_y = (xz * yz - xy * zz) / det_x;
				dir_z = (xy * yz - xz * yy) / det_x;
			} //det Y biggest
			else if( det_y >= det_x && det_y >= det_z){
				
				if(det_y <= 0.0){
					// not a plane
				}
				
				dir_x = (yz * xz - xy * zz) / det_y;
				dir_y = 1.0;
				dir_z = (xy * xz - yz * xx) / det_y;
			} // det Z biggest
			else{
				if(det_z <= 0.0){
					// not a plane
				}
				
				dir_x = (yz * xy - xz * yy ) / det_z;
				dir_y = (xz * xy - yz * xx ) / det_z;
				dir_z = 1.0;
			}
			
			float invnorm = 1/sqrt( dir_x * dir_x + dir_y * dir_y + dir_z * dir_z );
			
			result_x = dir_x * invnorm;
			result_y = dir_y * invnorm;
			result_z = dir_z * invnorm;



			// FLIP NORMALS
			float x_dir = flip_x - vertex_x;
			float y_dir = flip_y - vertex_y;
			float z_dir = flip_z - vertex_z;
			
			float scalar = x_dir * result_x + y_dir * result_y + z_dir * result_z;
			
			// gegebenfalls < durch > ersetzen
			if(scalar < 0)
			{
				result_x = -result_x;
				result_y = -result_y;
				result_z = -result_z;
			}

			D_Normals[tid * 3 ] = result_x;
			D_Normals[tid * 3 + 1 ] = result_y;
			D_Normals[tid * 3 + 2 ] = result_z;

			
		}

	}
}

