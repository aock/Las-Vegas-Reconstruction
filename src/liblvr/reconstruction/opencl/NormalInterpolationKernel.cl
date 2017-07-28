float getGaussianFactor(const unsigned int index, const unsigned int middle_i, const unsigned int ki, const float norm)
{
	float val = (float)(index);
	float middle = (float)(middle_i);
	float ki_2 = (float)(ki)/2.0;

	if(val > middle)
	{
		val = val - middle;
	}else{
		val = middle - val;
	}

	if(val > ki_2)
	{
		return 0.0;
	}else{
		float border_val = 0.2;
		float gaussian = 1.0 - pow((float)val/ki_2, (float)2.0) * (1.0-border_val) ;
		return gaussian * norm;
	}
	
}

__kernel void NormalInterpolationKernel(__global float* D_kd_tree_values, const unsigned int num_values,
										__global float* D_kd_tree_splits, const unsigned int num_splits,
										__global float* D_Normals, const unsigned int num_pointnormals,
										const unsigned int ki)

{
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

	for(;tid < num_pointnormals; tid += offset)
	{
		// Interpolate to the Left
		int c = 0;
		unsigned int offset = num_splits;
		
		unsigned int query_index = (unsigned int)(D_kd_tree_values[offset + tid]+ 0.5);
		unsigned int nearest_index;
		
		float gaussian = 5.0; 

		if(query_index < num_pointnormals)
		{
			float n_x = D_Normals[query_index * 3 + 0];
			float n_y = D_Normals[query_index * 3 + 1];
			float n_z = D_Normals[query_index * 3 + 2];

			if(tid > 1)
			{
				for(unsigned int i = tid-1; i > 0 && c < ki/2; i--,c++ )
				{
					nearest_index = (unsigned int)(D_kd_tree_values[i + offset]+ 0.5);

					if(nearest_index < num_pointnormals)
					{

						gaussian = getGaussianFactor(i, tid, ki, 5.0);

						n_x += gaussian * D_Normals[nearest_index * 3 + 0];
						n_y += gaussian * D_Normals[nearest_index * 3 + 1];
						n_z += gaussian * D_Normals[nearest_index * 3 + 2];
					}

				}
			}

			if(tid < num_pointnormals-1)
			{
				for(unsigned int i = tid+1; i < num_pointnormals && c < ki; i++,c++ )
				{
					nearest_index = (unsigned int)(D_kd_tree_values[i + offset]+ 0.5);
					
					if(nearest_index < num_pointnormals)
					{
						gaussian = getGaussianFactor(i, tid, ki, 5.0);

						n_x += gaussian * D_Normals[nearest_index * 3 + 0];
						n_y += gaussian * D_Normals[nearest_index * 3 + 1];
						n_z += gaussian * D_Normals[nearest_index * 3 + 2];
					}
				}
			}

			float norm = sqrt(pow(n_x,2) + pow(n_y,2) + pow(n_z,2));
			n_x = n_x/norm;
			n_y = n_y/norm;
			n_z = n_z/norm;
			D_Normals[query_index * 3 + 0] = n_x;
			D_Normals[query_index * 3 + 1] = n_y;
			D_Normals[query_index * 3 + 2] = n_z;
			
		}
	
	}
}
