#ifndef __MAPTREE_HPP
#define __MAPTREE_HPP

#include <math.h>
#include <boost/shared_ptr.hpp>
#include <map>

namespace lvr
{

/**
 * @brief The LBKdTree class implements a left-balanced array-based index kd-tree.
 *          Left-Balanced: minimum memory
 *          Array-Based: Good for GPU - Usage
 */
template<typename VertexT>
class MapTree {
public:

    void insert(VertexT point, unsigned int index);

    void kNN(VertexT point, std::vector<unsigned int>& neighbours);

private:

    std::map<float, unsigned int > m_tree[3];

};


}  /* namespace lvr */


#include "MapTree.tcc"
#endif // !__MAPTREE_HPP
