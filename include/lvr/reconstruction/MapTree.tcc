namespace lvr {

template<typename VertexT>
void MapTree<VertexT>::insert(VertexT point, unsigned int index)
{
    m_tree[0][point.x] = index;
    m_tree[1][point.y] = index;
    m_tree[2][point.z] = index;
} 

template<typename VertexT>
void MapTree<VertexT>::kNN(VertexT point, std::vector<unsigned int>& neighbours)
{
    std::map<double, double>::iterator low[3];
    for(int i=0; i<3; i++)
    {
        low[i] = m_tree[i].lower_bound(point[i]);
    }
     
} 


} // namespace lvr
