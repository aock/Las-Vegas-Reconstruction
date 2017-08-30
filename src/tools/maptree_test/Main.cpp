#include <boost/filesystem.hpp>

#include <lvr/reconstruction/MapTree.hpp>

#include <lvr/geometry/ColorVertex.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/io/IOUtils.hpp>
#include <fstream>
#include <string>
#include <algorithm>
#include "Options.hpp"


using namespace lvr;

typedef ColorVertex<double, unsigned char> cvertex;

void rgb(float minimum, float maximum, float value,
    unsigned int& r, unsigned int& g, unsigned int& b )
{
    float ratio = 2 * (value-minimum) / (maximum - minimum);
    b = int(std::max(float(0.0), 255*(1 - ratio)));
    r = int(std::max(float(0.0), 255*(ratio - 1)));
    g = 255 - b - r;
}

void savePly(std::vector<std::vector<cvertex>* >& points, unsigned int number=0 )
{
    unsigned int num_points = 0;
    for(unsigned int i=0; i< points.size(); i++)
    {
        num_points += points[i]->size();
    }

    std::string filename = std::string("debug") + std::to_string(number) + std::string(".ply");

    std::cout << "Writing " << num_points << " to " << filename << std::endl;

    ofstream myfile;
    
    
    myfile.open(filename.c_str());
    myfile << "ply" << std::endl;
    myfile << "format ascii 1.0" << std::endl;
    myfile << "element vertex " << num_points << std::endl;
    myfile << "property float32 x" << std::endl;
    myfile << "property float32 y" << std::endl;
    myfile << "property float32 z" << std::endl;
    myfile << "property uchar red" << std::endl;                   
    myfile << "property uchar green" << std::endl;
    myfile << "property uchar blue" << std::endl;
    myfile << "end_header" << std::endl;

    for(unsigned int i=0; i<points.size(); i++)
    {
        for(unsigned int j=0; j<points[i]->size(); j++)
        {
            
            unsigned int r,g,b;
            //convertToRgb(0, points.size(), i, r, g, b);
            rgb(0, points.size(), i, r, g, b);

            myfile << (*points[i])[j][0] << " " 
                   << (*points[i])[j][1] << " " 
                   << (*points[i])[j][2] << " "
                   << r << " "
                   << g << " "
                   << b << std::endl;
        }
    }

    myfile.close();

}

void readFile( string filename, maptree_test::Options& opt, PointBufferPtr& buffer)
{
    unsigned int max_leaf_size = opt.maxLeafSize();
    MapTree<cvertex> MTree;

    ModelPtr model = ModelFactory::readModel(filename);
    size_t num_points;
    size_t num_colors;

    floatArr points;
    ucharArr colors;

    if (model && model->m_pointCloud )
    {
        points = model->m_pointCloud->getPointArray(num_points);
        colors = model->m_pointCloud->getPointColorArray(num_colors);
        cout << timestamp << "Read " << num_points << " points from " << filename << endl;
    }
    else
    {
        cout << timestamp << "Warning: No point cloud data found in " << filename << endl;
        return;
    }

    for(size_t i=0; i<num_points; i++)
    {
        cvertex in_point;
        in_point.x = points[i*3+0];
        in_point.y = points[i*3+1];
        in_point.z = points[i*3+2]; 
        in_point.r = colors[i*3+0];
        in_point.g = colors[i*3+1];
        in_point.b = colors[i*3+2];
        MTree.insert(in_point, i);
    }

}

int main(int argc, char** argv){

    maptree_test::Options opt(argc, argv);
    cout << opt << endl;

    boost::filesystem::path inFile(opt.inputFile());

    
    PointBufferPtr buffer(new PointBuffer);
    readFile( opt.inputFile(), opt, buffer);

    ModelPtr out_model(new Model(buffer));
    ModelFactory::saveModel(out_model, opt.outputFile());


}
