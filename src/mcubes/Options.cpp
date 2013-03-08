/*
 * Options.cpp
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#include "Options.h"

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("inputFile,i", value< vector<string> >(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
		("voxelsize,v", value<float>(&m_voxelsize)->default_value(10), "Voxelsize of grid used for reconstruction.")
		("writeFaceNormals", "Writes all interpolated triangle normals together with triangle centroid to a file called 'face_normals.nor'")
		("cluster,c", "Extract planes and write result to 'planes.ply'")
		("optimizeCluster,o", "Shift all triangle vertices of a cluster onto their shared plane")
		("saveNormals,s", "Exports original point cloud data together with normals into a single file called 'points_and_normals.ply'")
		("recalcNormals,r", "Always estimate normals, even if given in .ply file.")
		("threads,t", value<int>(&m_numThreads)->default_value(2), "Number of threads")
		;

	m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);


}

float Options::getVoxelsize() const
{
	return m_variables["voxelsize"].as<float>();
}

int Options::getNumThreads() const
{
	return m_variables["threads"].as<int>();
}

string Options::getOutputFileName() const
{
	return (m_variables["inputFile"].as< vector<string> >())[0];
}

bool Options::printUsage() const
{
  if(!m_variables.count("inputFile"))
    {
      cout << "Error: You must specify an input file." << endl;
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
  
  if(m_variables.count("help"))
    {
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
  return false;
}

bool Options::writeFaceNormals() const
{
	return m_variables.count("writeFaceNormals");
}

bool Options::filenameSet() const
{
	return (m_variables["inputFile"].as< vector<string> >()).size() > 0;
}

bool Options::createClusters() const
{
	return (m_variables.count("cluster"));
}

bool Options::recalcNormals() const
{
	return (m_variables.count("recalcNormals"));
}

bool Options::saveNormals() const
{
	return (m_variables.count("saveNormals"));
}


bool Options::optimizeClusters() const
{
	return createClusters() && m_variables.count("optimizeCluster");
}

Options::~Options() {
	// TODO Auto-generated destructor stub
}