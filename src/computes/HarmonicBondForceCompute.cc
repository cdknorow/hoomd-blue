/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "HarmonicBondForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

/*! \file HarmonicBondForceCompute.cc
	\brief Contains code for the HarmonicBondForceCompute class
*/

/*! \param pdata Particle data to compute forces on
	\post Memory is allocated, and forces are zeroed.
*/
HarmonicBondForceCompute::HarmonicBondForceCompute(boost::shared_ptr<ParticleData> pdata) :	ForceCompute(pdata),
	m_K(NULL), m_r_0(NULL)
	{
	// access the bond data for later use
	m_bond_data = m_pdata->getBondData();
	
	// check for some silly errors a user could make 
	if (m_bond_data->getNBondTypes() == 0)
		{
		cout << endl << "***Error! No bond types specified" << endl << endl;
		throw runtime_error("Error initializing HarmonicBondForceCompute");
		}
		
	// allocate the parameters
	m_K = new Scalar[m_bond_data->getNBondTypes()];
	m_r_0 = new Scalar[m_bond_data->getNBondTypes()];
	}
	
HarmonicBondForceCompute::~HarmonicBondForceCompute()
	{
	delete[] m_K;
	delete[] m_r_0;
	}
	
/*! \param type Type of the bond to set parameters for
	\param K Stiffness parameter for the force computation
	\param r_0 Equilibrium length for the force computation
	
	Sets parameters for the potential of a particular bond type
*/
void HarmonicBondForceCompute::setParams(unsigned int type, Scalar K, Scalar r_0)
	{
	// make sure the type is valid
	if (type >= m_bond_data->getNBondTypes())
		{
		cout << endl << "***Error! Invalid bond typee specified" << endl << endl;
		throw runtime_error("Error setting parameters in HarmonicBondForceCompute");
		}
	
	m_K[type] = K;
	m_r_0[type] = r_0;

	// check for some silly errors a user could make 
	if (m_K <= 0)
		cout << "***Warning! K <= 0 specified for harmonic bond" << endl;
	if (m_r_0 <= 0)
		cout << "***Warning! r_0 <= 0 specified for harmonic bond" << endl;
	}

/*! BondForceCompute provides
	- \c harmonic_energy
*/
std::vector< std::string > HarmonicBondForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("harmonic_energy");
	return list;
	}
	
Scalar HarmonicBondForceCompute::getLogValue(const std::string& quantity)
	{
	if (quantity == string("harmonic_energy"))
		{
		return calcEnergySum();
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for BondForceCompute" << endl << endl;
		throw runtime_error("Error getting log value");
		}
	}	
	
/*! Actually perform the force computation
	\param timestep Current time step
 */
void HarmonicBondForceCompute::computeForces(unsigned int timestep)
 	{
	if (m_prof)
		m_prof->push("Bond");

 	assert(m_pdata);
 	// access the particle data arrays
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	// there are enough other checks on the input data: but it doesn't hurt to be safe
	assert(m_fx);
	assert(m_fy);
	assert(m_fz);
	assert(m_pe);
	assert(arrays.x);
	assert(arrays.y);
	assert(arrays.z);

	if (m_prof)
		m_prof->push("Compute");

	// get a local copy of the simulation box too
	const BoxDim& box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);

	// precalculate box lenghts
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	Scalar Lx2 = Lx / Scalar(2.0);
	Scalar Ly2 = Ly / Scalar(2.0);
	Scalar Lz2 = Lz / Scalar(2.0);
	
	// need to start from a zero force
	memset((void*)m_fx, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fy, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fz, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_pe, 0, sizeof(Scalar) * m_pdata->getN());
	
	// for each of the bonds
	const unsigned int size = (unsigned int)m_bond_data->getNumBonds(); 
	for (unsigned int i = 0; i < size; i++)
		{
		// lookup the tag of each of the particles participating in the bond
		const Bond& bond = m_bond_data->getBond(i);
		
		assert(bond.a < m_pdata->getN());
		assert(bond.b < m_pdata->getN());
				
		// transform a and b into indicies into the particle data arrays
		unsigned int idx_a = arrays.rtag[bond.a];
		unsigned int idx_b = arrays.rtag[bond.b];

		assert(idx_a < m_pdata->getN());
		assert(idx_b < m_pdata->getN());

		// calculate d\vec{r}		
		Scalar dx = arrays.x[idx_b] - arrays.x[idx_a];
		Scalar dy = arrays.y[idx_b] - arrays.y[idx_a];
		Scalar dz = arrays.z[idx_b] - arrays.z[idx_a];

		// if the vector crosses the box, pull it back
		if (dx >= Lx2)
			dx -= Lx;
		else
		if (dx < -Lx2)
			dx += Lx;
		
		if (dy >= Ly2)
			dy -= Ly;
		else
		if (dy < -Ly2)
			dy += Ly;
		
		if (dz >= Lz2)
			dz -= Lz;
		else
		if (dz < -Lz2)
			dz += Lz;

		// sanity check
		assert(dx >= box.xlo && dx < box.xhi);
		assert(dy >= box.ylo && dx < box.yhi);
		assert(dz >= box.zlo && dx < box.zhi);

		// on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
		// now calculate r
		Scalar r = sqrt(dx*dx+dy*dy+dz*dz);
		Scalar tmp = m_K[bond.type] * (m_r_0[bond.type] / r - Scalar(1.0));
		Scalar tmp_eng = Scalar(0.5) * m_K[bond.type] * (m_r_0[bond.type] - r) * (m_r_0[bond.type] - r);
		
		// add the force to the particles
		m_fx[idx_b] += tmp * dx;
		m_fy[idx_b] += tmp * dy;
		m_fz[idx_b] += tmp * dz;
		m_pe[idx_b] += Scalar(0.5)*tmp_eng;
		m_fx[idx_a] -= tmp * dx;
		m_fy[idx_a] -= tmp * dy;
		m_fz[idx_a] -= tmp * dz;
		m_pe[idx_a] += Scalar(0.5)*tmp_eng;
		} 

	m_pdata->release();

	#ifdef USE_CUDA
	// the data is now only up to date on the CPU
	m_data_location = cpu;
	#endif

	// each bond computes ~39 FLOPS
	// and 4 index reads, then 6 position reads and 6 force writes
	if (m_prof)
		{
		m_prof->pop(39*m_bond_data->getNumBonds(), (4*sizeof(int)+16*sizeof(Scalar))*m_bond_data->getNumBonds());
		m_prof->pop();
		}
	}
	
#ifdef USE_PYTHON
void export_HarmonicBondForceCompute()
	{
	class_<HarmonicBondForceCompute, boost::shared_ptr<HarmonicBondForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("HarmonicBondForceCompute", init< boost::shared_ptr<ParticleData> >())
		.def("setParams", &HarmonicBondForceCompute::setParams)
		;
	}
#endif