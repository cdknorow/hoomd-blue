/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: dnlebard

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "DLTAngleForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;


/*! \file DLTAngleForceCompute.cc
    \brief Contains code for the DLTAngleForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
DLTAngleForceCompute::DLTAngleForceCompute(boost::shared_ptr<SystemDefinition> sysdef)
    :  ForceCompute(sysdef), m_K1(NULL), m_K2(NULL), m_b_x(NULL),m_b_y(NULL),m_b_z(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing DLTAngleForceCompute" << endl;

    // access the angle data for later use
    m_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    if (m_angle_data->getNTypes() == 0)
        {
        m_exec_conf->msg->error() << "angle.DLT: No angle types specified" << endl;
        throw runtime_error("Error initializing DLTAngleForceCompute");
        }

    // allocate the parameters
    m_K1   = new Scalar[m_angle_data->getNTypes()];
    m_K2   = new Scalar[m_angle_data->getNTypes()];
    m_b_x  = new Scalar[m_angle_data->getNTypes()];
    m_b_y  = new Scalar[m_angle_data->getNTypes()];
    m_b_z  = new Scalar[m_angle_data->getNTypes()];

    }

DLTAngleForceCompute::~DLTAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying DLTAngleForceCompute" << endl;

    delete[] m_K1;
    delete[] m_K2;
    delete[] m_b_x;
    delete[] m_b_y;
    delete[] m_b_z;
    

    m_K1 = NULL;
    m_K2 = NULL;
    m_b_x = NULL;
    m_b_y = NULL;
    m_b_z = NULL;


    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param bx vector  for the force computation
    \param by vector  for the force computation
    \param bz vector  for the force computation

    Sets parameters for the potential of a particular angle type
*/
void DLTAngleForceCompute::setParams(unsigned int type, Scalar K1, Scalar K2, Scalar b_x, Scalar b_y, Scalar b_z)
    {
    // make sure the type is valid
    if (type >= m_angle_data->getNTypes())
        {
        m_exec_conf->msg->error() << "angle.DLT: Invalid angle type specified" << endl;
        throw runtime_error("Error setting parameters in DLTAngleForceCompute");
        }

    m_K1[type] = K1;
    m_K2[type] = K2;
    m_b_x[type] = b_x;
    m_b_y[type] = b_y;
    m_b_z[type] = b_z;

    }
/*! AngleForceCompute provides
    - \c angle_dlt_energy
*/
std::vector< std::string > DLTAngleForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("dlt_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar DLTAngleForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("dlt_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "angle.dlt: " << quantity << " is not a valid log quantity for DLTForceCompute" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void DLTAngleForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("DLT Energy");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    // for each of the angles
    const unsigned int size = (unsigned int)m_angle_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const AngleData::members_t& angle = m_angle_data->getMembersByIndex(i);
        assert(angle.tag[0] < m_pdata->getNGlobal());
        assert(angle.tag[1] < m_pdata->getNGlobal());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[angle.tag[0]];
        unsigned int idx_b = h_rtag.data[angle.tag[1]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL|| idx_b == NOT_LOCAL )
            {
            this->m_exec_conf->msg->error() << "angle.dlt: angle " <<
                angle.tag[0] << " " << angle.tag[1] << " incomplete." << endl << endl;
            throw std::runtime_error("Error in angle calculation");
            }

        assert(idx_a < m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN()+m_pdata->getNGhosts());

        // calculate vec{r}
        Scalar3 dx;
        dx.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dx.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dx.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        // apply minimum image conventions to the dx vector
        dx = box.minImage(dx);

        // actually calculate the force of V1
    	Scalar3 force_vector;

        // Get Coefficents
        unsigned int angle_type = m_angle_data->getTypeByIndex(i);
        Scalar K1 = m_K1[angle_type];
        Scalar K2 = m_K2[angle_type];
        Scalar b_x = m_b_x[angle_type];
        Scalar b_y = m_b_y[angle_type];
        Scalar b_z = m_b_z[angle_type];

        //Calculate ax ay az and a
        Scalar ax = dx.x + b_x;
        Scalar ay = dx.y + b_y;
        Scalar az = dx.z + b_z;	
        Scalar a  = ax * b_x + ay * b_y + az * b_z;

        // Force F1
        force_vector.x = - K1 * ax;
        force_vector.y = - K1 * ay;
        force_vector.z = - K1 * az;

        //Compute V1 Energy
    	Scalar bond_eng = Scalar(0.5) * K1 * ( ax * ax + ay * ay + az * az );
   
        // Force F2
        force_vector.x += - K2 * b_x * a;
        force_vector.y += - K2 * b_y * a;
        force_vector.z += - K2 * b_z * a;

        // Compute V2 Energy
    	bond_eng += Scalar(0.5) * K2 * (  ax * ax * b_x * b_x +
                                          ay * ay * b_y * b_y +
                                          az * az * b_z * b_z +
                                          2 * ax * b_x * ay * b_y +
                                          2 * ax * b_x * az * b_z +
                                          2 * az * b_z * ay * b_y);

	    // Half bond energy so it is not double counted
        bond_eng *=  Scalar(0.5);
            
	    // compute 1/2 of the virial, 1/2 for each atom in the bond
        // upper triangular version of virial tensor
        Scalar bond_virial[6];
    	bond_virial[0] = Scalar(0.5) * dx.x * force_vector.x; // Fx*x
    	bond_virial[1] = Scalar(0.5) * dx.y * force_vector.x; // Fx*y
    	bond_virial[2] = Scalar(0.5) * dx.z * force_vector.x; // Fx*z
    	bond_virial[3] = Scalar(0.5) * dx.y * force_vector.y; // Fy*y
    	bond_virial[4] = Scalar(0.5) * dx.z * force_vector.y; // Fy*z
    	bond_virial[5] = Scalar(0.5) * dx.z * force_vector.z; // Fz*z

        // Now, apply the force to each individual atom a,b, and accumlate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += force_vector.x;
            h_force.data[idx_a].y += force_vector.y;
            h_force.data[idx_a].z += force_vector.z;
            h_force.data[idx_a].w += bond_eng ;
            for (int j = 0; j < 6; j++)
                h_virial.data[j*virial_pitch+idx_a]  += bond_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= force_vector.x;
            h_force.data[idx_b].y -= force_vector.y;
            h_force.data[idx_b].z -= force_vector.z;
            h_force.data[idx_b].w += bond_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j*virial_pitch+idx_b]  += bond_virial[j];
            }
        }

    if (m_prof) m_prof->pop();
    }

void export_DLTAngleForceCompute()
    {
    class_<DLTAngleForceCompute, boost::shared_ptr<DLTAngleForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("DLTAngleForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setParams", &DLTAngleForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
