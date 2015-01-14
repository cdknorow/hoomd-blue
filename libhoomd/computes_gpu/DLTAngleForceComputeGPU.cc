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

/*! \file DLTAngleForceComputeGPU.cc
    \brief Defines DLTAngleForceComputeGPU
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "DLTAngleForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute angle forces on
*/
DLTAngleForceComputeGPU::DLTAngleForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : DLTAngleForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a DLTAngleForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing DLTAngleForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar2> params_k(m_angle_data->getNTypes(), exec_conf);
    GPUArray<Scalar4> params_b(m_angle_data->getNTypes(), exec_conf);
    m_params_k.swap(params_k);
    m_params_b.swap(params_b);
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "dlt_angle", this->m_exec_conf));
    }

DLTAngleForceComputeGPU::~DLTAngleForceComputeGPU()
    {
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void DLTAngleForceComputeGPU::setParams(unsigned int type, Scalar K1, Scalar K2, Scalar b_x, Scalar b_y, Scalar b_z)
    {
    DLTAngleForceCompute::setParams(type, K1, K2, b_x, b_y, b_z);

    ArrayHandle<Scalar2> params_k(m_params_k, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> params_b(m_params_b, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    params_k.data[type] = make_scalar2(K1, K2);
    params_b.data[type] = make_scalar4(b_x, b_y, b_z, Scalar(0.0));
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_angle_forces to do the dirty work.
*/
void DLTAngleForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "DLT Angle");

    // the angle table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);


    ArrayHandle<Scalar2> d_params_k(m_params_k, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_params_b(m_params_b, access_location::device, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_angle_data->getGPUTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_angle_data->getGPUPosTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_angle_data->getNGroupsArray(), access_location::device, access_mode::read);

    // run the kernel on the GPU
    m_tuner->begin();
    gpu_compute_dlt_angle_forces(d_force.data,
                                      d_virial.data,
                                      m_virial.getPitch(),
                                      m_pdata->getN(),
                                      d_pos.data,
                                      box,
                                      d_gpu_anglelist.data,
                                      d_gpu_angle_pos_list.data,
                                      m_angle_data->getGPUTableIndexer().getW(),
                                      d_gpu_n_angles.data,
                                      d_params_k.data,
                                      d_params_b.data,
                                      m_angle_data->getNTypes(),
                                      m_tuner->getParam(),
                                      m_exec_conf->getComputeCapability());

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop(exec_conf);
    }

void export_DLTAngleForceComputeGPU()
    {
    class_<DLTAngleForceComputeGPU, boost::shared_ptr<DLTAngleForceComputeGPU>, bases<DLTAngleForceCompute>, boost::noncopyable >
    ("DLTAngleForceComputeGPU", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

