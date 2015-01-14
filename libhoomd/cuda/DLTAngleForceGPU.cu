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

#include "DLTAngleForceGPU.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file DLTAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the DLT angle forces. Used by DLTAngleForceComputeGPU.
*/

//! Texture for reading angle parameters
scalar4_tex_t angle_params_tex4;
scalar2_tex_t angle_params_tex2;

//! Kernel for caculating DLT angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_params Parameters for the angle force
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
*/
extern "C" __global__ void gpu_compute_dlt_angle_forces_kernel(Scalar4* d_force,
                                                                    Scalar* d_virial,
                                                                    const unsigned int virial_pitch,
                                                                    const unsigned int N,
                                                                    const Scalar4 *d_pos,
                                                                    const Scalar2 *d_params_k,
                                                                    const Scalar4 *d_params_b,
                                                                    BoxDim box,
                                                                    const group_storage<3> *alist,
                                                                    const unsigned int *apos_list,
                                                                    const unsigned int pitch,
                                                                    const unsigned int *n_angles_list)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles = n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos,b_pos; // allocate space for the a and atom in the a-b-c triplet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar3 force_vector;

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch*angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_type = cur_angle.idx[2];

        int cur_angle_abc = apos_list[pitch*angle_idx + idx];

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // if curr_angle == 2, this is a dummy particle and we don't calculate anything
        if (cur_angle_abc != 2){
             // if curr_angle == 0 the b values are in order,
            if (cur_angle_abc == 0)
                {
                a_pos = idx_pos;
                b_pos = x_pos;
                }
            // if curr_angle == 1 the b values are backwards,
            if (cur_angle_abc == 1)
                {
                b_pos = idx_pos;
                a_pos = x_pos;
                }
 
            // calculate dr for a-b,c-b,and a-c
            Scalar3 dx = a_pos - b_pos;

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // get the angle parameters (MEM TRANSFER: 8 bytes)
            Scalar2 params_k = texFetchScalar2(d_params_k, angle_params_tex2, cur_angle_type);
            Scalar4 params_b = texFetchScalar4(d_params_b, angle_params_tex4, cur_angle_type);
            Scalar K1 = params_k.x;
            Scalar K2 = params_k.y;
            Scalar b_x = params_b.x;
            Scalar b_y = params_b.y;
            Scalar b_z = params_b.z;

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

            bond_eng *=  Scalar(0.5);
                
            // compute 1/2 of the virial, 1/2 for each atom in the bond
            // upper triangular version of virial tensor
            Scalar bond_virial[6];
            bond_virial[0] = Scalar(0.5) * dx.x * force_vector.x; // Fx*x
            bond_virial[1] = Scalar(0.5) * dx.y * force_vector.x; // Fx*y
            bond_virial[2] = Scalar(0.5) * dx.z * force_vector.x; // Fx*z
            bond_virial[3] = Scalar(0.5) * dx.y * force_vector.y; // Fy*y
            bond_virial[4] = Scalar(0.5) * dx.z * force_vector.y; // Fy*z

            if (cur_angle_abc == 0)
                {
                force_idx.x += force_vector.x;
                force_idx.y += force_vector.y;
                force_idx.z += force_vector.z;
                }
            if (cur_angle_abc == 1)
                {
                force_idx.x -= force_vector.x;
                force_idx.y -= force_vector.y;
                force_idx.z -= force_vector.z;
                }

            force_idx.w += bond_eng;

            for (int i = 0; i < 6; i++)
                virial[i] += bond_virial[i];
            }
        }
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    for (int i = 0; i < 6; i++)
        d_virial[i*virial_pitch+idx] = virial[i];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial arary
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params K and t_0 params packed as Scalar2 variables
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains K the spring constant
    and the y component contains t_0 the equilibrium angle.
*/
cudaError_t gpu_compute_dlt_angle_forces(Scalar4* d_force,
                                              Scalar* d_virial,
                                              const unsigned int virial_pitch,
                                              const unsigned int N,
                                              const Scalar4 *d_pos,
                                              const BoxDim& box,
                                              const group_storage<3> *atable,
                                              const unsigned int *apos_list,
                                              const unsigned int pitch,
                                              const unsigned int *n_angles_list,
                                              Scalar2 *d_params_k,
                                              Scalar4 *d_params_b,
                                              unsigned int n_angle_types,
                                              int block_size,
                                              const unsigned int compute_capability)
    {
    assert(d_params);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_dlt_angle_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);
    // bind the texture on pre sm 35 arches
    if (compute_capability < 350)
        {
        cudaError_t error = cudaBindTexture(0, angle_params_tex4, d_params_b, sizeof(Scalar4) * n_angle_types);
        if (error != cudaSuccess)
            return error;
        cudaError_t error2 = cudaBindTexture(0, angle_params_tex2, d_params_k, sizeof(Scalar2) * n_angle_types);
        if (error2 != cudaSuccess)
            return error2;
        }

    // run the kernel
    gpu_compute_dlt_angle_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, d_params_k, d_params_b, box,
        atable, apos_list, pitch, n_angles_list);

    return cudaSuccess;
    }

