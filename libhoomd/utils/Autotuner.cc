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

#include <iostream>
#include <stdexcept>
#include <algorithm>

#include <boost/python.hpp>
using namespace boost::python;

#include "Autotuner.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

using namespace std;

/*! \file Autotuner.cc
    \brief Definition of Autotuner
*/

/*! \param parameters List of valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration
*/
Autotuner::Autotuner(const std::vector<unsigned int>& parameters,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     boost::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_period(period), m_enabled(true), m_name(name), m_parameters(parameters),
      m_state(STARTUP), m_current_sample(0), m_current_element(0), m_calls(0),
      m_exec_conf(exec_conf), m_mode(mode_median)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << nsamples << " " << period << " " << name << endl;

    // ensure that m_nsamples is odd (so the median is easy to get). This also ensures that m_nsamples > 0.
    if ((m_nsamples & 1) == 0)
        m_nsamples += 1;

    // initialize memory
    if (m_parameters.size() == 0)
        {
        this->m_exec_conf->msg->error() << "Autotuner " << m_name << " got no parameters" << endl;
        throw std::runtime_error("Error initializing autotuner");
        }
    m_samples.resize(m_parameters.size());
    m_sample_median.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_nsamples);
        }

    m_current_param = m_parameters[m_current_element];

    // create CUDA events
    #ifdef ENABLE_CUDA
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
    #endif

    m_sync = false;
    }


/*! \param start first valid parameter
    \param end last valid parameter
    \param step spacing between valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration

    \post Valid parameters will be generated with a spacing of \a step in the range [start,end] inclusive.
*/
Autotuner::Autotuner(unsigned int start,
                     unsigned int end,
                     unsigned int step,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     boost::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_period(period), m_enabled(true), m_name(name),
      m_state(STARTUP), m_current_sample(0), m_current_element(0), m_calls(0), m_current_param(0),
      m_exec_conf(exec_conf), m_mode(mode_median)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << " " << start << " " << end << " " << step << " "
                                << nsamples << " " << period << " " << name << endl;

    // initialize the parameters
    m_parameters.resize((end - start) / step + 1);
    unsigned int cur_param = start;
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_parameters[i] = cur_param;
        cur_param += step;
        }

    // ensure that m_nsamples is odd (so the median is easy to get). This also ensures that m_nsamples > 0.
    if ((m_nsamples & 1) == 0)
        m_nsamples += 1;

    // initialize memory
    if (m_parameters.size() == 0)
        {
        m_exec_conf->msg->error() << "Autotuner " << m_name << " got no parameters" << endl;
        throw std::runtime_error("Error initializing autotuner");
        }
    m_samples.resize(m_parameters.size());
    m_sample_median.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_nsamples);
        }

    m_current_param = m_parameters[m_current_element];

    // create CUDA events
    #ifdef ENABLE_CUDA
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
    #endif

    m_sync = false;
    }

Autotuner::~Autotuner()
    {
    m_exec_conf->msg->notice(5) << "Destroying Autotuner " << m_name << endl;
    #ifdef ENABLE_CUDA
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
    CHECK_CUDA_ERROR();
    #endif
    }

void Autotuner::begin()
    {
    // skip if disabled
    if (!m_enabled)
        return;

    #ifdef ENABLE_CUDA
    // if we are scanning, record a cuda event - otherwise do nothing
    if (m_state == STARTUP || m_state == SCANNING)
        {
        cudaEventRecord(m_start, 0);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    #endif
    }

void Autotuner::end()
    {
    // skip if disabled
    if (!m_enabled)
        return;

    #ifdef ENABLE_CUDA
    // handle timing updates if scanning
    if (m_state == STARTUP || m_state == SCANNING)
        {
        cudaEventRecord(m_stop, 0);
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&m_samples[m_current_element][m_current_sample], m_start, m_stop);
        m_exec_conf->msg->notice(9) << "Autotuner " << m_name << ": t(" << m_current_param << "," << m_current_sample
                                     << ") = " << m_samples[m_current_element][m_current_sample] << endl;

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    #endif

    // handle state data updates and transitions
    if (m_state == STARTUP)
        {
        // move on to the next sample
        m_current_sample++;

        // if we hit the end of the samples, reset and move on to the next element
        if (m_current_sample >= m_nsamples)
            {
            m_current_sample = 0;
            m_current_element++;

            // if we hit the end of the elements, transition to the IDLE state and compute the optimal parameter
            if (m_current_element >= m_parameters.size())
                {
                m_current_element = 0;
                m_state = IDLE;
                m_current_param = computeOptimalParameter();
                }
            else
                {
                // if moving on to the next element, update the cached parameter to set
                m_current_param = m_parameters[m_current_element];
                }
            }
        }
    else if (m_state == SCANNING)
        {
        // move on to the next element
        m_current_element++;

        // if we hit the end of the elements, transition to the IDLE state and compute the optimal parameter, and move
        // on to the next sample for next time
        if (m_current_element >= m_parameters.size())
            {
            m_current_element = 0;
            m_state = IDLE;
            m_current_param = computeOptimalParameter();
            m_current_sample = (m_current_sample + 1) % m_nsamples;
            }
        else
            {
            // if moving on to the next element, update the cached parameter to set
            m_current_param = m_parameters[m_current_element];
            }
        }
    else if (m_state == IDLE)
        {
        // increment the calls counter and see if we should transition to the scanning state
        m_calls++;

        if (m_calls > m_period)
            {
            // reset state for the next time
            m_calls = 0;

            // initialize a scan
            m_current_param = m_parameters[m_current_element];
            m_state = SCANNING;
            m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " - beginning scan" << std::endl;
            }
        }
    }

/*! \returns The optimal parameter given the current data in m_samples

    computeOptimalParameter computes the median time among all samples for a given element. It then chooses the
    fastest time (with the lowest index breaking a tie) and returns the parameter that resulted in that time.
*/
unsigned int Autotuner::computeOptimalParameter()
    {
    bool is_root = true;

    #ifdef ENABLE_MPI
    unsigned int nranks = 0;
    if (m_sync)
        {
        nranks = m_exec_conf->getNRanks();
        is_root = !m_exec_conf->getRank();
        }
    #endif

    // start by computing the median for each element
    std::vector<float> v;
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        v = m_samples[i];
        #ifdef ENABLE_MPI
        if (m_sync && nranks)
            {
            // combine samples from all ranks on rank zero
            std::vector< std::vector<float> > all_v;
            MPI_Barrier(m_exec_conf->getMPICommunicator());
            gather_v(v, all_v, 0, m_exec_conf->getMPICommunicator());
            if (is_root)
                {
                v.clear();
                assert(all_v.size() == nranks);
                for (unsigned int j = 0; j < nranks; ++j)
                    v.insert(v.end(), all_v[j].begin(), all_v[j].end());
                }
            }
        #endif
        if (is_root)
            {
            if (m_mode == mode_avg)
                {
                // compute average
                float sum = 0.0f;
                for (std::vector<float>::iterator it = v.begin(); it != v.end(); ++it)
                    sum += *it;
                m_sample_median[i] = sum/v.size();
                }
            else if (m_mode == mode_max)
                {
                // compute maximum
                m_sample_median[i] = -FLT_MIN;
                for (std::vector<float>::iterator it = v.begin(); it != v.end(); ++it)
                    {
                    if (*it > m_sample_median[i])
                        {
                        m_sample_median[i] = *it;
                        }
                    }
                }
            else
                {
                // compute median
                size_t n = v.size() / 2;
                nth_element(v.begin(), v.begin()+n, v.end());
                m_sample_median[i] = v[n];
                }
            }
        }

    unsigned int opt;

    if (is_root)
        {
        // now find the minimum and maximum times in the medians
        float min = m_sample_median[0];
        unsigned int min_idx = 0;
        //float max = m_sample_median[0];
        //unsigned int max_idx = 0;

        for (unsigned int i = 1; i < m_parameters.size(); i++)
            {
            if (m_sample_median[i] < min)
                {
                min = m_sample_median[i];
                min_idx = i;
                }
            /*if (m_sample_median[i] > max)
                {
                max = m_sample_median[i];
                max_idx = i;
                }*/
            }

        // get the optimal param
        opt = m_parameters[min_idx];
        // unsigned int percent = int(max/min * 100.0f)-100;

        // print stats
        m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " found optimal parameter " << opt << endl;
        }

    #ifdef ENABLE_MPI
    if (m_sync && nranks) bcast(opt, 0, m_exec_conf->getMPICommunicator());
    #endif
    return opt;
    }

void export_Autotuner()
    {
    class_<Autotuner, boost::noncopyable>
    ("Autotuner", init< unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const std::string&, boost::shared_ptr<ExecutionConfiguration> >())
    .def("getParam", &Autotuner::getParam)
    .def("setEnabled", &Autotuner::setEnabled)
    .def("setMoveRatio", &Autotuner::isComplete)
    .def("setNSelect", &Autotuner::setPeriod)
    ;
    }
