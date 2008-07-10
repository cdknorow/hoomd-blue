# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$

import hoomd;
import globals;
import update;

## \package hoomd_script.init
# \brief Data initialization commands
#
# These commands initialize the particle data. The execution
# configuration must be defined first (todo: document this),
# and the initialization must be performed before any other 
# script commands can be run.  
#

## Reads particles from a hoomd_xml file
#
# \param file_name File to read in
#
# \b Examples:<br>
# init.read_xml(file_name="data.xml")<br>
# pdata = init.read_xml(file_name="directory/data.xml")<br>
#
# All particles, bonds, etc...  are read from the hoomd_xml file given.
# After this command completes, the system is initialized allowing 
# many other commands in hoomd_script to be run. For more details
# on the file format read by this command, see hoomd_xml.
#
# Initialization can only occur once. An error will be generated
# if any initialization command is called after read_xml().
#
def read_xml(file_name):
	print "init.read_xml(file_name=", file_name, ")";

	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError('Error initializing');

	# read in the data
	initializer = hoomd.HOOMDInitializer(file_name);
	globals.particle_data = hoomd.ParticleData(initializer);
	
	# TEMPORARY HACK for bond initialization
	globals.initializer = initializer;

	# initialize the system
	globals.system = hoomd.System(globals.particle_data, initializer.getTimeStep());
	
	_perform_common_init_tasks();
	return globals.particle_data;


## Generates randomly positioned particles
#
# \param N Number of particles to create
# \param phi_p Packing fraction of particles in the simulation box
# \param name Name of the particle type to create
# \param min_dist Minimum distance particles will be separated by
# \param wall_offset (optional) If specified, walls are created a distance of 
#	\a wall_offset in from the edge of the simulation box
#
# \b Examples:<br>
# init.create_random(N=2400, phi_p=0.20)<br>
# init.create_random(N=2400, phi_p=0.40, min_dist=0.5)<br>
# init.create_random(wall_offset=3.1, phi_p=0.10, N=6000)<br>
#
# \a N particles are randomly placed in the simulation box. The 
# dimensions of the created box are such that the packing fraction
# of particles in the box is \a phi_p. A number density \e n
# can be related to the packing fraction by \f$n = 6/\pi \cdot \phi_P\f$.
# All particles are created with the same type, 0.
# After this command completes, the system is initialized allowing 
# many other commands in hoomd_script to be run.
#
# Initialization can only occur once. An error will be generated
# if any initialization command is called after create_random().
#
def create_random(N, phi_p, name="A", min_dist=1.0, wall_offset=None):
	print "init.create_random(N =", N, ", phi_p =", phi_p, ", name = ", name, ", min_dist =", min_dist, ", wall_offset =", wall_offset, ")";

	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError('Error initializing');

	# read in the data
	if wall_offset == None:
		initializer = hoomd.RandomInitializer(N, phi_p, min_dist, name);
	else:
		initializer = hoomd.RandomInitializerWithWalls(N, phi_p, min_dist, wall_offset, name);
		
	globals.particle_data = hoomd.ParticleData(initializer);

	# initialize the system
	globals.system = hoomd.System(globals.particle_data, 0);
	
	_perform_common_init_tasks();
	return globals.particle_data;

## Generates randomly positioned polymers
#
# \param box BoxDim specifying the simulation box to generate the polymers in
# \param polymers Specification for the different polymers to create (see below)
# \param separation Separation radii for different particle types (see below)
# \param seed Random seed to use
#
# A lot of information must be passed into the generator so that the desired polymer
# system is generated. This requires packing a lot of information into only a few
# arguments. Any number of polymers can be generated, of the same or different types.
# \a polymers is the argument that specifies this. For each polymer, there is a 
# bond length, particle type list, bond list, and count.
#
# The syntax is best shown by example. The below line specifies a that 600 block copolymers
# A6B7A6 with a bond length of 1.2 be generated.
# \code
# polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="TODO", count=600)
# \endcode
# Here is an example for a second polymer, specifying just 100 polymers made of 4 B beads
# \code
# polymer2 = dict(bond_len=1.2, type=['B']*4, bond="TODO", count=100)
# \endcode
# The \a polymers argument can be given a list of any number of polymer types specified
# as above. \a count randomly generated polymers of each polymer in the list will be
# generated in the system.
# 
# \a separation \b must contain one entry for each particle type in the polymers
# (only 'A' and 'B' in the examples above). It is OK to specify more particles in
# separation than are needed. The value given is the radius of each
# particle of that type. The generated polymer system will have no two overlapping 
# particles.
#
# \b Examples:<br>
# init.create_random_polymers(box=hoomd.BoxDim(25), polymers=[polymer1, polymer2], separation=dict(A=0.5, B=0.5));
# init.create_random_polymers(box=hoomd.BoxDim(20), polymers=[polymer1], separation=dict(A=0.5, B=0.5), seed=52);
# init.create_random_polymers(box=hoomd.BoxDim(19), polymers=[polymer2], separation=dict(A=0.3, B=0.3), seed=12345);
#
# With all other parameters the same, create_random_polymers will always create the
# same system if \a seed is the same. Set a different \a seed (any integer) to create
# a different random system with the same parameters. Note that different versions
# of HOOMD \e may generate different systems even with the same seed due to programming
# changes.
#
# \note For relatively dense systems (packing fraction 0.2 and higher) the simple random
# generation algorithm may fail and print an error message. There are two methods to solve this.
# First, you can lower the separation radii allowing particles to be placed closer together.
# Then setup integrate.nve with the \a limit option set to a relatively small value. A few
# thousand timesteps should relax the system so that continuing the simulation can be
# continued without the limit or with a different integrator. For extremely troublesome systems,
# generate at a low density and shrink the box (TODO, write box shrink updater) to the desired 
# final size.
#
def create_random_polymers(box, polymers, separation, seed=1):
	print "init.create_random_polymers(box =", box, ", polymers =", polymers, ", separation = ", separation, ", seed =", seed, ")";
	
	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError("Error creating random polymers");
	
	if type(polymers) != type([]) or len(polymers) == 0:
		print "Argument error: polymers specified incorrectly. See the hoomd_script documentation"
		raise RuntimeError("Error creating random polymers");
	 
	if type(separation) != type(dict()) or len(separation) == 0:
		print "Argument error: polymers specified incorrectly. See the hoomd_script documentation"
		raise RuntimeError("Error creating random polymers");
	
	# create the generator
	generator = hoomd.RandomGenerator(box, seed);
	
	# make a list of types used for an eventual check vs the types in separation for completeness
	types_used = [];
	
	# build the polymer generators
	for poly in polymers:
		type_list = [];
		# check that all fields are specified
		if not 'bond_len' in poly:
			print 'Polymer specification missing bond_len';
			raise RuntimeError("Error creating random polymers");
		if not 'type' in poly:
			print 'Polymer specification missing type';
			raise RuntimeError("Error creating random polymers");
		if not 'count' in poly:	
			print 'Polymer specification missing count';
			raise RuntimeError("Error creating random polymers");
		
		# build type list
		type_vector = hoomd.std_vector_string();
		for t in poly['type']:
			type_vector.append(t);
			if not t in types_used:
				types_used.append(t);
		
		# create the generator
		generator.addGenerator(poly['count'], hoomd.PolymerParticleGenerator(poly['bond_len'], type_vector, 100));
		
		
	# check that all used types are in the separation list
	for t in types_used:
		if not t in separation:
			print "No separation radius specified for type ", t;
			raise RuntimeError("Error creating random polymers");
			
	# set the separation radii
	for t,r in separation.items():
		generator.setSeparationRadius(t, r);
		
	# generate the particles
	generator.generate();
	
	globals.particle_data = hoomd.ParticleData(generator);
	
	# TEMPORARY HACK for bond initialization
	globals.initializer = generator;
	
	# initialize the system
	globals.system = hoomd.System(globals.particle_data, 0);
	
	_perform_common_init_tasks();
	return globals.particle_data;


## Performs common initialization tasks
#
# \internal
# Initialization tasks that are performed for every simulation are to
# be done here. For example, setting up the SFCPackUpdater, initializing
# the log writer, etc...
#
# Currently only creates the sorter
def _perform_common_init_tasks():
	# create the sorter, using the evil import __main__ trick to provide the user with a default variable
	import __main__;
	__main__.sorter = update.sort();