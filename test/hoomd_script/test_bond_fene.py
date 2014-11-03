# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests bond.fene
class bond_fene_tests (unittest.TestCase):
    def setUp(self):
        print
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = data.boxdim(L=35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);

        sorter.set_params(grid=8)

    # test to see that se can create a force.constant
    def test_create(self):
        bond.fene();

    # test setting coefficients
    def test_set_coeff(self):
        fene = bond.fene();
        fene.bond_coeff.set('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon=2.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    # test integrating with a zero force constant
    def test_zero_coeff(self):
        fene = bond.fene();
        fene.bond_coeff.set('polymer', k=0.0, r0=.001, sigma=1.0, epsilon=0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        fene = bond.fene();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
