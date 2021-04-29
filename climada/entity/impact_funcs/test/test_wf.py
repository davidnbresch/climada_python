"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test IFWildfire class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.wildfire import IFWildfire

THRESH_step = 331
i_half_check = 523.8

class TestIFWildfire(unittest.TestCase):

    """Impact function test"""
    def test_default_values_FIRMS_pass(self):
        """Compute mdr interpolating values. For the calibrated function"""
        imp_fun = IFWildfire()
        imp_fun.set_default_FIRMS(i_half_check)
        self.assertEqual(imp_fun.name, 'wildfire default 10 km')
        self.assertEqual(imp_fun.haz_type, 'WFsingle')
        self.assertEqual(imp_fun.id, 1)
        self.assertEqual(imp_fun.intensity_unit, 'K')
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(295,500,5)))

        i_thresh = 295
        i_half = i_half_check
        intensity = np.arange(295,500,5)
        i_n = (intensity-i_thresh)/(i_half-i_thresh)
        paa = i_n**3/(1+i_n**3)
        self.assertTrue(np.array_equal(imp_fun.paa, paa))
        self.assertTrue(np.array_equal(imp_fun.mdd, np.ones(len(paa))))


    def test_step_values_pass(self, threshold=THRESH_step):
        """Compute mdr interpolating values. For the step function"""
        imp_fun = IFWildfire()
        imp_fun.set_step(THRESH_step)
        self.assertEqual(imp_fun.name, 'wildfire step')
        self.assertEqual(imp_fun.haz_type, 'WFsingle')
        self.assertEqual(imp_fun.id, 1)
        self.assertEqual(imp_fun.intensity_unit, 'K')
        self.assertTrue(np.array_equal(imp_fun.intensity,
                            np.array([295, THRESH_step, THRESH_step, 500])))
        self.assertTrue(np.array_equal(imp_fun.paa, np.array([1, 1, 1, 1])))
        self.assertTrue(np.array_equal(imp_fun.mdd, np.array([0, 0, 1, 1])))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIFWildfire)
    unittest.TextTestRunner(verbosity=2).run(TESTS)



