"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test of lines_polys_handler
"""

import unittest

import climada.util.lines_polys_handler as u_lp_handler
from climada.engine import Impact


class TestMakeExposure(unittest.TestCase):
    """Test yearset functions"""
    def test_point_exposure_from_lines(self):
        """..."""
        #TODO: Implement
        pass

    def test_point_exposure_from_polygons(self):
        """..."""
        #TODO: Implement
        pass

class TestAggregate(unittest.TestCase):
    """Test yearset functions"""
    def test_agg_point_impact_to_lines(self):
        """..."""
        #TODO: Implement
        pass

    def test_agg_point_impact_to_polygons(self):
        """..."""
        #TODO: Implement
        pass
    

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestMakeExposure)
    TESTS.addTest(TestAggregate)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
