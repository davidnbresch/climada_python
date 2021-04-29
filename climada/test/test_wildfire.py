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

Test Wild fire class
"""
from pathlib import Path
import unittest
import numpy as np

from climada.hazard import WildFire

DATA_DIR = (Path(__file__).parent).joinpath('../test/data')
TEST_FIRMS = Path.joinpath(DATA_DIR, "WF_FIRMS.csv")

# DATA_DIR = os.path.join(os.path.dirname(__file__), '../test/data')
# TEST_FIRMS = os.path.join(DATA_DIR, "WF_FIRMS.csv")

description = ''

class TestWildFire(unittest.TestCase):
    """Test loading functions from the WildFire class"""

    def test_hist_fire_firms_pass(self):
        """ Test set_hist_events """
        wf = WildFire()
        wf.set_hist_fire_FIRMS(TEST_FIRMS)
        wf.check()

        self.assertEqual(wf.tag.haz_type, 'WFsingle')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 13)))
        self.assertTrue(np.allclose(wf.date,
            np.array([736186, 736185, 736185, 736184, 736184, 736185, 736184, 736185,
               736185, 736184, 736185, 736186])))
        self.assertTrue(np.allclose(wf.orig, np.ones(12, bool)))
        self.assertEqual(wf.event_name, ['1', '2', '3', '4', '5', '6', '7', '8', '9',
            '10', '11', '12'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(12)))
        self.assertEqual(wf.intensity.shape, (12, 51042))
        self.assertEqual(wf.fraction.shape, (12, 51042))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 42159)
        self.assertEqual(wf.intensity[3, :].nonzero()[1][5], 27737)
        self.assertEqual(wf.intensity[7, :].nonzero()[1][4], 18821)
        self.assertAlmostEqual(wf.intensity[0, 42159], 324.1)
        self.assertAlmostEqual(wf.intensity[3, 27737], 392.5)
        self.assertAlmostEqual(wf.intensity[7, 18821], 312.8)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

    def test_hist_fire_season_firms_pass(self):
        """ Test set_hist_event_year_set """
        wf = WildFire()
        wf.set_hist_fire_seasons_FIRMS(TEST_FIRMS)

        self.assertEqual(wf.tag.haz_type, 'WFseason')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 2)))
        self.assertTrue(np.allclose(wf.date, np.array([735964])))
        self.assertTrue(np.allclose(wf.orig, np.ones(1, bool)))
        self.assertEqual(wf.event_name, ['2016'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(1)))
        self.assertEqual(wf.intensity.shape, (1, 51042))
        self.assertEqual(wf.fraction.shape, (1, 51042))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 47)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][20], 1296)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][100], 4005)
        self.assertAlmostEqual(wf.intensity[0, 47], 305.0)
        self.assertAlmostEqual(wf.intensity[0, 1296], 342.5)
        self.assertAlmostEqual(wf.intensity[0, 4005], 318.5)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

    def test_proba_fire_season_pass(self):
        """ Test probabilistic set_probabilistic_event_year_set """
        wf = WildFire()
        wf.set_hist_fire_seasons_FIRMS(TEST_FIRMS)
        wf.set_proba_fire_seasons(1,[3,4])

        self.assertEqual(wf.size, 2)
        orig = np.zeros(2, bool)
        orig[0] = True
        self.assertTrue(np.allclose(wf.orig, orig))
        self.assertEqual(len(wf.event_name), 2)
        self.assertEqual(wf.intensity.shape, (2, 51042))
        self.assertEqual(wf.fraction.shape, (2, 51042))
        self.assertEqual(wf.intensity[1, :].nonzero()[1][0], 5080)
        self.assertEqual(wf.intensity[1, :].nonzero()[1][4], 32991)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][11], 939)
        self.assertAlmostEqual(wf.intensity[1, 5080], 334.3)
        self.assertAlmostEqual(wf.intensity[1, 32991], 309.9)
        self.assertAlmostEqual(wf.intensity[0, 939], 356.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

    def test_summarize_fires_to_seasons_pass(self):
        """ Test probabilistic set_probabilistic_event_year_set """
        wf = WildFire()
        wf.set_hist_fire_FIRMS(TEST_FIRMS)
        wf.summarize_fires_to_seasons()

        self.assertEqual(wf.tag.haz_type, 'WFseason')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 2)))
        self.assertTrue(np.allclose(wf.date, np.array([735964])))
        self.assertTrue(np.allclose(wf.orig, np.ones(1, bool)))
        self.assertEqual(wf.event_name, ['2016'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(1)))
        self.assertEqual(wf.intensity.shape, (1, 51042))
        self.assertEqual(wf.fraction.shape, (1, 51042))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 47)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][20], 1296)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][100], 4005)
        self.assertAlmostEqual(wf.intensity[0, 47], 305.0)
        self.assertAlmostEqual(wf.intensity[0, 1296], 342.5)
        self.assertAlmostEqual(wf.intensity[0, 4005], 318.5)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestWildFire)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
