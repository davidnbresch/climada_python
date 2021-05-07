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

Test TropCyclone class
"""

import unittest
import datetime as dt
import numpy as np
from scipy import sparse

from climada import CONFIG
from climada.util import ureg
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.trop_cyclone import TropCyclone,\
     _bs_hol08, _close_centroids, _stat_holland, _vtrans
from climada.hazard.centroids.centr import Centroids

DATA_DIR = CONFIG.hazard.test_data.dir()
HAZ_TEST_MAT = DATA_DIR.joinpath('atl_prob_no_name.mat')
TEST_TRACK = DATA_DIR.joinpath("trac_brb_test.csv")
TEST_TRACK_SHORT = DATA_DIR.joinpath("trac_short_test.csv")

CENTR_TEST_BRB = Centroids()
CENTR_TEST_BRB.read_mat(DATA_DIR.joinpath('centr_brb_test.mat'))


class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_set_one_pass(self):
        """Test _tc_from_track function."""
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = {
            "geosphere": [25.60794285, 26.90906280, 28.26649026, 25.54076797, 31.21986961,
                          36.17171808, 21.11408573, 28.01457948, 32.65349378, 31.34027741, 0,
                          40.27362679],
            "equirect": [25.60778909, 26.90887264, 28.26624642, 25.54092386, 31.21941738,
                         36.16596567, 21.11399856, 28.01452136, 32.65076804, 31.33884098, 0,
                         40.27002104]
        }
        # the values for the two metrics should agree up to first digit at least
        for i, val in enumerate(intensity_values["geosphere"]):
            self.assertAlmostEqual(intensity_values["equirect"][i], val, 1)

        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data = tc_track.data[:1]

        for metric in ["equirect", "geosphere"]:
            tc_haz = TropCyclone()
            tc_haz.set_from_tracks(tc_track, centroids=CENTR_TEST_BRB, model='H08',
                                   store_windfields=True, metric=metric)

            self.assertEqual(tc_haz.tag.haz_type, 'TC')
            self.assertEqual(tc_haz.tag.description, '')
            self.assertEqual(tc_haz.tag.file_name, 'Name: 1951239N12334')
            self.assertEqual(tc_haz.units, 'm/s')
            self.assertEqual(tc_haz.centroids.size, 296)
            self.assertEqual(tc_haz.event_id.size, 1)
            self.assertEqual(tc_haz.date.size, 1)
            self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).year, 1951)
            self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).month, 8)
            self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).day, 27)
            self.assertEqual(tc_haz.event_id[0], 1)
            self.assertEqual(tc_haz.event_name, ['1951239N12334'])
            self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
            self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
            self.assertEqual(tc_haz.fraction.shape, (1, 296))
            self.assertEqual(tc_haz.fraction[0, 100], 1)
            self.assertEqual(tc_haz.fraction[0, 260], 0)
            self.assertEqual(tc_haz.fraction.nonzero()[0].size, 280)

            self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
            self.assertEqual(tc_haz.intensity.shape, (1, 296))
            self.assertEqual(np.nonzero(tc_haz.intensity)[0].size, 280)

            for idx, val in zip(intensity_idx, intensity_values[metric]):
                if val == 0:
                    self.assertEqual(tc_haz.intensity[0, idx], 0)
                else:
                    self.assertAlmostEqual(tc_haz.intensity[0, idx], val)

            windfields = tc_haz.windfields[0].toarray()
            windfields = windfields.reshape(windfields.shape[0], -1, 2)
            windfield_norms = np.linalg.norm(windfields, axis=-1).max(axis=0)
            intensity = tc_haz.intensity.toarray()[0, :]
            msk = (intensity > 0)
            np.testing.assert_array_equal(windfield_norms[msk], intensity[msk])

    def test_set_one_file_pass(self):
        """Test set function set_from_tracks with one input."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'Name: 1951239N12334')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertEqual(tc_haz.category, tc_track.data[0].category)
        self.assertTrue(np.isnan(tc_haz.basin[0]))
        self.assertIsInstance(tc_haz.basin, list)
        self.assertIsInstance(tc_haz.category, np.ndarray)
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

    def test_two_files_pass(self):
        """Test set function set_from_tracks with two ibtracs."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv([TEST_TRACK_SHORT, TEST_TRACK_SHORT])
        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.remove_duplicates()
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, ['Name: 1951239N12334', 'Name: 1951239N12334'])
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(np.array_equal(tc_haz.orig, np.array([True])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

class TestWindfieldHelpers(unittest.TestCase):
    """Test helper functions of TC wind field model"""

    def test_close_centroids_pass(self):
        """Test _close_centroids function."""
        t_lat = np.array([0, 0, 0])
        t_lon = np.array([1, 2, 3])
        centroids = np.array([[0, 0], [0, 0.9], [-0.9, 1.2], [1, 2.1], [0, 4], [0.5, 3.8]])
        test_mask = np.array([False, True, True, False, False, True])
        mask = _close_centroids(t_lat, t_lon, centroids, buffer=1)
        np.testing.assert_equal(mask, test_mask)

        # example where antimeridian is crossed
        t_lat = np.linspace(-10, 10, 11)
        t_lon = np.linspace(170, 200, 11)
        t_lon[t_lon > 180] -= 360
        centroids = np.array([[-11, 169], [-7, 176], [4, -170], [10, 170], [-10, -160]])
        test_mask = np.array([True, True, True, False, False])
        mask = _close_centroids(t_lat, t_lon, centroids, buffer=5)
        np.testing.assert_equal(mask, test_mask)

    def test_bs_hol08_pass(self):
        """Test _bs_hol08 function. Compare to MATLAB reference."""
        v_trans = 5.241999541820597
        penv = 1010
        pcen = 1005.263333333329
        prepcen = 1005.258500000000
        lat = 12.299999504631343
        tint = 1
        _bs_res = _bs_hol08(v_trans, penv, pcen, prepcen, lat, tint)
        self.assertAlmostEqual(_bs_res, 1.270856908796045)

        v_trans = 5.123882725120426
        penv = 1010
        pcen = 1005.268166666671
        prepcen = 1005.263333333329
        lat = 12.299999279463769
        tint = 1
        _bs_res = _bs_hol08(v_trans, penv, pcen, prepcen, lat, tint)
        self.assertAlmostEqual(_bs_res, 1.265551666104679)

    def test_stat_holland(self):
        """Test _stat_holland function. Compare to MATLAB reference."""
        d_centr = np.array([[293.6067129546862, 298.2652319413182]])
        r_max = np.array([75.547902916671745])
        hol_b = np.array([1.265551666104679])
        penv = np.array([1010.0])
        pcen = np.array([1005.268166666671])
        lat = np.array([12.299999279463769])
        mask = np.ones_like(d_centr, dtype=bool)

        _v_arr = _stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, mask)[0]
        self.assertAlmostEqual(_v_arr[0], 5.384115724400597)
        self.assertAlmostEqual(_v_arr[1], 5.281356766052531)

        d_centr = np.array([[]])
        mask = np.ones_like(d_centr, dtype=bool)
        _v_arr = _stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, mask)[0]
        self.assertTrue(np.array_equal(_v_arr, np.array([])))

        d_centr = np.array([
                [299.4501244109841, 291.0737897183741, 292.5441003235722]
        ])
        r_max = np.array([40.665454622610511])
        hol_b = np.array([1.486076257880692])
        penv = np.array([1010.0])
        pcen = np.array([970.8727666672957])
        lat = np.array([14.089110370469488])
        mask = np.ones_like(d_centr, dtype=bool)

        _v_arr = _stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, mask)[0]
        self.assertAlmostEqual(_v_arr[0], 11.279764005440288)
        self.assertAlmostEqual(_v_arr[1], 11.682978583939310)
        self.assertAlmostEqual(_v_arr[2], 11.610940769149384)

    def test_vtrans_pass(self):
        """Test _vtrans function. Compare to MATLAB reference."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()

        v_trans, _ = _vtrans(
            tc_track.data[0].lat.values, tc_track.data[0].lon.values,
            tc_track.data[0].time_step.values)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude

        self.assertEqual(v_trans.size, tc_track.data[0].time.size)
        self.assertEqual(v_trans[0], 0)
        self.assertAlmostEqual(v_trans[1] * to_kn, 10.191466246)


class TestClimateSce(unittest.TestCase):
    def test_apply_criterion_track(self):
        """Test _apply_criterion function."""
        tc = TropCyclone()
        tc.intensity = np.zeros((4, 10))
        tc.intensity[0, :] = np.arange(10)
        tc.intensity[1, 5] = 10
        tc.intensity[2, :] = np.arange(10, 20)
        tc.intensity[3, 3] = 3
        tc.intensity = sparse.csr_matrix(tc.intensity)
        tc.basin = ['NA'] * 4
        tc.basin[3] = 'NO'
        tc.category = np.array([2, 0, 4, 1])
        tc.event_id = np.arange(4)
        tc.frequency = np.ones(4) * 0.5

        tc_cc = tc.set_climate_scenario_knu(ref_year=2050, rcp_scenario=45)
        self.assertTrue(np.allclose(tc.intensity[1, :].toarray(), tc_cc.intensity[1, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[3, :].toarray(), tc_cc.intensity[3, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[0, :].toarray(), tc_cc.intensity[0, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[2, :].toarray(), tc_cc.intensity[2, :].toarray()))
        self.assertTrue(np.allclose(tc.frequency, tc_cc.frequency))
        self.assertEqual(
            tc_cc.tag.description,
            'climate change scenario for year 2050 and RCP 45 from Knutson et al 2015.')

    def test_apply_criterion_track(self):
        """Test _apply_criterion function."""
        criterion = [{'basin': 'NA', 'category': [1, 2, 3, 4, 5],
                   'year': 2100, 'change': 1.045, 'variable': 'intensity'}
                   ]
        scale = 0.75

        # artificially increase the size of the hazard by repeating (tiling) the data:
        ntiles = 8

        tc = TropCyclone()
        tc.intensity = np.zeros((4, 10))
        tc.intensity[0, :] = np.arange(10)
        tc.intensity[1, 5] = 10
        tc.intensity[2, :] = np.arange(10, 20)
        tc.intensity[3, 3] = 3
        tc.intensity = np.tile(tc.intensity, (ntiles, 1))
        tc.intensity = sparse.csr_matrix(tc.intensity)
        tc.basin = ['NA'] * 4
        tc.basin[3] = 'WP'
        tc.basin = ntiles * tc.basin
        tc.category = np.array(ntiles * [2, 0, 4, 1])
        tc.event_id = np.arange(tc.intensity.shape[0])

        tc_cc = tc._apply_knutson_criterion(criterion, scale)
        for i_tile in range(ntiles):
            offset = i_tile * 4
            # no factor applied because of category 0
            np.testing.assert_array_equal(
                tc.intensity[offset + 1, :].toarray(), tc_cc.intensity[offset + 1, :].toarray())
            # no factor applied because of basin "WP"
            np.testing.assert_array_equal(
                tc.intensity[offset + 3, :].toarray(), tc_cc.intensity[offset + 3, :].toarray())
            # factor is applied to the remaining events
            np.testing.assert_array_almost_equal(
                tc.intensity[offset + 0, :].toarray() * 1.03375,
                tc_cc.intensity[offset + 0, :].toarray())
            np.testing.assert_array_almost_equal(
                tc.intensity[offset + 2, :].toarray() * 1.03375,
                tc_cc.intensity[offset + 2, :].toarray())

    def test_two_criterion_track(self):
        """Test _apply_criterion function with two criteria"""
        criterion = [
            {'basin': 'NA', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1.045, 'variable': 'intensity'},
            {'basin': 'WP', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1.025, 'variable': 'intensity'},
            {'basin': 'WP', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1.025, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [0, 1, 2, 3, 4, 5],
             'year': 2100, 'change': 0.7, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [3, 4, 5],
             'year': 2100, 'change': 1, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [4, 5],
             'year': 2100, 'change': 2, 'variable': 'frequency'}
            ]
        scale = 0.75

        tc = TropCyclone()
        tc.intensity = np.zeros((4, 10))
        tc.intensity[0, :] = np.arange(10)
        tc.intensity[1, 5] = 10
        tc.intensity[2, :] = np.arange(10, 20)
        tc.intensity[3, 3] = 3
        tc.intensity = sparse.csr_matrix(tc.intensity)
        tc.frequency = np.ones(4) * 0.5
        tc.basin = ['NA'] * 4
        tc.basin[3] = 'WP'
        tc.category = np.array([2, 0, 4, 1])
        tc.event_id = np.arange(4)

        tc_cc = tc._apply_knutson_criterion(criterion, scale)
        self.assertTrue(np.allclose(tc.intensity[1, :].toarray(), tc_cc.intensity[1, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[3, :].toarray(), tc_cc.intensity[3, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[0, :].toarray(), tc_cc.intensity[0, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[2, :].toarray(), tc_cc.intensity[2, :].toarray()))
        self.assertTrue(
            np.allclose(tc.intensity[0, :].toarray() * 1.03375, tc_cc.intensity[0, :].toarray()))
        self.assertTrue(
            np.allclose(tc.intensity[2, :].toarray() * 1.03375, tc_cc.intensity[2, :].toarray()))
        self.assertTrue(
            np.allclose(tc.intensity[3, :].toarray() * 1.01875, tc_cc.intensity[3, :].toarray()))

        res_frequency = np.ones(4) * 0.5
        res_frequency[1] = 0.5 * (1 + (0.7 - 1) * scale)
        res_frequency[2] = 0.5 * (1 + (2 - 1) * scale)
        res_frequency[3] = 0.5 * (1 + (1.025 - 1) * scale)
        self.assertTrue(np.allclose(tc_cc.frequency, res_frequency))

    def test_negative_freq_error(self):
        """Test _apply_knutson_criterion with infeasible input."""
        criterion = [{'basin': 'NA', 'category': [0, 1],
                      'year': 2100, 'change': -1.1,
                      'variable': 'frequency'},
                     {'basin': 'NA', 'category': [1],
                      'year': 2100, 'change': 3,
                      'variable': 'frequency'},
                     ]

        tc = TropCyclone()
        tc.frequency = np.ones(2)
        tc.basin = ['NA', 'NA']
        tc.category = np.array([0, 1])
        with self.assertRaises(ValueError):
            tc._apply_knutson_criterion(criterion, 1)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWindfieldHelpers))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestClimateSce))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
