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

# import os
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse

from climada.hazard.wildfire import WildFire
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import ONE_LAT_KM

DATA_DIR = (Path(__file__).parent).joinpath('data')
TEST_FIRMS = Path.joinpath(DATA_DIR, "California_firms_Soberanes_2016_viirs.csv")


description = ''

def def_ori_centroids(firms, centr_res_factor):
    # res_data in km
    if firms['instrument'].any() == 'MODIS':
        res_data = 1.0
    else:
        res_data = 0.375 # For VIIRS data

    centroids = Centroids()
    dlat_km = abs(firms['latitude'].min() - firms['latitude'].max()) * ONE_LAT_KM
    dlon_km = abs(firms['longitude'].min() - firms['longitude'].max()) * ONE_LAT_KM* \
        np.cos(np.radians((abs(firms['latitude'].min() - firms['latitude'].max()))/2))
    nb_centr_lat = int(dlat_km/res_data * centr_res_factor)
    nb_centr_lon = int(dlon_km/res_data * centr_res_factor)
    coord = (np.mgrid[firms['latitude'].min() : firms['latitude'].max() : complex(0, nb_centr_lat), \
        firms['longitude'].min() : firms['longitude'].max() : complex(0, nb_centr_lon)]). \
        reshape(2, nb_centr_lat*nb_centr_lon).transpose()
    centroids.set_lat_lon(coord[:, 0], coord[:, 1])

    centroids.set_area_approx()
    centroids.set_on_land()
    centroids.empty_geometry_points()
    return centroids, res_data

DEF_CENTROIDS = def_ori_centroids(WildFire._clean_firms_csv(TEST_FIRMS), 1/2)

class TestMethodsFirms(unittest.TestCase):
    """Test loading functions from the WildFire class"""

    def test_clean_firms_pass(self):
        """ Test _clean_firms_csv """
        wf = WildFire()
        firms = wf._clean_firms_csv(TEST_FIRMS)

        self.assertEqual(firms['latitude'][0], 36.46245)
        self.assertEqual(firms['longitude'][0], -121.8989)
        self.assertEqual(firms['latitude'].iloc[-1], 36.17266)
        self.assertAlmostEqual(firms['longitude'].iloc[-1], -121.61211)
        self.assertEqual(firms['datenum'].iloc[-1], 736245)
        self.assertTrue(np.array_equal(firms['iter_ev'].values, np.ones(len(firms), bool)))
        self.assertTrue(np.array_equal(firms['cons_id'].values, np.zeros(len(firms), int)-1))
        self.assertTrue(np.array_equal(firms['clus_id'].values, np.zeros(len(firms), int)-1))

    def test_firms_cons_days_1_pass(self):
        """ Test _firms_cons_days """
        wf = WildFire()
        firms_ori = wf._clean_firms_csv(TEST_FIRMS)
        firms = wf._firms_cons_days(firms_ori.copy())
        self.assertEqual(len(firms), 9325)
        cons_id = np.zeros(9325, int)
        cons_id[8176:-4] = 1
        cons_id[-4:] = 2
        self.assertTrue(np.allclose(firms.cons_id.values, cons_id))
        for col_name in firms.columns:
            if col_name not in ('cons_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'cons_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_firms_cons_days_2_pass(self):
        """ Test _firms_cons_days """
        wf = WildFire()
        firms_ori = wf._clean_firms_csv(TEST_FIRMS)
        firms_ori['datenum'].values[100] = 7000
        firms = wf._firms_cons_days(firms_ori.copy())
        self.assertEqual(len(firms), 9325)
        cons_id = np.ones(9325, int)
        cons_id[100] = 0
        cons_id[8176:-4] = 2
        cons_id[-4:] = 3
        self.assertTrue(np.allclose(firms.cons_id.values, cons_id))
        for col_name in firms.columns:
            if col_name not in ('cons_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'cons_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_firms_cons_days_3_pass(self):
        """ Test _firms_cons_days """
        ori_thres = WildFire.days_thres_firms
        WildFire.days_thres_firms = 3
        wf = WildFire()
        firms_ori = wf._clean_firms_csv(TEST_FIRMS)
        firms = wf._firms_cons_days(firms_ori.copy())
        self.assertEqual(len(firms), 9325)
        cons_id = np.zeros(9325, int)
        cons_id[-4:] = 1
        self.assertTrue(np.allclose(firms.cons_id.values, cons_id))
        for col_name in firms.columns:
            if col_name not in ('cons_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'cons_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)
        WildFire.days_thres_firms = ori_thres

    def test_firms_cluster_pass(self):
        """ Test _firms_clustering """
        wf = WildFire()
        firms_ori = wf._clean_firms_csv(TEST_FIRMS)
        firms_ori['datenum'].values[100] = 7000
        firms_ori = wf._firms_cons_days(firms_ori)
        firms = wf._firms_clustering(firms_ori.copy(), 0.375/2/15, 15)
        self.assertEqual(len(firms), 9325)
        self.assertTrue(np.allclose(firms.clus_id.values[-4:], np.zeros(4, int)))
        self.assertEqual(firms.clus_id.values[100], 0)
        self.assertEqual(firms.clus_id.values[2879], 3)
        self.assertEqual(firms.clus_id.values[0], 2)
        self.assertEqual(firms.clus_id.values[2], 2)
        self.assertEqual(firms.clus_id.values[9320], 0)
        self.assertEqual(firms.clus_id.values[4254], 2)
        self.assertEqual(firms.clus_id.values[4255], 0)
        self.assertEqual(firms.clus_id.values[8105], 1)
        self.assertTrue(np.allclose(np.unique(firms.clus_id.values), np.arange(4)))
        for col_name in firms.columns:
            if col_name not in ('clus_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'clus_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_firms_fire_pass(self):
        """ Test _firms_fire """
        wf = WildFire()
        firms_ori = wf._clean_firms_csv(TEST_FIRMS)
        firms_ori['datenum'].values[100] = 7000
        firms_ori = wf._firms_cons_days(firms_ori)
        firms_ori = wf._firms_clustering(firms_ori, 0.375/2/15, 15)
        firms = firms_ori.copy()
        wf._firms_fire(2, firms.cons_id.values, firms.clus_id.values,
            firms.event_id.values, firms.iter_ev.values, firms.datenum.values)
        self.assertEqual(len(firms), 9325)
        self.assertTrue(np.allclose(np.unique(firms.event_id), np.arange(7)))
        self.assertTrue(np.allclose(firms.event_id[-4:], np.ones(4)*6))
        self.assertEqual(firms.event_id.values[100], 0)
        self.assertEqual(firms.event_id.values[4255], 1)
        self.assertEqual(firms.event_id.values[8105], 2)
        self.assertEqual(firms.event_id.values[0], 3)
        self.assertEqual(firms.event_id.values[2], 3)
        self.assertEqual(firms.event_id.values[4254], 3)
        self.assertEqual(firms.event_id.values[2879], 4)
        self.assertEqual(firms.event_id.values[9320], 5)

        self.assertFalse(firms.iter_ev.any())

        for ev_id in np.unique(firms.event_id):
            if np.unique(firms[firms.event_id == ev_id].cons_id).size != 1:
                self.assertEqual(1, 0)
            if np.unique(firms[firms.event_id == ev_id].clus_id).size != 1:
                self.assertEqual(1, 0)

        for col_name in firms.columns:
            if col_name not in ('event_id', 'acq_date', 'confidence', 'instrument', 'satellite', 'iter_ev'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name not in ('event_id', 'iter_ev'):
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_iter_events_pass(self):
        """ Test identification of events """
        ori_thres = WildFire.days_thres_firms
        wf = WildFire()
        firms = wf._clean_firms_csv(TEST_FIRMS)
        firms['datenum'].values[100] = 7000
        i_iter = 0
        WildFire.days_thres_firms = 3
        while firms.iter_ev.any():
            # Compute cons_id: consecutive events in current iteration
            wf._firms_cons_days(firms)
            # Compute clus_id: cluster identifier inside cons_id
            wf._firms_clustering(firms, 0.375, 15)
            # compute event_id
            WildFire.days_thres_firms = 2
            wf._firms_fire(WildFire.days_thres_firms, firms.cons_id.values, firms.clus_id.values,
                firms.event_id.values, firms.iter_ev.values, firms.datenum.values)
            i_iter += 1
        self.assertEqual(i_iter, 2)
        WildFire.days_thres_firms = ori_thres


    def test_calc_bright_pass(self):
        """ Test _calc_brightness """
#        from pathos.pools import ProcessPool as Pool
#        pool = Pool()
        wf = WildFire()
        firms = wf._clean_firms_csv(TEST_FIRMS)
        firms['datenum'].values[100] = 7000
        firms = wf._firms_cons_days(firms)
        firms = wf._firms_clustering(firms, DEF_CENTROIDS[1]/2/15, 15)
        wf._firms_fire(2, firms.cons_id.values, firms.clus_id.values,
            firms.event_id.values, firms.iter_ev.values, firms.datenum.values)
        firms.latitude[8169] = firms.loc[16]['latitude']
        firms.longitude[8169] = firms.loc[16]['longitude']
        wf._calc_brightness(firms, DEF_CENTROIDS[0], DEF_CENTROIDS[1])
        wf.check()

        self.assertEqual(wf.tag.haz_type, 'WFsingle')
        self.assertEqual(wf.tag.description, '')
        self.assertEqual(wf.units, 'K')
        self.assertEqual(wf.centroids.size, 19454)
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 8)))
        self.assertEqual(wf.event_name, ['1', '2', '3', '4', '5', '6', '7'])
        self.assertTrue(wf.frequency.size, 0)
        self.assertTrue(isinstance(wf.intensity, sparse.csr_matrix))
        self.assertTrue(isinstance(wf.fraction, sparse.csr_matrix))
        self.assertEqual(wf.intensity.shape, (7, 19454))
        self.assertEqual(wf.fraction.shape, (7, 19454))
        self.assertEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.intensity[0, 16618], firms.loc[100].brightness)
        self.assertAlmostEqual(wf.intensity[1, 123], max(firms.loc[6721].brightness, firms.loc[6722].brightness))
        self.assertAlmostEqual(wf.intensity[2, :].max(), firms.loc[8105].brightness)
        self.assertAlmostEqual(wf.intensity[4, 19367], firms.loc[2879].brightness)
        self.assertAlmostEqual(wf.intensity[5, 10132], firms.loc[8176].brightness)
        self.assertAlmostEqual(wf.intensity[6, 10696], max(firms.loc[9322].brightness, firms.loc[9324].brightness))
        self.assertAlmostEqual(wf.intensity[6, 10697], max(firms.loc[9321].brightness, firms.loc[9323].brightness))
        self.assertEqual((wf.intensity[0, :]>0).sum(), 1)
        self.assertEqual((wf.intensity[1, :]>0).sum(), 424)
        self.assertEqual((wf.intensity[2, :]>0).sum(), 1)
        self.assertEqual((wf.intensity[3, :]>0).sum(), 1112)
        self.assertEqual((wf.intensity[4, :]>0).sum(), 1)
        self.assertEqual((wf.intensity[5, :]>0).sum(), 264)
        self.assertEqual((wf.intensity[6, :]>0).sum(), 2)

    def test_set_frequency_pass(self):
        """ Test _set_frequency """
        wf = WildFire()
        wf.date = np.array([736167, 736234, 736235])
        wf.orig = np.ones(3, bool)
        wf.event_id = np.arange(3)
        wf._set_frequency()
        self.assertTrue(np.allclose(wf.frequency, np.ones(3, float)))

        wf = WildFire()
        wf.date = np.array([736167, 736167, 736167, 736234, 736234, 736234, 736235, 736235, 736235])
        wf.orig = np.zeros(9, bool)
        wf.orig[[0, 3, 6]] = True
        wf.event_id = np.arange(9)
        wf._set_frequency()
        self.assertTrue(np.allclose(wf.frequency, np.ones(9, float)/3))

    def test_centroids_pass(self):
        """ Test _firms_centroids_creation """
        wf = WildFire()
        firms = wf._clean_firms_csv(TEST_FIRMS)
        centroids = wf._firms_centroids_creation(firms, 0.375/ONE_LAT_KM, 1/2)
        self.assertEqual(centroids.meta['width'], 144)
        self.assertEqual(centroids.meta['height'], 138)
        self.assertAlmostEqual(centroids.meta['transform'][0], 0.006749460043196544)
        self.assertAlmostEqual(centroids.meta['transform'][1], 0.0)
        self.assertTrue(centroids.meta['transform'][2]<= firms.longitude.min())
        self.assertAlmostEqual(centroids.meta['transform'][3], 0.0)
        self.assertAlmostEqual(centroids.meta['transform'][4], -centroids.meta['transform'][0])
        self.assertTrue(centroids.meta['transform'][5] >= firms.latitude.max())
        self.assertTrue(firms.latitude.max() <= centroids.total_bounds[3])
        self.assertTrue(firms.latitude.min() >= centroids.total_bounds[1])
        self.assertTrue(firms.longitude.max() <= centroids.total_bounds[2])
        self.assertTrue(firms.longitude.min() >= centroids.total_bounds[0])
        self.assertTrue(centroids.lat.size)
        self.assertTrue(centroids.area_pixel.size)
        self.assertTrue(centroids.on_land.size)

    def test_firms_resolution_pass(self):
        """ Test _firms_resolution """
        wf = WildFire()
        firms = wf._clean_firms_csv(TEST_FIRMS)
        self.assertAlmostEqual(wf._firms_resolution(firms), 0.375/ONE_LAT_KM)
        firms['instrument'][0] = 'MODIS'
        self.assertAlmostEqual(wf._firms_resolution(firms), 1.0/ONE_LAT_KM)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestMethodsFirms)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
