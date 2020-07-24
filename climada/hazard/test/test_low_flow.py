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

Test low flow module.
"""
import os
import unittest
import numpy as np
import pandas as pd
import datetime as dt

from climada.hazard.low_flow import LowFlow, unique_clusters, _compute_threshold_grid
from climada.util.constants import DATA_DIR
from climada.hazard.centroids import Centroids

INPUT_DIR = os.path.join(DATA_DIR, 'demo')
FN_STR_DEMO = 'co2_dis_global_daily_DEMO_FR'


def init_test_data_unique_clusters():
    """creates sandbox test data for 2D cluster IDs for test of identification of
    unique 3D clusters"""

    df = pd.DataFrame(columns=['target_cluster', 'cluster_id', 'c_lat_lon',
                               'c_lat_dt_month', 'c_lon_dt_month'])

    df.c_lon_dt_month = np.array([1, 1, 1, 1, 2, 2, 3, 4, 5, 4, 4, 5, 6, -1, -1])
    df.c_lat_dt_month = np.array([1, -1, 2, 2, 2, 3, 5, 3, 4, 6, 6, 5, 7, -1, 1])
    df.c_lat_lon = np.array([1, 3, 1, 3, 3, 3, 5, 3, 5, 3, 4, 5, 2, -1, -1])
    df.target_cluster = [1, -1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 3, -1, -1]
    df.cluster_id = np.zeros(len(df.target_cluster), dtype=int) - 1
    return df

def init_test_data_clustering():
    """creates sandbox test data for monthly days below threshold data
    for testing clustering"""

    df = pd.DataFrame(columns=['lat', 'lon', 'ndays',
                               'dt_month', 'target_cluster'])

    df.lat = np.array([-0, -0, -.5, -.5, -1, -.5, -1, -0, -.5, -1, -1, -1.5, -2.5])
    df.lon = np.array([0, 1, 0, 1.5, 2, 0, 0, 1, 1.5, 0, 2, 0, 2.5])
    df.dt_month = np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3])
    df['dtime'] = df['dt_month'].apply(lambda x: dt.datetime.toordinal(dt.datetime(1,x,1)))
    df.ndays = [5, 11, 5, 11, 11, 10, 10, 22, 22, 20, 22, 20, 1]

    df['iter_ev'] = np.ones(len(df), bool)
    df['cons_id'] = np.zeros(len(df), int) - 1
    return df


class TestLowFlowDummyData(unittest.TestCase):
    """Test for defining low flow event from dummy processed discharge data"""

    def test_unique_clusters(self):
        """Test unique_clusters:
            unique 3D cluster identification from 2D cluster data"""
        data = init_test_data_unique_clusters()
        data = unique_clusters(data)
        self.assertEqual(data.size, 75)
        self.assertListEqual(list(data.cluster_id), list(data.target_cluster))

    def test_identify_clusters_default(self):
        """Test identify_clusters:
            clustering event from monthly days below threshold data"""
        haz = LowFlow()
        # 1) direct neighbors only (allowing over cross in space):
        haz.data = init_test_data_clustering()
        haz.identify_clusters(clus_thres_xy=1.5, clus_thresh_t=1, min_samples=1)
        target_cluster = [1, 2, 1, 2, 2, 1, 1, 3, 3, 1, 3, 1, 4]
        self.assertListEqual(list(haz.data.cluster_id), target_cluster)

        # as (1), but allowing 1 month break in between:
        haz.data = init_test_data_clustering()
        haz.identify_clusters(clus_thres_xy=1.5, clus_thresh_t=2, min_samples=1)
        target_cluster = [1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 3]
        self.assertListEqual(list(haz.data.cluster_id), target_cluster)

        # as (1), but allowing 1 gridcell break in between:
        haz.data = init_test_data_clustering()
        haz.identify_clusters(clus_thres_xy=2., clus_thresh_t=1, min_samples=1)
        target_cluster = [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 3]
        self.assertListEqual(list(haz.data.cluster_id), target_cluster)

    def test_events_from_clusters(self):
        """Test events_from_clusters: creation of events and computation of intensity based on clusters,
        requires: identify_clusters, Centroids, also tests correct intensity sum"""
        haz = LowFlow()
        haz.data = init_test_data_clustering()
        haz.identify_clusters(clus_thres_xy=1.5, clus_thresh_t=1, min_samples=1)

        # define centroids for test data:
        centroids = Centroids()
        grid = np.meshgrid(np.arange(haz.data.lat.min(), haz.data.lat.max()+.5, .5),
                           np.arange(haz.data.lon.min(), haz.data.lon.max()+.5, .5))
        lat = list()
        lon = list()
        for arrlat, arrlon in zip(list(grid[0]), list(grid[1])):
            lat += list(arrlat)
            lon += list(arrlon)
        centroids.set_lat_lon(np.array(lat), np.array(lon))
        centroids.set_lat_lon_to_meta()

        haz.events_from_clusters(centroids)
        target_intensity_e1 = [ 0.,  0., 20., 30., 15.,  5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
        self.assertEqual(haz.intensity.size, 11)
        self.assertEqual(haz.intensity.todense().size, 144)
        self.assertEqual(haz.intensity.sum(), 170.)
        self.assertListEqual(list(np.array(haz.intensity.todense()[0])[0]), target_intensity_e1)

class TestLowFlowNETCDF(unittest.TestCase):
    """Test for defining low flow event from discharge data file"""

    haz = LowFlow()
    haz.set_from_nc(input_dir=INPUT_DIR, percentile=2.5,
                    yearrange=(2001, 2005), yearrange_ref=(2001, 2005),
                    gh_model='h08', cl_model='gfdl-esm2m',
                    scenario='historical', scenario_ref='historical', soc='histsoc',
                    soc_ref='histsoc', fn_str_var=FN_STR_DEMO, keep_dis_data=True)

    def test_load_FR_all(self, haz=haz):
        """Test defining low flow hazard from complete demo file (France)
        and keep monthly data"""
        self.assertEqual(haz.data.shape[0], 1653)
        self.assertEqual(haz.data.shape[1], 14)
        self.assertEqual(haz.data.ndays.max(), 31.0)
        self.assertAlmostEqual(haz.data.ndays.mean(), 10.588021778584393)
        self.assertAlmostEqual(haz.data.relative_dis.max(), 0.41278067)
        self.assertEqual(haz.centroids.lon.min(), -4.75)
        self.assertEqual(haz.centroids.lon.max(), 8.25)
        self.assertEqual(haz.centroids.lat.min(), 42.25)
        self.assertEqual(haz.centroids.lat.max(), 51.25)
        self.assertEqual(haz.intensity.shape, (66, 513))
        self.assertEqual(haz.event_id.size, 66)
        self.assertEqual(haz.intensity.max(), 46.0)
        self.assertEqual(haz.data.cluster_id.unique().size, haz.event_id.size)

    def test_combine_nc(self, haz=haz):
        """test if the hazard is the same if defined from combining chunked data files"""

        haz2 = LowFlow()
        haz2.set_from_nc(input_dir=INPUT_DIR, percentile=2.5,
                         yearrange=(2001, 2005), yearrange_ref=(2001, 2005),
                         gh_model='h08', cl_model='gfdl-esm2m',
                         scenario='historical', scenario_ref='historical', soc='histsoc',
                         soc_ref='histsoc', fn_str_var=FN_STR_DEMO, keep_dis_data=True,
                         yearchunks=['2001_2003', '2004_2005'])

        self.assertEqual(haz.data.shape[0], haz2.data.shape[0])
        self.assertEqual(haz.data.shape[1], haz2.data.shape[1], )
        self.assertEqual(haz.data.ndays.max(), haz2.data.ndays.max())
        self.assertAlmostEqual(haz.data.ndays.mean(), haz2.data.ndays.mean())
        self.assertAlmostEqual(haz.data.relative_dis.max(), haz2.data.relative_dis.max())
        self.assertAlmostEqual(0.41278067, haz2.data.relative_dis.max(), places=5)
        self.assertEqual(haz.centroids.lon.min(), haz2.centroids.lon.min())
        self.assertEqual(haz.centroids.lon.max(), haz2.centroids.lon.max())
        self.assertEqual(haz.centroids.lat.min(), haz2.centroids.lat.min())
        self.assertEqual(haz.centroids.lat.max(), haz2.centroids.lat.max())
        self.assertEqual(haz.intensity.shape, haz2.intensity.shape)
        self.assertEqual(haz.event_id.size, haz2.event_id.size)
        self.assertEqual(haz.intensity.max(), haz2.intensity.max())
        self.assertEqual(haz.data.cluster_id.unique().size, haz2.data.cluster_id.unique().size)

    def test_filter_events(self):
        """test if the right events are being filtered out"""
        haz = LowFlow()
        haz.set_from_nc(input_dir=INPUT_DIR, percentile=2.5, min_intensity=10,
                        min_number_cells=10, min_days_per_month=10,
                        yearrange=(2001, 2005), yearrange_ref=(2001, 2005),
                        gh_model='h08', cl_model='gfdl-esm2m',
                        scenario='historical', scenario_ref='historical', soc='histsoc',
                        soc_ref='histsoc', fn_str_var=FN_STR_DEMO, keep_dis_data=True)
        self.assertGreaterEqual(haz.data.ndays.min(), 10)
        self.assertGreaterEqual(haz.intensity[haz.intensity != 0].min(), 10)
        for event in range(haz.intensity.shape[0]):
            self.assertGreaterEqual(haz.intensity[event, :].nnz, 10)

class TestDischargeDataHandling(unittest.TestCase):
    """test additiopnal functions in low_flow and required for class LowFlow reading and
    processing ISIMIP input data with variable discharge (dis)"""

    def test_compute_threshold_grid(self):
        """test computation of percentile and mean on grid and masking of area"""
        perc_data, mean_data = _compute_threshold_grid(5, (2001, 2005), INPUT_DIR, 'h08', 'gfdl-esm2m',
                            'historical', 'histsoc', FN_STR_DEMO, None,
                            ['2001_2005'], mask_threshold=None)
        perc_data_mask, mean_data_mask = _compute_threshold_grid(5, (2001, 2005), INPUT_DIR, 'h08', 'gfdl-esm2m',
                            'historical', 'histsoc', FN_STR_DEMO, None,
                            ['2001_2005'], mask_threshold=('mean', 1500))
        self.assertLess(np.sum(mean_data_mask.dis>0).data.max(), np.sum(mean_data.dis>0).data.max())
        self.assertEqual(np.sum(mean_data.dis>0).data.max(), 417)
        self.assertEqual(np.sum(mean_data_mask.dis>0).data.max(), 10)
        self.assertEqual(np.sum(perc_data.dis>0).data.max(), 392)
        self.assertEqual(np.sum(perc_data_mask.dis>0).data.max(), 10)
        self.assertListEqual(list(perc_data_mask.lon.data), list(perc_data.lon.data))
        self.assertEqual(len(perc_data_mask.lon.data), 27)
        self.assertEqual(max(perc_data_mask.lon.data), 8.25)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLowFlowDummyData)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLowFlowNETCDF))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDischargeDataHandling))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
