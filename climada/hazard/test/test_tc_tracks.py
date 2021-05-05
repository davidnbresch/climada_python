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

Test tc_tracks module.
"""

import unittest
import xarray as xr
import numpy as np
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString

import climada.hazard.tc_tracks as tc
from climada import CONFIG
from climada.util import ureg
from climada.util.constants import TC_ANDREW_FL
import climada.util.coordinates as u_coord
from climada.entity import Exposures

DATA_DIR = CONFIG.hazard.test_data.dir()
TEST_TRACK = DATA_DIR.joinpath("trac_brb_test.csv")
TEST_TRACK_SHORT = DATA_DIR.joinpath("trac_short_test.csv")
TEST_RAW_TRACK = DATA_DIR.joinpath('Storm.2016075S11087.ibtracs_all.v03r10.csv')
TEST_TRACK_GETTELMAN = DATA_DIR.joinpath('gettelman_test_tracks.nc')
TEST_TRACK_EMANUEL = DATA_DIR.joinpath('emanuel_test_tracks.mat')
TEST_TRACK_EMANUEL_CORR = DATA_DIR.joinpath('temp_mpircp85cal_full.mat')
TEST_TRACK_CHAZ = DATA_DIR.joinpath('chaz_test_tracks.nc')
TEST_TRACK_STORM = DATA_DIR.joinpath('storm_test_tracks.txt')
TEST_TRACKS_ANTIMERIDIAN = DATA_DIR.joinpath('tracks-antimeridian')


class TestIbtracs(unittest.TestCase):
    """Test reading and model of TC from IBTrACS files"""

    def test_raw_ibtracs_empty_pass(self):
        """Test reading TC from IBTrACS files"""
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1988234N13299')
        self.assertEqual(tc_track.get_track(), [])

    def test_write_read_pass(self):
        """Test writting and reading netcdf4 TCTracks instances"""
        path = DATA_DIR.joinpath("tc_tracks_nc")
        path.mkdir(exist_ok=True)
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1988234N13299',
                                     estimate_missing=True)
        tc_track.write_netcdf(str(path))

        tc_read = tc.TCTracks()
        tc_read.read_netcdf(str(path))

        self.assertEqual(tc_track.get_track().sid, tc_read.get_track().sid)

    def test_penv_rmax_penv_pass(self):
        """read_ibtracs_netcdf"""
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1992230N11325')
        penv_ref = np.ones(97) * 1010
        penv_ref[26:36] = [1011, 1012, 1013, 1014, 1015, 1014, 1014, 1014, 1014, 1012]

        self.assertTrue(np.allclose(
            tc_track.get_track().environmental_pressure.values, penv_ref))
        self.assertTrue(np.allclose(
            tc_track.get_track().radius_max_wind.values, np.zeros(97)))

    def test_read_raw_pass(self):
        """Read a tropical cyclone."""
        tc_track = tc.TCTracks()

        # read without specified provider or estimation of missing values
        tc_track.read_ibtracs_netcdf(storm_id='2017242N16333')
        track_ds = tc_track.get_track()
        self.assertEqual(len(tc_track.data), 1)
        self.assertEqual(track_ds.time.dt.year.values[0], 2017)
        self.assertEqual(track_ds.time.dt.month.values[0], 8)
        self.assertEqual(track_ds.time.dt.day.values[0], 30)
        self.assertEqual(track_ds.time.dt.hour.values[0], 0)
        self.assertAlmostEqual(track_ds.lat.values[0], 16.1, places=5)
        self.assertAlmostEqual(track_ds.lon.values[0], -26.9, places=5)
        self.assertAlmostEqual(track_ds.max_sustained_wind.values[0], 30)
        self.assertAlmostEqual(track_ds.central_pressure.values[0], 1008)
        self.assertAlmostEqual(track_ds.environmental_pressure.values[0], 1012)
        self.assertAlmostEqual(track_ds.radius_max_wind.values[0], 60)
        self.assertEqual(track_ds.time.size, 123)

        self.assertAlmostEqual(track_ds.lat.values[-1], 36.8, places=5)
        self.assertAlmostEqual(track_ds.lon.values[-1], -90.1, places=4)
        self.assertAlmostEqual(track_ds.central_pressure.values[-1], 1005)
        self.assertAlmostEqual(track_ds.max_sustained_wind.values[-1], 15)
        self.assertAlmostEqual(track_ds.environmental_pressure.values[-1], 1008)
        self.assertAlmostEqual(track_ds.radius_max_wind.values[-1], 60)

        self.assertFalse(np.isnan(track_ds.radius_max_wind.values).any())
        self.assertFalse(np.isnan(track_ds.environmental_pressure.values).any())
        self.assertFalse(np.isnan(track_ds.max_sustained_wind.values).any())
        self.assertFalse(np.isnan(track_ds.central_pressure.values).any())
        self.assertFalse(np.isnan(track_ds.lat.values).any())
        self.assertFalse(np.isnan(track_ds.lon.values).any())

        np.testing.assert_array_equal(track_ds.basin, 'NA')
        self.assertEqual(track_ds.max_sustained_wind_unit, 'kn')
        self.assertEqual(track_ds.central_pressure_unit, 'mb')
        self.assertEqual(track_ds.sid, '2017242N16333')
        self.assertEqual(track_ds.name, 'IRMA')
        self.assertEqual(track_ds.orig_event_flag, True)
        self.assertEqual(track_ds.data_provider, 'ibtracs_official_3h_mixed')
        self.assertEqual(track_ds.category, 5)

    def test_read_with_provider(self):
        """Read a tropical cyclone with and without explicit provider."""
        tc_track = tc.TCTracks()
        storm_id = '2012152N12130'

        tc_track.read_ibtracs_netcdf(storm_id=storm_id, provider='usa')
        track_ds = tc_track.get_track()
        self.assertEqual(track_ds.time.size, 51)
        self.assertEqual(track_ds.data_provider, 'ibtracs_usa')
        self.assertAlmostEqual(track_ds.lat.values[50], 34.3, places=5)
        self.assertAlmostEqual(track_ds.central_pressure.values[50], 989, places=5)
        self.assertAlmostEqual(track_ds.radius_max_wind.values[46], 20, places=5)

        tc_track.read_ibtracs_netcdf(storm_id=storm_id)
        track_ds = tc_track.get_track()
        self.assertEqual(track_ds.time.size, 35)
        self.assertEqual(track_ds.data_provider, 'ibtracs_official_3h_mixed')
        self.assertAlmostEqual(track_ds.lat.values[-1], 31.40, places=5)
        self.assertAlmostEqual(track_ds.central_pressure.values[-1], 980, places=5)

    def test_read_antimeridian(self):
        """Read a track that crosses the antimeridian and make sure that lon is consistent"""
        tc_track = tc.TCTracks()
        storm_id = '2013224N12220'

        # the officially responsible agencies 'usa' and 'tokyo' use different signs in lon, but we
        # have to `estimate_missing` because both have gaps in reported values
        tc_track.read_ibtracs_netcdf(storm_id=storm_id, provider=['official_3h'],
                                     estimate_missing=True)
        track_ds = tc_track.get_track()
        np.testing.assert_array_less(0, track_ds.lon)

    def test_read_estimate_missing(self):
        """Read a tropical cyclone and estimate missing values."""
        tc_track = tc.TCTracks()
        storm_id = '2012152N12130'

        tc_track.read_ibtracs_netcdf(storm_id=storm_id, estimate_missing=True)
        track_ds = tc_track.get_track()
        # less time steps are discarded, leading to a larger total size
        self.assertEqual(track_ds.time.size, 99)
        self.assertEqual(track_ds.data_provider, 'ibtracs_official_3h_mixed')
        self.assertAlmostEqual(track_ds.lat.values[44], 33.30, places=5)
        self.assertAlmostEqual(track_ds.central_pressure.values[44], 976, places=5)
        self.assertAlmostEqual(track_ds.central_pressure.values[42], 980, places=5)
        # the wind speed at position 44 is missing in the original data
        self.assertAlmostEqual(track_ds.max_sustained_wind.values[44], 58, places=0)
        self.assertAlmostEqual(track_ds.radius_oci.values[40], 160, places=0)
        # after position 42, ROCI is missing in the original data
        self.assertAlmostEqual(track_ds.radius_oci.values[42], 200, places=-1)
        self.assertAlmostEqual(track_ds.radius_oci.values[85], 165, places=-1)
        self.assertAlmostEqual(track_ds.radius_oci.values[95], 155, places=-1)

    def test_read_official(self):
        """Read a tropical cyclone, only officially reported values."""
        tc_track = tc.TCTracks()
        storm_id = '2012152N12130'

        tc_track.read_ibtracs_netcdf(
            storm_id=storm_id, interpolate_missing=False, provider='official')
        track_ds = tc_track.get_track()
        self.assertEqual(track_ds.time.size, 21)
        self.assertEqual(track_ds.data_provider, 'ibtracs_official')
        self.assertAlmostEqual(track_ds.lon.values[19], 137.6, places=4)
        self.assertAlmostEqual(track_ds.central_pressure.values[19], 980, places=5)
        np.testing.assert_array_equal(track_ds.radius_max_wind.values, 0)

    def test_read_scale_wind(self):
        """Read a tropical cyclone and scale wind speed according to agency."""
        tc_track = tc.TCTracks()
        storm_id = '2012152N12130'

        tc_track.read_ibtracs_netcdf(storm_id=storm_id, rescale_windspeeds=True)
        track_ds = tc_track.get_track()
        self.assertAlmostEqual(track_ds.max_sustained_wind.values[34], (55 - 23.3) / 0.6, places=5)

        tc_track.read_ibtracs_netcdf(storm_id=storm_id, rescale_windspeeds=False)
        track_ds = tc_track.get_track()
        self.assertAlmostEqual(track_ds.max_sustained_wind.values[34], 55, places=5)

    def test_read_interpolate_missing(self):
        """Read a tropical cyclone with and without interpolating missing values."""
        tc_track = tc.TCTracks()
        storm_id = '2010066S19050'

        tc_track.read_ibtracs_netcdf(storm_id=storm_id, interpolate_missing=False)
        track_ds = tc_track.get_track()
        self.assertEqual(track_ds.time.size, 50)
        self.assertAlmostEqual(track_ds.central_pressure.values[30], 992, places=5)
        self.assertAlmostEqual(track_ds.central_pressure.values[31], 1006, places=5)

        tc_track.read_ibtracs_netcdf(storm_id=storm_id, interpolate_missing=True)
        track_ds = tc_track.get_track()
        self.assertEqual(track_ds.time.size, 65)
        self.assertAlmostEqual(track_ds.central_pressure.values[30], 992, places=5)
        self.assertAlmostEqual(track_ds.central_pressure.values[38], 999, places=5)
        self.assertAlmostEqual(track_ds.central_pressure.values[46], 1006, places=5)

    def test_read_range(self):
        """Read several TCs."""
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id=None,
                                     year_range=(1915, 1916), basin='WP')
        self.assertEqual(tc_track.size, 0)

        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', year_range=(1993, 1994),
                                     basin='EP', estimate_missing=False)
        self.assertEqual(tc_track.size, 33)

        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', year_range=(1993, 1994),
                                     basin='EP', estimate_missing=True)
        self.assertEqual(tc_track.size, 45)

    def test_ibtracs_correct_pass(self):
        """Check estimate_missing option"""
        tc_try = tc.TCTracks()
        tc_try.read_ibtracs_netcdf(provider='usa', storm_id='1982267N25289',
                                   estimate_missing=True)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[0], 1013, places=0)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[5], 1008, places=0)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[-1], 1012, places=0)

    def test_read_with_basin(self):
        """Filter TCs by (genesis) basin."""
        # South Atlantic (not usually a TC location at all)
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(basin="SA")
        self.assertEqual(tc_track.size, 3)

        # the basin is not necessarily the genesis basin
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(year_range=(1995, 1995), basin="SP", estimate_missing=True)
        self.assertEqual(tc_track.size, 6)
        self.assertEqual(tc_track.data[0].basin[0], 'SP')
        self.assertEqual(tc_track.data[5].basin[0], 'SI')

        # genesis in NI
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(
            year_range=(1994, 1994), genesis_basin="NI", estimate_missing=True)
        self.assertEqual(tc_track.size, 5)
        for tr in tc_track.data:
            np.testing.assert_array_equal(tr.basin[0], "NI")

        # genesis in EP, but crosses WP at some point
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(year_range=(2002, 2003), basin="WP", genesis_basin="EP")
        self.assertEqual(tc_track.size, 3)
        for tr in tc_track.data:
            self.assertEqual(tr.basin[0], "EP")
            self.assertIn("WP", tr.basin)

class TestIO(unittest.TestCase):
    """Test reading of tracks from files of different formats"""

    def test_read_processed_ibtracs_csv(self):
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)

        self.assertEqual(tc_track.data[0].time.size, 38)
        self.assertEqual(tc_track.data[0].lon[11], -39.60)
        self.assertEqual(tc_track.data[0].lat[23], 14.10)
        self.assertEqual(tc_track.data[0].time_step[7], 6)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 55)
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[29], 976, places=0)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0].time.dt.year[13], 1951)
        self.assertEqual(tc_track.data[0].time.dt.month[26], 9)
        self.assertEqual(tc_track.data[0].time.dt.day[7], 29)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].sid, '1951239N12334')
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        np.testing.assert_array_equal(tc_track.data[0].basin, 'NA')
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

    def test_read_simulations_emanuel(self):
        tc_track = tc.TCTracks()

        tc_track.read_simulations_emanuel(TEST_TRACK_EMANUEL, hemisphere='N')
        self.assertEqual(len(tc_track.data), 4)
        self.assertEqual(tc_track.data[0].time.size, 93)
        self.assertEqual(tc_track.data[0].lon[11], -115.57)
        self.assertEqual(tc_track.data[0].lat[23], 10.758)
        self.assertEqual(tc_track.data[0].time_step[7], 2)
        self.assertAlmostEqual(tc_track.data[0].radius_max_wind[15], 44.27645788336934)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 27.1)
        self.assertEqual(tc_track.data[0].central_pressure[29], 995.31)
        self.assertTrue(np.all(tc_track.data[0].environmental_pressure == 1010))
        self.assertTrue(np.all(tc_track.data[0].time.dt.year == 1950))
        self.assertEqual(tc_track.data[0].time.dt.month[26], 10)
        self.assertEqual(tc_track.data[0].time.dt.day[7], 26)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].sid, '1')
        self.assertEqual(tc_track.data[0].name, '1')
        self.assertTrue(np.all([np.all(d.basin == 'N') for d in tc_track.data]))
        self.assertEqual(tc_track.data[0].category, 3)

        tc_track.read_simulations_emanuel(TEST_TRACK_EMANUEL_CORR)
        self.assertEqual(len(tc_track.data), 2)
        self.assertTrue(np.all([np.all(d.basin == 'S') for d in tc_track.data]))
        self.assertEqual(tc_track.data[0].radius_max_wind[15], 102.49460043196545)
        self.assertEqual(tc_track.data[0].time.dt.month[343], 2)
        self.assertEqual(tc_track.data[0].time.dt.day[343], 28)
        self.assertEqual(tc_track.data[0].time.dt.month[344], 3)
        self.assertEqual(tc_track.data[0].time.dt.day[344], 1)
        self.assertEqual(tc_track.data[1].time.dt.year[0], 2009)
        self.assertEqual(tc_track.data[1].time.dt.year[256], 2009)
        self.assertEqual(tc_track.data[1].time.dt.year[257], 2010)
        self.assertEqual(tc_track.data[1].time.dt.year[-1], 2010)

    def test_read_one_gettelman(self):
        """Test reading and model of TC from Gettelman track files"""
        tc_track_G = tc.TCTracks()
        # populate tracks by loading data from NetCDF:
        nc_data = nc.Dataset(TEST_TRACK_GETTELMAN)
        nstorms = nc_data.dimensions['storm'].size
        for i in range(nstorms):
            tc_track_G.read_one_gettelman(nc_data, i)

        self.assertEqual(tc_track_G.data[0].time.size, 29)
        self.assertEqual(tc_track_G.data[0].lon[11], 60.0)
        self.assertEqual(tc_track_G.data[0].lat[23], 10.20860481262207)
        self.assertEqual(tc_track_G.data[0].time_step[7], 3.)
        self.assertEqual(np.max(tc_track_G.data[0].radius_max_wind), 65)
        self.assertEqual(np.min(tc_track_G.data[0].radius_max_wind), 65)
        self.assertEqual(tc_track_G.data[0].max_sustained_wind[21], 39.91877223718089)
        self.assertEqual(tc_track_G.data[0].central_pressure[27], 1005.969482421875)
        self.assertEqual(np.max(tc_track_G.data[0].environmental_pressure), 1015)
        self.assertEqual(np.min(tc_track_G.data[0].environmental_pressure), 1015)
        self.assertEqual(tc_track_G.data[0].maximum_precipitation[14], 219.10108947753906)
        self.assertEqual(tc_track_G.data[0].average_precipitation[12], 101.43893432617188)
        self.assertEqual(tc_track_G.data[0].time.dt.year[13], 1979)
        self.assertEqual(tc_track_G.data[0].time.dt.month[26], 1)
        self.assertEqual(tc_track_G.data[0].time.dt.day[7], 2)
        self.assertEqual(tc_track_G.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track_G.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track_G.data[0].sid, '0')
        self.assertEqual(tc_track_G.data[0].name, '0')
        np.testing.assert_array_equal(tc_track_G.data[0].basin, 'NI')
        self.assertEqual(tc_track_G.data[0].category, 0)

    def test_read_simulations_chaz(self):
        """Test reading NetCDF output from CHAZ simulations"""
        tc_track = tc.TCTracks()

        tc_track.read_simulations_chaz(TEST_TRACK_CHAZ)
        self.assertEqual(len(tc_track.data), 13)
        self.assertEqual(tc_track.data[0].time.size, 5)
        self.assertEqual(tc_track.data[0].lon[3], 74.1388328911036)
        self.assertEqual(tc_track.data[0].lat[4], -9.813585651475156)
        self.assertEqual(tc_track.data[0].time_step[3], 6)
        self.assertEqual(tc_track.data[0].max_sustained_wind[2], 20.188325232226354)
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[1], 1008, places=0)
        self.assertTrue(np.all(tc_track.data[0].time.dt.year == 1991))
        self.assertEqual(tc_track.data[0].time.dt.month[2], 1)
        self.assertEqual(tc_track.data[0].time.dt.day[3], 15)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].sid, 'chaz_test_tracks.nc-1-0')
        self.assertEqual(tc_track.data[0].name, 'chaz_test_tracks.nc-1-0')
        self.assertTrue(np.all([np.all(d.basin == 'GB') for d in tc_track.data]))
        self.assertEqual(tc_track.data[4].category, 0)
        self.assertEqual(tc_track.data[3].category, -1)

        tc_track.read_simulations_chaz(TEST_TRACK_CHAZ, year_range=(1990, 1991))
        self.assertEqual(len(tc_track.data), 3)

        tc_track.read_simulations_chaz(TEST_TRACK_CHAZ, year_range=(1950, 1955))
        self.assertEqual(len(tc_track.data), 0)

        tc_track.read_simulations_chaz(TEST_TRACK_CHAZ, ensemble_nums=[0, 2])
        self.assertEqual(len(tc_track.data), 9)

    def test_read_simulations_storm(self):
        """Test reading NetCDF output from STORM simulations"""
        tc_track = tc.TCTracks()

        tc_track.read_simulations_storm(TEST_TRACK_STORM)
        self.assertEqual(len(tc_track.data), 6)
        self.assertEqual(tc_track.data[0].time.size, 15)
        self.assertEqual(tc_track.data[0].lon[3], 245.3)
        self.assertEqual(tc_track.data[0].lat[4], 11.9)
        self.assertEqual(tc_track.data[0].time_step[3], 3)
        self.assertEqual(tc_track.data[0].max_sustained_wind[2], 37.127429805615556)
        self.assertEqual(tc_track.data[0].radius_max_wind[5], 19.07407454551836)
        self.assertEqual(tc_track.data[0].central_pressure[1], 999.4)
        self.assertTrue(np.all(tc_track.data[0].time.dt.year == 1980))
        self.assertEqual(tc_track.data[0].time.dt.month[2].item(), 6)
        self.assertEqual(tc_track.data[0].time.dt.day[3].item(), 1)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].sid, 'storm_test_tracks.txt-0-0')
        self.assertEqual(tc_track.data[0].name, 'storm_test_tracks.txt-0-0')
        self.assertTrue(np.all([np.all(d.basin == 'EP') for d in tc_track.data]))
        self.assertEqual(tc_track.data[4].category, 0)
        self.assertEqual(tc_track.data[3].category, 1)

        tc_track.read_simulations_storm(TEST_TRACK_STORM, years=[0, 2])
        self.assertEqual(len(tc_track.data), 4)

        tc_track.read_simulations_storm(TEST_TRACK_STORM, years=[7])
        self.assertEqual(len(tc_track.data), 0)

    def test_to_geodataframe_points(self):
        """Conversion of TCTracks to GeoDataFrame using Points."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)

        gdf_points = tc_track.to_geodataframe(as_points=True)
        self.assertIsInstance(gdf_points.unary_union.bounds, tuple)
        self.assertEqual(gdf_points.shape[0], len(tc_track.data[0].time))
        self.assertEqual(gdf_points.shape[1], len(tc_track.data[0].variables)+len(tc_track.data[0].attrs)-1)
        self.assertAlmostEqual(gdf_points.buffer(3).unary_union.area, 348.79972062947854)
        self.assertIsInstance(gdf_points.iloc[0].time, pd._libs.tslibs.timestamps.Timestamp)

    def test_to_geodataframe_line(self):
        """Conversion of TCTracks to GeoDataFrame using LineStrings."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)

        gdf_line = tc_track.to_geodataframe()
        self.assertEqual(gdf_line.shape[0], 1)
        self.assertAlmostEqual(gdf_line.geometry[0].length, 54.0634224372971)
        self.assertIsInstance(gdf_line.bounds.minx, pd.core.series.Series)

        anti_track = tc.TCTracks()
        # test data set with two tracks:
        # * 1980052S16155: crosses the antimeridian
        # * 2018079S09162: close, but doesn't cross antimeridian; has self-intersections
        anti_track.read_netcdf(TEST_TRACKS_ANTIMERIDIAN)

        split = anti_track.to_geodataframe(split_lines_antimeridian=True)
        split.set_index('sid', inplace=True)
        self.assertIsInstance(split.loc['1980052S16155'].geometry, MultiLineString)
        self.assertIsInstance(split.loc['2018079S09162'].geometry, LineString)
        self.assertEqual(len(split.loc['1980052S16155'].geometry), 8)
        self.assertFalse(split.loc['2018079S09162'].geometry.is_simple)

        nosplit = anti_track.to_geodataframe(split_lines_antimeridian=False)
        nosplit.set_index('sid', inplace=True)
        self.assertIsInstance(nosplit.loc['1980052S16155'].geometry, LineString)
        self.assertIsInstance(nosplit.loc['2018079S09162'].geometry, LineString)
        self.assertFalse(nosplit.loc['2018079S09162'].geometry.is_simple)

        np.testing.assert_array_almost_equal(
            split.loc['1980052S16155'].geometry.bounds, (-180, -38.583244, 180, -17))
        np.testing.assert_array_almost_equal(
            nosplit.loc['1980052S16155'].geometry.bounds, (150.800003, -38.583244, 185, -17))
        np.testing.assert_array_almost_equal(
            split.loc['2018079S09162'].geometry.bounds, (148.600006, -22.41, 162.399994, -13.4))
        np.testing.assert_array_almost_equal(
            nosplit.loc['2018079S09162'].geometry.bounds, (148.600006, -22.41, 162.399994, -13.4))

class TestFuncs(unittest.TestCase):
    """Test functions over TC tracks"""

    def test_get_track_pass(self):
        """Test get_track."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        self.assertIsInstance(tc_track.get_track(), xr.Dataset)
        self.assertIsInstance(tc_track.get_track('1951239N12334'), xr.Dataset)

        tc_track_bis = tc.TCTracks()
        tc_track_bis.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_track.append(tc_track_bis)
        self.assertIsInstance(tc_track.get_track(), list)
        self.assertIsInstance(tc_track.get_track('1951239N12334'), xr.Dataset)

    def test_subset(self):
        """Test subset."""
        storms = ['1988169N14259', '2002073S16161', '2002143S07157']
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(storm_id=storms)
        self.assertEqual(tc_track.subset({'basin': 'SP'}).size, 2)

    def test_get_extent(self):
        """Test extent/bounds attributes."""
        storms = ['1988169N14259', '2002073S16161', '2002143S07157']
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(storm_id=storms, provider=["usa", "bom"])
        bounds = (153.585022, -23.200001, 258.714996, 17.514986)
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])
        bounds_buf = (153.485022, -23.300001, 258.814996, 17.614986)
        np.testing.assert_array_almost_equal(tc_track.bounds, bounds)
        np.testing.assert_array_almost_equal(tc_track.get_bounds(deg_buffer=0.1), bounds_buf)
        np.testing.assert_array_almost_equal(tc_track.extent, extent)

    def test_interp_track_pass(self):
        """Interpolate track to min_time_step. Compare to MATLAB reference."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep(time_step_h=1)

        self.assertEqual(tc_track.data[0].time.size, 223)
        self.assertAlmostEqual(tc_track.data[0].lon.values[11], -27.426151640151684)
        self.assertAlmostEqual(float(tc_track.data[0].lat[23]), 12.300006169591480)
        self.assertEqual(tc_track.data[0].time_step[7], 1)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 25)
        self.assertTrue(np.isfinite(tc_track.data[0].central_pressure.values).all())
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[29], 1008, places=0)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0]['time.year'][13], 1951)
        self.assertEqual(tc_track.data[0]['time.month'][26], 8)
        self.assertEqual(tc_track.data[0]['time.day'][7], 27)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        np.testing.assert_array_equal(tc_track.data[0].basin, 'NA')
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

        # test some "generic floats"
        for time_step_h in [0.6663545049172093, 2.509374054925788, 8.175754471661111]:
            # artifically create data that doesn't start at full hour
            for loffset in [0, 22, 30]:
                tc_track = tc.TCTracks()
                tc_track.read_processed_ibtracs_csv(TEST_TRACK)
                tc_track.data[0].time.values[:] += np.timedelta64(loffset, "m")
                tc_track.equal_timestep(time_step_h=time_step_h)
                np.testing.assert_array_equal(tc_track.data[0].time_step, time_step_h)
                self.assertTrue(np.isfinite(tc_track.data[0].central_pressure.values).all())

        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep(time_step_h=0.16667)

        self.assertEqual(tc_track.data[0].time.size, 1332)
        self.assertTrue(np.all(tc_track.data[0].time_step == 0.16667))
        self.assertTrue(np.isfinite(tc_track.data[0].central_pressure.values).all())
        self.assertAlmostEqual(tc_track.data[0].lon.values[65], -27.397636528537127)

        for time_step_h in [0, -0.5, -1]:
            tc_track = tc.TCTracks()
            tc_track.read_processed_ibtracs_csv(TEST_TRACK)
            msg = f"time_step_h is not a positive number: {time_step_h}"
            with self.assertRaises(ValueError, msg=msg) as _cm:
                tc_track.equal_timestep(time_step_h=time_step_h)

    def test_interp_origin_pass(self):
        """Interpolate track to min_time_step crossing lat origin"""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.data[0].lon.values = np.array([
            167.207761, 168.1, 168.936535, 169.728947, 170.5, 171.257176,
            171.946822, 172.5, 172.871797, 173.113396, 173.3, 173.496375,
            173.725522, 174., 174.331591, 174.728961, 175.2, 175.747632,
            176.354929, 177., 177.66677, 178.362433, 179.1, 179.885288,
            -179.304661, -178.5, -177.726442, -176.991938, -176.3, -175.653595,
            -175.053513, -174.5, -173.992511, -173.527342, -173.1, -172.705991,
            -172.340823, -172.
        ])
        tc_track.data[0].lat.values = np.array([
            40.196053, 40.6, 40.930215, 41.215674, 41.5, 41.816354, 42.156065,
            42.5, 42.833998, 43.16377, 43.5, 43.847656, 44.188854, 44.5,
            44.764269, 44.991925, 45.2, 45.402675, 45.602707, 45.8, 45.995402,
            46.193543, 46.4, 46.615718, 46.82312, 47., 47.130616, 47.225088,
            47.3, 47.369224, 47.435786, 47.5, 47.562858, 47.628064, 47.7,
            47.783047, 47.881586, 48.
        ])
        tc_track.equal_timestep(time_step_h=1)

        self.assertEqual(tc_track.data[0].time.size, 223)
        self.assertAlmostEqual(tc_track.data[0].lon.values[0], 167.207761)
        self.assertAlmostEqual(tc_track.data[0].lon.values[-1], -172)
        self.assertAlmostEqual(tc_track.data[0].lon.values[137], 179.75187272)
        self.assertAlmostEqual(tc_track.data[0].lon.values[138], 179.885288)
        self.assertAlmostEqual(tc_track.data[0].lon.values[139], -179.98060885)
        self.assertAlmostEqual(tc_track.data[0].lon.values[140], -179.84595743)
        self.assertAlmostEqual(tc_track.data[0].lat.values[0], 40.196053)
        self.assertAlmostEqual(tc_track.data[0].lat.values[-1], 48.)
        self.assertEqual(tc_track.data[0].time_step[7], 1)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 25)
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[29], 1008, places=0)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0]['time.year'][13], 1951)
        self.assertEqual(tc_track.data[0]['time.month'][26], 8)
        self.assertEqual(tc_track.data[0]['time.day'][7], 27)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        np.testing.assert_array_equal(tc_track.data[0].basin, 'NA')
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

    def test_interp_origin_inv_pass(self):
        """Interpolate track to min_time_step crossing lat origin"""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.data[0].lon.values = np.array([
            167.207761, 168.1, 168.936535, 169.728947, 170.5, 171.257176,
            171.946822, 172.5, 172.871797, 173.113396, 173.3, 173.496375,
            173.725522, 174., 174.331591, 174.728961, 175.2, 175.747632,
            176.354929, 177., 177.66677, 178.362433, 179.1, 179.885288,
            -179.304661, -178.5, -177.726442, -176.991938, -176.3, -175.653595,
            -175.053513, -174.5, -173.992511, -173.527342, -173.1, -172.705991,
            -172.340823, -172.
        ])
        tc_track.data[0].lon.values = - tc_track.data[0].lon.values
        tc_track.data[0].lat.values = np.array([
            40.196053, 40.6, 40.930215, 41.215674, 41.5, 41.816354, 42.156065,
            42.5, 42.833998, 43.16377, 43.5, 43.847656, 44.188854, 44.5,
            44.764269, 44.991925, 45.2, 45.402675, 45.602707, 45.8, 45.995402,
            46.193543, 46.4, 46.615718, 46.82312, 47., 47.130616, 47.225088,
            47.3, 47.369224, 47.435786, 47.5, 47.562858, 47.628064, 47.7,
            47.783047, 47.881586, 48.
        ])
        tc_track.equal_timestep(time_step_h=1)

        self.assertEqual(tc_track.data[0].time.size, 223)
        self.assertAlmostEqual(tc_track.data[0].lon.values[0], -167.207761)
        self.assertAlmostEqual(tc_track.data[0].lon.values[-1], 172)
        self.assertAlmostEqual(tc_track.data[0].lon.values[137], -179.75187272)
        self.assertAlmostEqual(tc_track.data[0].lon.values[138], -179.885288)
        self.assertAlmostEqual(tc_track.data[0].lon.values[139], 179.98060885)
        self.assertAlmostEqual(tc_track.data[0].lon.values[140], 179.84595743)
        self.assertAlmostEqual(tc_track.data[0].lat.values[0], 40.196053)
        self.assertAlmostEqual(tc_track.data[0].lat.values[-1], 48.)
        self.assertEqual(tc_track.data[0].time_step[7], 1)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 25)
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[29], 1008, places=0)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0]['time.year'][13], 1951)
        self.assertEqual(tc_track.data[0]['time.month'][26], 8)
        self.assertEqual(tc_track.data[0]['time.day'][7], 27)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        np.testing.assert_array_equal(tc_track.data[0].basin, 'NA')
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

    def test_dist_since_lf_pass(self):
        """Test _dist_since_lf for andrew tropical cyclone."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        track = tc_track.get_track()
        track['on_land'] = ('time', u_coord.coord_on_land(track.lat.values, track.lon.values))
        track['dist_since_lf'] = ('time', tc._dist_since_lf(track))

        msk = ~track.on_land
        self.assertTrue(np.all(np.isnan(track.dist_since_lf.values[msk])))
        self.assertEqual(track.dist_since_lf.values[msk].size, 38)

        self.assertGreater(
            track.dist_since_lf.values[-1],
            u_coord.dist_to_coast(track.lat.values[-1], track.lon.values[-1]) / 1000)
        self.assertEqual(1020.5431562223974, track['dist_since_lf'].values[-1])

        # check distances on land always increase, in second landfall
        dist_on_land = track.dist_since_lf.values[track.on_land]
        self.assertTrue(np.all(np.diff(dist_on_land)[1:] > 0))

    def test_category_pass(self):
        """Test category computation."""
        max_sus_wind = np.array([25, 30, 35, 40, 45, 45, 45, 45, 35, 25])
        max_sus_wind_unit = 'kn'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([25, 25, 25, 30, 30, 30, 30, 30, 25, 25, 20])
        max_sus_wind_unit = 'kn'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([80, 90, 100, 115, 120, 125, 130,
                                 120, 110, 80, 75, 80, 65])
        max_sus_wind_unit = 'kn'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

        max_sus_wind = np.array([
            28.769475, 34.52337, 40.277265, 46.03116, 51.785055, 51.785055,
            51.785055, 51.785055, 40.277265, 28.769475
        ])
        max_sus_wind_unit = 'mph'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([
            12.86111437, 12.86111437, 12.86111437, 15.43333724, 15.43333724,
            15.43333724, 15.43333724, 15.43333724, 12.86111437, 12.86111437,
            10.2888915
        ])
        max_sus_wind_unit = 'm/s'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([
            148.16, 166.68, 185.2, 212.98, 222.24, 231.5, 240.76, 222.24,
            203.72, 148.16, 138.9, 148.16, 120.38
        ])
        max_sus_wind_unit = 'km/h'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

    def test_estimate_params_pass(self):
        """Test track parameter estimation functions."""
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        v_max = np.array([45, np.nan, 50, 55, 0, 60, 75])
        lat = np.array([13.8, 13.9, 14, 14.1, 14.1, np.nan, -999])
        lon = np.array([np.nan, -52.8, -54.4, -56, -58.4, -59.7, -61.1])
        ref_pres = np.array([np.nan, 993, 990, 986, np.nan, 1004, np.nan])
        out_pres = tc._estimate_pressure(cen_pres, lat, lon, v_max)
        np.testing.assert_array_almost_equal(ref_pres, out_pres, decimal=0)

        v_max = np.array([45, np.nan, 50, 55, 0, 60, 75])
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        lat = np.array([13.8, 13.9, 14, 14.1, 14.1, np.nan, -999])
        lon = np.array([np.nan, -52.8, -54.4, -56, -58.4, -59.7, -61.1])
        ref_vmax = np.array([45, 46, 50, 55, np.nan, 60, 75])
        out_vmax = tc._estimate_vmax(v_max, lat, lon, cen_pres)
        np.testing.assert_array_almost_equal(ref_vmax, out_vmax, decimal=0)

        roci = np.array([np.nan, -1, 145, 170, 180, 0, -5])
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        ref_roci = np.array([np.nan, 182.792715, 145, 170, 180, 161.5231086, np.nan])
        out_roci = tc.estimate_roci(roci, cen_pres)
        np.testing.assert_array_almost_equal(ref_roci, out_roci)

        rmw = np.array([17, 33, -1, 25, np.nan, -5, 13])
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        ref_rmw = np.array([17, 33, np.nan, 25, np.nan, 43.95543761, 13])
        out_rmw = tc.estimate_rmw(rmw, cen_pres)
        np.testing.assert_array_almost_equal(ref_rmw, out_rmw)

    def test_estimate_rmw_pass(self):
        """Test estimate_rmw function."""
        NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude

        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        rad_max_wind = tc.estimate_rmw(
            tc_track.data[0].radius_max_wind.values,
            tc_track.data[0].central_pressure.values) * NM_TO_KM

        self.assertAlmostEqual(rad_max_wind[0], 87, places=0)
        self.assertAlmostEqual(rad_max_wind[10], 87, places=0)
        self.assertAlmostEqual(rad_max_wind[128], 56, places=0)
        self.assertAlmostEqual(rad_max_wind[129], 55, places=0)
        self.assertAlmostEqual(rad_max_wind[130], 54, places=0)
        self.assertAlmostEqual(rad_max_wind[189], 53, places=0)
        self.assertAlmostEqual(rad_max_wind[190], 55, places=0)
        self.assertAlmostEqual(rad_max_wind[191], 57, places=0)
        self.assertAlmostEqual(rad_max_wind[192], 58, places=0)
        self.assertAlmostEqual(rad_max_wind[200], 71, places=0)

    def test_tracks_in_exp_pass(self):
        """Check if tracks in exp are filtered correctly"""

        # Load two tracks from ibtracks
        storms = {'in': '2000233N12316', 'out': '2000160N21267'}
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(storm_id=list(storms.values()))

        # Define exposure from geopandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        exp_world = Exposures(world)
        exp = Exposures(exp_world.gdf[exp_world.gdf.name=='Cuba'])

        # Compute tracks in exp
        tracks_in_exp = tc_track.tracks_in_exp(exp.gdf, buffer=1.0)

        self.assertTrue(tracks_in_exp.get_track(storms['in']))
        self.assertFalse(tracks_in_exp.get_track(storms['out']))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIO))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIbtracs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
