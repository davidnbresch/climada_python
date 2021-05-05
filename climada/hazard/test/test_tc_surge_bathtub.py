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

Test tc_surge_bathtub module
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from climada.hazard import Centroids, TCTracks, TropCyclone
from climada.hazard.tc_surge_bathtub import _fraction_on_land, TCSurgeBathtub


class tmp_artifical_topo(object):
    """Context manager for a temporary artificial elevation dataset (given as path)."""

    def __init__(self, bounds, res_deg):
        """Read distance to coast values from raster file for later use.

        Parameters
        ----------
        bounds : tuple (lon_min, lat_min, lon_max, lat_max)
            Coordinates of bounding box
        res_deg : float
            Resolution in degrees
        """
        lat = np.arange(bounds[3] - 0.5 * res_deg, bounds[1], -res_deg)
        lon = np.arange(bounds[0] + 0.5 * res_deg, bounds[2], res_deg)
        self.shape = (lat.size, lon.size)
        self.transform = rasterio.Affine(res_deg, 0, bounds[0], 0, -res_deg, bounds[3])
        centroids = Centroids()
        centroids.set_lat_lon(*[ar.ravel() for ar in np.meshgrid(lon, lat)][::-1])
        centroids.set_dist_coast(signed=True, precomputed=True)
        self.dist_coast = centroids.dist_coast

    def __enter__(self):
        """Write artifical elevation data to a temporary raster file and provide path as string."""
        elevation = -self.dist_coast / 168
        elevation = np.fmax(-1, elevation).reshape(self.shape)
        dst_meta = {
            'driver': 'GTiff',
            'compress': 'deflate',
            'width': elevation.shape[1],
            'height': elevation.shape[0],
            'dtype': 'float64',
            'count': 1,
            'transform': self.transform,
            'crs': 'epsg:4326',
            'nodata': -32767.0,
        }

        # In Windows, unlike Unix, the temporary file cannot be opened before it is closed.
        # Therefore it is closed right after creation and only the path/name is kept.
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        self.topo_path = tmpfile.name
        tmpfile.close()
        with rasterio.open(self.topo_path, 'w', **dst_meta) as dst:
            dst.write_band(1, elevation)
        return self.topo_path

    def __exit__(self, *args, **kwargs):
        """Remove temporary raster file."""
        Path(self.topo_path).unlink()


class TestTCSurgeBathtub(unittest.TestCase):
    """Test TCSurgeBathtub class."""

    def test_fraction_on_land(self):
        """Test _fraction_on_land helper function."""
        res_deg = 10 / (60 * 60)
        bounds = (-149.54, -23.42, -149.40, -23.33)
        lat = np.arange(bounds[1] + 0.5 * res_deg, bounds[3], res_deg)
        lon = np.arange(bounds[0] + 0.5 * res_deg, bounds[2], res_deg)
        shape = (lat.size, lon.size)
        lon, lat = [ar.ravel() for ar in np.meshgrid(lon, lat)]
        centroids = Centroids()
        centroids.set_lat_lon(lat, lon)
        centroids.set_dist_coast(signed=True, precomputed=True)

        dem_bounds = (bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1)
        dem_res = 3 / (60 * 60)
        with tmp_artifical_topo(dem_bounds, dem_res) as topo_path:
            fraction = _fraction_on_land(centroids, topo_path)
        fraction = fraction.reshape(shape)
        dist_coast = centroids.dist_coast.reshape(shape)

        # check valid range and order of magnitude
        self.assertTrue(np.all((fraction >= 0) & (fraction <= 1)))
        np.testing.assert_array_equal(fraction[dist_coast > 1000], 0)
        np.testing.assert_array_equal(fraction[dist_coast < -1000], 1)

        # check individual known pixel values
        self.assertAlmostEqual(fraction[24, 10], 0.0)
        self.assertAlmostEqual(fraction[22, 11], 0.21)
        self.assertAlmostEqual(fraction[22, 12], 0.93)
        self.assertAlmostEqual(fraction[21, 14], 1.0)


    def test_surge_from_track(self):
        """Test TCSurgeBathtub.from_tc_winds function."""
        # similar to IBTrACS 2010029S12177 (OLI, 2010) hitting Tubuai, but much stronger
        track = xr.Dataset({
            'radius_max_wind': ('time', [15., 15, 15, 15, 15, 17, 20, 20]),
            'radius_oci': ('time', [202., 202, 202, 202, 202, 202, 202, 202]),
            'max_sustained_wind': ('time', [155., 147, 140, 135, 130, 122, 115, 116]),
            'central_pressure': ('time', [894., 901, 906, 909, 913, 918, 924, 925]),
            'environmental_pressure': ('time', np.full((8,), 1004.0, dtype=np.float64)),
            'time_step': ('time', np.full((8,), 3.0, dtype=np.float64)),
            'basin': ('time', np.full((8,), "SP")),
        }, coords={
            'time': np.arange('2010-02-05T09:00', '2010-02-06T09:00',
                              np.timedelta64(3, 'h'), dtype='datetime64[h]'),
            'lat': ('time', [-24.33, -25.54, -24.79, -24.05,
                             -23.35, -22.7, -22.07, -21.50]),
            'lon': ('time', [-147.27, -148.0, -148.51, -148.95,
                             -149.41, -149.85, -150.27, -150.56]),
        }, attrs={
            'max_sustained_wind_unit': 'kn',
            'central_pressure_unit': 'mb',
            'name': 'test',
            'sid': '2010029S12177_test',
            'orig_event_flag': True,
            'data_provider': 'unit_test',
            'id_no': 0,
            'category': 4,
        })
        tc_track = TCTracks()
        tc_track.data = [track]
        tc_track.equal_timestep(time_step_h=1)

        res_deg = 10 / (60 * 60)
        bounds = (-149.54, -23.42, -149.40, -23.33)
        lat = np.arange(bounds[1] + 0.5 * res_deg, bounds[3], res_deg)
        lon = np.arange(bounds[0] + 0.5 * res_deg, bounds[2], res_deg)
        shape = (lat.size, lon.size)
        lon, lat = [ar.ravel() for ar in np.meshgrid(lon, lat)]
        centroids = Centroids()
        centroids.set_lat_lon(lat, lon)
        centroids.set_dist_coast(signed=True, precomputed=True)

        wind_haz = TropCyclone()
        wind_haz.set_from_tracks(tc_track, centroids=centroids)

        dem_bounds = (bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1)
        dem_res = 3 / (60 * 60)
        with tmp_artifical_topo(dem_bounds, dem_res) as topo_path:
            for slr in [0, 0.5, 1.5]:
                surge_haz = TCSurgeBathtub.from_tc_winds(wind_haz, topo_path,
                                                         add_sea_level_rise=slr)
                inten = surge_haz.intensity.toarray().reshape(shape)
                fraction = surge_haz.fraction.toarray().reshape(shape)

                # check valid range and order of magnitude
                np.testing.assert_array_equal(inten >= 0, True)
                np.testing.assert_array_equal(inten <= 10, True)
                np.testing.assert_array_equal((fraction >= 0) & (fraction <= 1), True)
                np.testing.assert_array_equal(inten[fraction == 0], 0)

                # check individual known pixel values
                self.assertAlmostEqual(inten[9, 31], max(-0.391 + slr, 0), places=2)
                self.assertAlmostEqual(inten[14, 34] - slr, 3.637, places=2)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTCSurgeBathtub)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
