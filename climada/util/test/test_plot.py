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

Test plot module.
"""

import unittest
import cartopy
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import climada.util.plot as u_plot

BUFFER = 1.0

class TestFuncs(unittest.TestCase):

    def test_get_transform_4326_pass(self):
        """Check _get_transformation for 4326 epsg."""
        res, unit = u_plot.get_transformation({'init': 'epsg:4326'})
        self.assertIsInstance(res, cartopy.crs.PlateCarree)
        self.assertEqual(unit, '°')

    def test_get_transform_3395_pass(self):
        """Check that assigned attribute is correctly set."""
        res, unit = u_plot.get_transformation({'init': 'epsg:3395'})
        self.assertIsInstance(res, cartopy.crs.Mercator)
        self.assertEqual(unit, 'm')

    def test_get_transform_3035_pass(self):
        """Check that assigned attribute is correctly set."""
        res, unit = u_plot.get_transformation({'init': 'epsg:3035'})
        self.assertIsInstance(res, cartopy._epsg._EPSGProjection)
        self.assertEqual(unit, 'm')

class TestPlots(unittest.TestCase):

    def test_geo_scatter_categorical(self):
        """Plots ones with geo_scatteR_categorical"""
        # test default with one plot
        values = np.array([1, 2.0, 1, 'a'])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        u_plot.geo_scatter_categorical(values, coord, 'value', 'test plot',
                        pop_name=True)
        plt.close()

        #test multiple plots with non default kwargs
        values = np.array([[1, 2.0, 1, 'a'], [0, 0, 0, 0]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        u_plot.geo_scatter_categorical(values, coord, 'value', 'test plot',
                        cat_name={0: 'zero',
                                  1: 'int',
                                  2.0: 'float',
                                  'a': 'string'},
                        pop_name=False, cmap='Set1')
        plt.close()

        #test colormap warning
        values = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                           [12, 13, 14, 15]])
        coord = np.array([[26, 0], [26, 4], [28, 0], [29, 1]])
        u_plot.geo_scatter_categorical(values, coord, 'value', 'test plot',
                        pop_name=False, cmap='Set1')

        plt.close()

        #test colormap warning with 256 colors
        values = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                           [12, 13, 14, 15]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        u_plot.geo_scatter_categorical(values, coord, 'value', 'test plot',
                        pop_name=False, cmap='tab20c')
        plt.close()

    def test_geo_scatter_from_array(self):
        values = np.array([1, 2.0, 1, 1])
        coord = np.array([[-17, 178], [-10, 180], [-27, 175], [-16, 186]])
        var_name = 'test'
        title = 'test'
        projection = ccrs.PlateCarree()
        cmap = 'viridis'
        ax = u_plot.geo_scatter_from_array(values, coord, var_name, title,
                               pop_name=True, buffer=BUFFER, extend='neither',
                               shapes=True, axes=None, proj=projection,
                               figsize=(9, 13), cmap=cmap)
        self.assertEqual(var_name, ax.get_title())
        self.assertAlmostEqual(np.max(values), ax.collections[0].colorbar.vmax)
        self.assertAlmostEqual(np.min(values), ax.collections[0].colorbar.vmin)
        self.assertEqual(cmap, ax.collections[0].cmap.name)
        plt.close()

    def test_geo_bin_from_array(self):
        values = np.array([1, 2.0, 5, 1])
        coord = np.array([[-10, 17], [-30, 20], [5, 75], [-16, 20]])
        var_name = 'test'
        title = 'test'
        projection = ccrs.PlateCarree()
        cmap = 'viridis'
        ax = u_plot.geo_bin_from_array(values, coord, var_name, title,
                                           pop_name=True, buffer=BUFFER, extend='neither',
                                           shapes=True, axes=None, proj=projection,
                                           figsize=(9, 13), cmap=cmap)
        self.assertEqual(var_name, ax.get_title())
        self.assertAlmostEqual(np.max(values), ax.collections[0].colorbar.vmax)
        self.assertAlmostEqual(np.min(values), ax.collections[0].colorbar.vmin)
        self.assertEqual(cmap, ax.collections[0].cmap.name)
        plt.close()

    def test_geo_im_from_array(self):
        values = np.array([1, 2.0, 5, 1])
        coord = np.array([[-17, 178], [-10, 180], [-27, 175], [-16, 186]])
        var_name = 'test'
        title = 'test'
        projection = ccrs.PlateCarree()
        cmap = 'viridis'
        ax = u_plot.geo_im_from_array(values, coord, var_name, title,
                      proj=projection, smooth=True, axes=None, figsize=(9, 13), cmap=cmap)
        self.assertEqual(var_name, ax.get_title())
        self.assertAlmostEqual(np.max(values), ax.collections[0].colorbar.vmax)
        self.assertAlmostEqual(np.min(values), ax.collections[0].colorbar.vmin)
        self.assertEqual(cmap, ax.collections[0].cmap.name)
        plt.close()


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlots))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
