#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:21:42 2020

@author: ckropf
"""

import logging
import math
import numpy as np
import decimal


LOGGER = logging.getLogger(__name__)

ABBREV = {1:'',
          1000: 'K',
          1000000: 'M',
          1000000000: 'Bn',
          1000000000000: 'Tn'}


def sig_dig(x, n_sig_dig = 16):
    """
    Rounds x to n_sig_dig number of significant digits.
    Examples: 1.234567 -> 1.2346, 123456.89 -> 123460.0

    Parameters
    ----------
    x : float
        number to be rounded
    n_sig_dig : int, optional
        Number of significant digits. The default is 16.

    Returns
    -------
    float
        Rounded number

    """
    num_of_digits = len(str(x).replace(".", ""))
    if n_sig_dig >= num_of_digits:
        return x
    n = math.floor(math.log10(abs(x)) + 1 - n_sig_dig)
    result = decimal.Decimal(str(np.round(x * 10**(-n)))) \
              * decimal.Decimal(str(10**n))
    return float(result)


def sig_dig_list(iterable, n_sig_dig=16):
    """
    Vectorized form of sig_dig. Rounds a list of float to a number
    of significant digits

    Parameters
    ----------
    iterable : iter(float) (1D or 2D)
        iterable of numbers to be rounded
    n_sig_dig : int, optional
        Number of significant digits. The default is 16.


    Returns
    -------
    list
        list of rounded floats

    """
    return np.vectorize(sig_dig)(iterable, n_sig_dig)


def value_to_monetary_unit(values, n_sig_dig=None, abbreviations=None):
    """
    Converts values to closest common monetary unit, default: (K, M Bn, Tn)

    Parameters
    ----------
    values : int or float, list(int or float) or np.ndarray(int or float)
        Values to be converted
    n_sig_dig : int, optional
        Number of significant digits to return.
        Examples n_sig_di=5: 1.234567 -> 1.2346, 123456.89 -> 123460.0
        Default: all digits are returned.
    abbreviations: dict, optional
        Name of the abbreviations for the money 1000s counts
        Default:
         {0:'',
          1000: 'K',
          1000000: 'M',
          1000000000: 'Bn',
          1000000000000: 'Tn'}

    Returns
    -------
    mon_val : np.ndarray
        Array of values in monetary unit
    name : string
        Monetary unit

    """

    if isinstance(values, (int, float)):
        values = [values]

    if abbreviations is None:
        abbreviations= ABBREV

    exponents = []
    for val in values:
        if val == 0:
            exponents.append(0)
            continue
        exponents.append(math.log10(abs(val)))

    max_exp = max(exponents)
    min_exp = min(exponents)

    avg_exp = math.floor((max_exp + min_exp) / 2)  # rounded down
    mil_exp = 3 * math.floor(avg_exp/3)

    name = ''
    thsder = int(10**mil_exp)

    try:
        name = abbreviations[thsder]
    except KeyError:
        LOGGER.warning("Warning: The numbers are larger than %s", list(abbreviations.keys())[-1])
        thsder, name = list(abbreviations.items())[-1]

    mon_val = np.array(values) / thsder

    if n_sig_dig is not None:
        mon_val = [sig_dig(val, n_sig_dig=n_sig_dig) for val in mon_val]

    return (mon_val, name)

def val_to_cat(values):
    """
    Converts an array of values to numbered categories.

    Parameters
    ----------
    values : list or array  (1D or 2D)
        List of categories (any type that can be used as input for np.unique())
        Note: int and float are considered different numbers (1 != 1.0)
    Returns
    -------
    valcat : np.array
        List of the input values mapped onto categories.
        The categories are deduced from the ordered input values.
        
    Example
    -------
    val_to_cat([1, 2, 1, 2, 2, 10]) = np.array([0, 1, 0, 1, 1, 2])
    val_to_cat([1, 'a', 'a']) = np.array([0, 1, 1])
    val_to_cat([1, 1.0, 'a']) = np.array([0, 1, 2])
    val_to_cat([[1, 1, 2], [1, 'a']]) = np.array([[0, 0, 1],
                                                  [0, 2]])

    """
    
    values = np.array(values)
    all_cat = {
        str(val): cat
        for cat, val in enumerate(np.unique(values.flatten()))
        }
    
    return np.vectorize(_val_to_cat_single)(np.array(values, dtype=str), all_cat)

def _val_to_cat_single(value, all_cat):
    """
    Helper function to vectorize dictionnary look-up

    """
    return all_cat[value]
    
