#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2017, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2017. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
This module corrects array's content.

It contains set of functions, each function correcting certain characteristic.
Caller will use "replace" interface to start the correction, and will provide
list of functions to apply.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import censor.common.constants as const
import logging

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['replace_negative',
           'replace_nan',
           'to_type',
           'replace']

fix_mapper = {
               1 : 'negative',
               2 : 'nan',
               3 : 'type' 
             } 

def replace_negative(arr, value):
    """
    This function replaces negative values in array arr with the given value.

    Parameters
    ----------
    arr : ndarray
        repaired array
    value : the type of array
        replacement value
    Returns
    -------
    arr : ndarray
        corrected array
    """
    arr[arr < 0] = value
    return arr


def replace_nan(arr, value):
    """
    This function replaces nan values (not a number) in array arr with the given value.

    Parameters
    ----------
    arr : ndarray
        repaired array
    value : the type of array
        replacement value
    Returns
    -------
    arr : ndarray
        corrected array
    """
    arr[np.isnan(arr)] = value
    return arr


def to_type(arr, type):
    """
    This function changes the type of elements in array to the given type.

    Parameters
    ----------
    arr : ndarray
        repaired array
    type : numpy.dtype
        new type
    Returns
    -------
    arr : ndarray
        corrected array
    """
    return arr.astype(type)


function_mapper = { const.REPLACE_NEGATIVE : replace_negative,
                    const.REPLACE_NAN : replace_nan,
                    const.TO_TYPE : to_type
                   }


def replace(arr, fixers, data_tag='mydata', logger=None):
    """
    This function provides data repair.

    It runs all the repair functions included in "fixers" dictionary. Each function
    is identified by integer ID, arbitrary defined.
    The "fixers" dictionary's keys are the functions IDs, and the values are
    corresponding arguments grouped in tuple. Fixers dictionary example:
    fixers = {const.REPLACE_NEGATIVE:(0), const.REPLACE_NAN:(0), const.TO_TYPE:(np.dtype(np.float))}

    Parameters
    ----------
    arr : ndarray
        repaired array
    fixers : dict
        contains functions ids as keys, and corresponding tuple of parameters as value
    data_tag : str
        string identifying the data
    logger : logger instance
        logger used to log events
    Returns
    -------
    arr : ndarray
        corrected array
    """
    # if logger not provided, create default
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('default.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    for fix in sorted(fixers):
        if fix < 100:
            arr = function_mapper[fix](arr, fixers[fix])
            logger.info(data_tag + ' repaired ' + fix_mapper[fix] )
    return arr



