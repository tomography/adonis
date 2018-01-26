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
This module verifies array's content.

It contains set of functions, each function verifying certain characteristic.
Caller will use "check" interface to start the verification, and will provide
list of verification functions to apply.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from multiprocessing import Queue, Process
import logging
import censor.handler as handler
import censor.common.containers as ct

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['is_nparray',
           'has_no_negative',
           'has_no_nan',
           'is_int',
           'is_float',
           'is_complex',
           'is_size',
           'check_slices',
           'check']


def is_nparray(arr, *args):
    """
    This function returns True if the given instance's type is ndarray, False otherwise.

    Parameters
    ----------
    arr : instance
        an evaluated instance
    Returns
    -------
        boolean
    """
    return isinstance(arr, np.ndarray)


def has_no_negative(arr, *args):
    """
    This function returns True if the given array has no negative elements, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    Returns
    -------
        boolean
    """
    return not ((arr < 0).any())


def has_no_nan(arr, *args):
    """
    This function returns True if the given array has no nan elements, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    Returns
    -------
        boolean
    """
    return not (np.isnan(arr).any())


def is_int(arr, *args):
    """
    This function returns True if the given array type is int, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    Returns
    -------
    boolean
    """
    return arr.dtype is np.dtype(np.int)

def is_float(arr, *args):
    """
    This function returns True if the given array type is float, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    Returns
    -------
        boolean
    """
    return arr.dtype is np.dtype(np.float)


def is_complex(arr, *args):
    """
    This function returns True if the given array type is complex, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    Returns
    -------
        boolean
    """
    return arr.dtype is np.dtype(np.cfloat)


def is_size(arr, *args):
    """
    This function returns True if the given array shape matches the arguments, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        contains shape the array is validated to
    Returns
    -------
        boolean
    """
    if len(arr.shape) != len(args):
        return False
    for i in range (len(arr.shape)):
        if arr.shape[i] != args[i]:
            return False
    return True


function_mapper = { 'IS_NPARRAY' : is_nparray,
                    'HAS_NO_NEGATIVE' : has_no_negative,
                    'HAS_NO_NAN' : has_no_nan,
                    'IS_INT' : is_int,
                    'IS_FLOAT' : is_float,
                    'IS_COMPLEX': is_complex,
                    'IS_SIZE' : is_size
                   }


def check_slices(arr, checks, data_tag, logger, axis):
    """
    This function provides data validation using functions validating frame by frame.

    It starts a handler process that will receive data frame by frame via queue.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    checks : dict
        contains functions ids as keys, and corresponding tuple of parameters as value
    data_tag : str
        string identifying the data
    logger : logger instance
        logger used to log events
    Returns
    -------
        True if all functions are verified, False otherwise
    """
    dataq = Queue()
    returnq = Queue()
    p = Process(target=handler.handle_data, args=(dataq, checks, returnq, data_tag, logger))
    p.start()

    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis)

    arr = np.moveaxis(arr,axis, 0)

    for num_slice in range(arr.shape[0]):
        slice = arr[num_slice,:,:]
        dataq.put(ct.Data(ct.Data.DATA_STATUS_DATA, slice))
    dataq.put(ct.Data(ct.Data.DATA_STATUS_END))

    result = returnq.get()
    return result


def check(arr, checks, data_tag='mydata', logger=None, axis=0):
    """
    This function provides data validation.

    It runs validation functions defined in "checks" dictionary. Each function
    is identified by integer ID, arbitrary defined.
    The function ID value has additional meaning. IDs below 100 are reserved
    for functions that evaluate array without slicing it, where the functions
    with IDs over 100 evalute array frame by frame.
    The "checks" dictionary's keys are the functions IDs, and the values are
    corresponding arguments grouped in tuple. Checks dictionary example:
    checks = {const.IS_NPARRAY:(), const.IS_SIZE:(2,3)}.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    checks : dict
        contains functions ids as keys, and corresponding tuple of parameters as value
    data_tag : str
        string identifying the data
    logger : logger instance
        logger used to log events
    axis : int
        an axis by which the frames are ordered, only used when "frame" functions are requested
    Returns
    -------
        True if all functions are verified, False otherwise

    Example:
    checks = {'IS_NPARRAY':(),
              'HAS_NO_NEGATIVE':(),
              'HAS_NO_NAN':(),
              'IS_INT':(),
              'IS_SIZE':(2,3),
              'MEAN_IN_RANGE':(-1,5),
              'SAT_IN_RANGE':(1, 7)}
    censor.checks.check(arr, checks)

    """
    # if logger not provided, create default
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('default.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    verified = True
    for check in sorted(checks):
        if check in function_mapper:
            args = checks[check]
            res = function_mapper[check](arr, *args)
            logger.info(data_tag + ' evaluated "' + check.lower() + '" with result ' + str(res))
            if not res:
                verified = False
            del checks[check]
    if len(checks) > 0:
        res = check_slices(arr, checks, data_tag, logger, axis)
        if not res:
            verified = False

    return verified

def is_numpy():
    return 1
