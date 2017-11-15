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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from multiprocessing import Queue, Process
import censor.common.constants as const
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
           'is_string',
           'is_size',
           'check_slices',
           'check']

check_mapper = {
               1 : '"is array"',
               2 : '"no negative"',
               3 : '"no nan"',
               4 : '"int type"',
               5 : '"float type"',
               6 : '"complex type"',
               7 : '"string type"',
               8 : '"correct array size"'
               }


def is_nparray(arr, args):
    """
    This method returns True if the given instance's type is ndarray, False otherwise.

    Parameters
    ----------
    arr : instance
        an evaluated instance
    args : tuple
        not used
    Returns
    -------
    boolean
    """
    return isinstance(arr, np.ndarray)

def has_no_negative(arr, args):
    """
    This method returns True if the given array has no negative elements, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        not used
    Returns
    -------
    boolean
    """
    return not ((arr < 0).any())

def has_no_nan(arr, args):
    """
    This method returns True if the given array has no nan elements, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        not used
    Returns
    -------
    boolean
    """
    return not (np.isnan(arr).any())

def is_int(arr, args):
    """
    This method returns True if the given array type is int, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        not used
    Returns
    -------
    boolean
    """
    return arr.dtype is np.dtype(np.int)

def is_float(arr, args):
    """
    This method returns True if the given array type is float, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        not used
    Returns
    -------
    boolean
    """
    return arr.dtype is np.dtype(np.float)

def is_complex(arr, args):
    """
    This method returns True if the given array type is complex, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        not used
    Returns
    -------
    boolean
    """
    return arr.dtype is np.dtype(np.cfloat)

def is_string(arr, args):
    """
    This method returns True if the given array type is string, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        not used
    Returns
    -------
    boolean
    """
    return arr.dtype is np.dtype(np.string)

def is_size(arr, args):
    """
    This method returns True if the given array size matches the arguments, False otherwise.

    Parameters
    ----------
    arr : ndarray
        an evaluated array
    args : tuple
        contains size the array is validated to
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


function_mapper = { const.IS_NPARRAY : is_nparray,
                    const.HAS_NO_NEGATIVE : has_no_negative,
                    const.HAS_NO_NAN : has_no_nan,
                    const.IS_INT : is_int,
                    const.IS_FLOAT : is_float,
                    const.IS_COMPLEX: is_complex,
                    const.IS_STRING : is_string,
                    const.IS_SIZE : is_size
                   }

def check_slices(logger, checks, arr, data_tag):
    """
    This method provides data validation using functions validating frame by frame.

    It starts a handler process that will receive data frame by frame via queue.

    Parameters
    logger : logger instance
        logger used to log events
    checks : dict
        contains functions ids as keys, and corresponding tuple of parameters as value
    arr : ndarray
        an evaluated array
    data_tag : str
        string identifying the data
    Returns
    -------
    none
    """
    dataq = Queue()
    p = Process(target=handler.handle_data, args=(logger,dataq, checks, data_tag))
    p.start()

    arr = np.atleast_3d(arr)

    for num_slice in range(arr.shape[2]):
        slice = arr[:,:,num_slice]
        dataq.put(ct.Data(ct.Data.DATA_STATUS_DATA, slice))
    dataq.put(ct.Data(ct.Data.DATA_STATUS_END))


def check(logger, checks, arr, data_tag):
    """
    This method provides data validation.

    It runs validation functions defined in checks dictionary.
    The function id value has additional meaning. Ids below 100 are reserved
    for functions that evaluate array without slicing it, where the functions
    with ids over 100 evalute array frame by frame.

    Parameters
    logger : logger instance
        logger used to log events
    checks : dict
        contains functions ids as keys, and corresponding tuple of parameters as value
    arr : ndarray
        an evaluated array
    data_tag : str
        string identifying the data
    Returns
    -------
    none
    """
    for check in sorted(checks):
        if check < 100:
            res = function_mapper[check](arr, checks[check])
            logger.info(data_tag + ' evaluated ' + check_mapper[check] + ' with result ' + str(res))
            del checks[check]
    if len(checks) > 0:
        check_slices(logger, checks, arr, data_tag)

        

        


