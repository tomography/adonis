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
This file is a suite of verification functions for scientific data.

It is assumed that the evaluated data is 2D shape.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import censor.common.containers as ct

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['sat_in_range',
           'mean_in_range',
           'process_frame']

def sat_in_range(arr, args):
    """
    This method validates saturation. The arguments are positional.

    It calculates number of saturated points. Point is saturated if it's intensity is greater
    than given value (args[0]). If the number of saturated points exceeds limit, given as args[1],
    the frame is saturated.

    Parameters
    ----------
    arr : 2D array
        a frame
    args : tuple
        a tuple containing positional arguments
    Returns
    -------
        result : object
    """
    # find number of saturated pixels, args[0] is the pixel saturation limit
    sat_pixels = (arr > args[0]).sum()
    # args[1] is a limit of saturated pixels
    res = sat_pixels < args[1]
    result = ct.Result(res, 'saturation_in_range')
    return result


def mean_in_range(arr, args):
    """
    This method validates mean value. The arguments are positional.

    It calculates mean value of the frame and checks if the value is within limits, given as args.

    Parameters
    ----------
    arr : 2D array
        a frame
    args : tuple
        a tuple containing positional arguments
    Returns
    -------
        result : object
    """
    mn = np.mean(arr)
    res = mn > args[0] and mn < args[1]
    return ct.Result(res, 'mean_in_range')


# maps the quality check ID to the function object
function_mapper = {
                     'MEAN_IN_RANGE' : mean_in_range,
                     'SAT_IN_RANGE' : sat_in_range
                   }


def process_frame(data, index, resultsq, functions):
    """
    This method dispatches validation/repair functions that are included in the functions dictionary.

    It calls a function defined in the functions dictionary, using the dictionary value as an argument.
    The Results objects returned by each function are encapsulated in Results object and enqueued in a
    results queue.

    Parameters
    ----------
    data : 2D array
        a frame
    index : int
        a frame index
    resultsq : queue
        a queue that will deliver results to parent process
    functions : dict
        a dictionary containing functins ids, and tuple values, the tuple containing positional arguments.
    Returns
    -------
        none
    """
    results_list = []
    failed = False
    for function_id in functions:
        function = function_mapper[function_id]
        result = function(data.slice, functions[function_id])
        results_list.append(result)
        if not result.res:
            failed = True

    results = ct.Results(index, failed, results_list)
    resultsq.put(results)

def process_frame_seq(data, index, functions):
    """
    This method dispatches validation/repair functions that are included in the functions dictionary.

    It calls a function defined in the functions dictionary, using the dictionary value as an argument.
    The Results objects returned by each function are encapsulated in Results object and enqueued in a
    results queue.

    Parameters
    ----------
    data : 2D array
        a frame
    index : int
        a frame index
    resultsq : queue
        a queue that will deliver results to parent process
    functions : dict
        a dictionary containing functins ids, and tuple values, the tuple containing positional arguments.
    Returns
    -------
        none
    """
    results_list = []
    failed = False
    for function_id in functions:
        function = function_mapper[function_id]
        result = function(data.slice, functions[function_id])
        results_list.append(result)
        if not result.res:
            failed = True

    results = ct.Results(index, failed, results_list)
    return results
