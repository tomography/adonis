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


import logging
import numpy as np
import os
#import censor.common.constantsx as const
import censor.checks as ck


arr_2D = np.array([[1, 2, 3], [np.log(-1.), -5, -7]])
arr_3D = np.array([[[1, 2, 3], [np.log(-1.), -5, -7]],[[1, 2, 3], [4, 5, 6]],
[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logfile = 'test.log'
handler = logging.FileHandler(logfile)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

data_tag = 'test'


def is_text_in_file(file, text):
    return text in open(file).read()
    file.close()


def test_nparray():
    checks = {'IS_NPARRAY': ()}
    verified = ck.check(arr_2D, checks, data_tag, logger)
    assert verified


def test_no_nparra():
    checks = {'IS_NPARRAY': ()}
    verified = ck.check('a', checks, data_tag, logger)
    assert not verified


def test_no_negative():
    checks = {'HAS_NO_NEGATIVE': ()}
    arr = arr_2D.copy()
    arr[arr < 0] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified


def test_has_negative():
    checks = {'HAS_NO_NEGATIVE': ()}
    verified = ck.check(arr_3D, checks, data_tag, logger)
    assert not verified


def test_no_nans():
    checks = {'HAS_NO_NAN': ()}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified


def test_has_nans():
    checks = {'HAS_NO_NAN': ()}
    verified = ck.check(arr_3D, checks, data_tag, logger)
    assert not verified


def test_is_int():
    checks = {'IS_INT': ()}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr = arr.astype(int)
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified


def test_no_int():
    checks = {'IS_INT': ()}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr = arr.astype(np.dtype(np.float))
    verified = ck.check(arr, checks, data_tag, logger)
    assert not verified


def test_is_float():
    checks = {'IS_FLOAT': ()}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr = arr.astype(np.dtype(np.float))
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified


def test_no_float():
    checks = {'IS_FLOAT': ()}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr = arr.astype(int)
    verified = ck.check(arr, checks, data_tag, logger)
    assert not verified


def test_is_complex():
    checks = {'IS_COMPLEX': ()}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr = arr.astype(np.dtype(np.complex))
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified


def test_no_complex():
    checks = {'IS_COMPLEX': ()}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert not verified


def test_is_size():
    checks = {'IS_SIZE': (4,2,3)}
    verified = ck.check(arr_3D, checks, data_tag, logger)
    assert verified


def test_no_size_dims():
    checks = {'IS_SIZE': arr_3D.shape}
    verified = ck.check(arr_2D, checks, data_tag, logger)
    assert not verified


def test_no_size_shape():
    checks = {'IS_SIZE': (9,10)}
    verified = ck.check(arr_2D, checks, data_tag, logger)
    assert not verified


def test_is_sat_in_range():
    checks = {'SAT_IN_RANGE':(1, 17)}
    arr = arr_2D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified


def test_no_sat_no_in_range():
    checks = {'SAT_IN_RANGE':(1, 2)}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert not verified


def test_is_mean_in_range():
    checks = {'MEAN_IN_RANGE':(0, 7)}
    arr = arr_2D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified


def test_no_mean_no_in_range():
    checks = {'MEAN_IN_RANGE':(100, 107)}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert not verified


def test_logger():
    checks = {'IS_NPARRAY': ()}
    verified = ck.check(arr_2D, checks, data_tag, logger)
    assert verified
    assert is_text_in_file(logfile, data_tag+' evaluated "is_nparray" with result True')


def test_no_logger():
    checks = {'IS_NPARRAY': ()}
    verified = ck.check(arr_2D, checks, data_tag)
    assert verified
    assert is_text_in_file('default.log', data_tag+' evaluated "is_nparray" with result True')


def test_no_tag():
    open('default.log', 'w').close()
    checks = {'IS_NPARRAY': ()}
    verified = ck.check(arr_2D, checks)
    assert verified
    assert is_text_in_file('default.log', 'mydata'+' evaluated "is_nparray" with result True')
    os.remove('default.log')


def test_2D():
    open(logfile, 'w').close()
    checks = {'MEAN_IN_RANGE': (0, 7)}
    arr = arr_2D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified
    assert is_text_in_file(logfile, data_tag+' evaluated frame #0 mean_in_range with result True')
    assert not is_text_in_file(logfile, 'frame #1')


def test_3D_no_axis():
    open(logfile, 'w').close()
    checks = {'MEAN_IN_RANGE': (0, 7)}
    arr = arr_2D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    verified = ck.check(arr, checks, data_tag, logger)
    assert verified
    assert is_text_in_file(logfile, data_tag + ' evaluated frame #0 mean_in_range with result True')
    assert not is_text_in_file(logfile, 'frame #1')


def test_3D_no_axis():
    open(logfile, 'w').close()
    checks = {'MEAN_IN_RANGE': (0, 7)}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    ck.check(arr, checks, data_tag, logger)
    assert is_text_in_file(logfile, 'frame #0')
    assert is_text_in_file(logfile, 'frame #1')
    assert is_text_in_file(logfile, 'frame #2')
    assert is_text_in_file(logfile, 'frame #3')
    assert not is_text_in_file(logfile, 'frame #4')


def test_3D_axis0():
    open(logfile, 'w').close()
    checks = {'MEAN_IN_RANGE': (0, 7)}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    ck.check(arr, checks, data_tag, logger, 0)
    assert is_text_in_file(logfile, 'frame #0')
    assert is_text_in_file(logfile, 'frame #1')
    assert is_text_in_file(logfile, 'frame #2')
    assert is_text_in_file(logfile, 'frame #3')
    assert not is_text_in_file(logfile, 'frame #4')


def test_3D_axis1():
    open(logfile, 'w').close()
    checks = {'MEAN_IN_RANGE': (0, 7)}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    ck.check(arr, checks, data_tag, logger, 1)
    assert is_text_in_file(logfile, 'frame #0')
    assert is_text_in_file(logfile, 'frame #1')
    assert not is_text_in_file(logfile, 'frame #2')


def test_3D_axis2():
    open(logfile, 'w').close()
    checks = {'MEAN_IN_RANGE': (0, 7)}
    arr = arr_3D.copy()
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0
    ck.check(arr, checks, data_tag, logger, 2)
    assert is_text_in_file(logfile, 'frame #0')
    assert is_text_in_file(logfile, 'frame #1')
    assert is_text_in_file(logfile, 'frame #2')
    assert not is_text_in_file(logfile, 'frame #3')
    os.remove(logfile)


