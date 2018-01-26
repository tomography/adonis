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
#import censor.common.constants as const
import censor.repairs as rp


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


def test_replace_negative():
    fixers = {'REPLACE_NEGATIVE':(0)}
    arr = rp.replace(arr_3D, fixers, data_tag, logger)
    assert not ((arr < 0).any())
    arr = rp.replace(arr_2D, fixers, data_tag, logger)
    assert not ((arr < 0).any())


def test_replace_nan():
    fixers = {'REPLACE_NAN': (0)}
    arr = rp.replace(arr_3D, fixers, data_tag, logger)
    assert not (np.isnan(arr).any())
    arr = rp.replace(arr_2D, fixers, data_tag, logger)
    assert not (np.isnan(arr).any())


def test_to_type():
    fixers = {'TO_TYPE':(np.dtype(np.cfloat))}
    arr = rp.replace(arr_3D, fixers, data_tag, logger)
    assert arr.dtype is np.dtype(np.cfloat)
    arr = rp.replace(arr_2D, fixers, data_tag, logger)
    assert arr.dtype is np.dtype(np.cfloat)


