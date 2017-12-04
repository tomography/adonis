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
This file handles verification and repair frame by frame.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

from multiprocessing import Queue, Process
import sys
import censor.frame as framer
import censor.common.containers as ct
if sys.version[0] == '2':
    import Queue as queue
else:
    import queue as queue

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['handle_data']


def handle_data(dataq, checks, returnq, data_tag, logger):
    """
    This method validates and repairs data applying checks and repairs functions.

    It receives data frame by frame via multiprocessing queue. It uses one process to validate/repair
    each frame. The results are sent to aggregate for processing.

    Parameters
    ----------
    dataq : Queue
        multiprocessing queue delivering data slice by slice
    checks : dictionary
        a dictionary containing methods ids that will be applied to validate/repair each frame
    returnq : Queue
        multiprocessing queue used to transfer final result to the parent process
    data_tag : string
        a string associated with the data, used when logging events
    logger : logger instance
        logger used to log events
    Returns
    -------
        none
    """
    aggregate = ct.Aggregate(logger, data_tag)
    resultsq = Queue()
    interrupted = False
    index = 0
    num_processes = 0
    verified = True
    while not interrupted:
        try:
            data = dataq.get(timeout=0.001)
            if data.status == ct.Data.DATA_STATUS_END:
                interrupted = True
                while num_processes > 0:
                    results = resultsq.get()
                    if results.failed:
                        verified = False
                    aggregate.handle_results(logger, results)
                    num_processes -= 1
            elif data.status == ct.Data.DATA_STATUS_DATA:
                p = Process(target=framer.process_frame,
                            args=(data, index, resultsq, checks))
                p.start()
                num_processes += 1
                index += 1

        except queue.Empty:
            pass

        while not resultsq.empty():
            results = resultsq.get_nowait()
            if results.failed:
                verified = False
            aggregate.handle_results(logger, results)
            num_processes -= 1

    returnq.put(verified)


