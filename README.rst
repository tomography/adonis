======
Censor
======

Censor is a batch testing and repairing library for scientific data. 


Features
========
* Checks for correctness of data
* Repair data for errors
* Provide reports on data health


Installation
============
clone censor from GitHub repository:
git clone https://github.com/tomography/censor
then:
cd censor
python setup.py install


Usage
=====
checking data://
//
Create check:parameters dictionary://
checks_dir = {censor.common.constants.IS_NPARRAY:(),//
                 censor.common.constants.HAS_NO_NEGATIVE:(),//
                 censor.common.constants.HAS_NO_NAN:(), //
                 censor.common.constants.IS_INT:(),//
                 censor.common.constants.IS_SIZE:(360, 1024, 1024) }//
Run the verification://
result = censor.checks.check(arr, checks_dir[data_tag, logger, axis])//
//
Fixing data://
//
Create fixer:parameters dictionary://
fixers_dir = {censor.common.constants.REPLACE_NEGATIVE:(0),//
                      censor.common.constants.REPLACE_NAN: (0),//
                      censor.common.constants.TO_TYPE:(np.dtype(np.cfloat))}//
Run the repair://
arr = censor.repairs.replace(arr, fixers_dir, [data_tag, logger])
