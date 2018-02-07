import censor.checks as ck
import numpy as np

arr_3D = np.array([[[1, 2, 3], [np.log(-1.), -5, -7]],[[1, 2, 3], [4, 5, 6]],
[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

checks = {'MEAN_IN_RANGE': (0, 7)}
arr = arr_3D.copy()
arr[np.isnan(arr)] = 0
arr[arr < 0] = 0
ck.check(arr, checks)

