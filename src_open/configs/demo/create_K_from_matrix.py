# Helper: not used by Hydra; just a reference.
# Given a 3x3 intrinsic, write fx fy cx cy or full K.txt.

import numpy as np

K = np.array([
    [1440.6351,    0.0,     642.3443],
    [   0.0,    1442.8019,  282.1061],
    [   0.0,       0.0,       1.0    ]
], dtype=np.float32)

np.savetxt('K_full.txt', K.reshape(1,9))
np.savetxt('K_xycc.txt', np.array([[K[0,0], K[1,1], K[0,2], K[1,2]]], dtype=np.float32))
