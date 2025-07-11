def reset_random():
    print('[INFO] Resetting Random Seeds')
    seed = 1
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import warnings
    warnings.filterwarnings('ignore', category=Warning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf
    tf.compat.v1.random.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)
