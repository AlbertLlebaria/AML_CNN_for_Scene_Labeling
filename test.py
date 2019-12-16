import tensorflow as tf


print(tf.random.truncated_normal(
    [8, 8, 10, 25],
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
).shape)
