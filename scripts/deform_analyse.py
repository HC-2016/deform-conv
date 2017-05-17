"""
An example to illustrate the deformable convolution
"""
import numpy as np
import tensorflow as tf

# specify input feature map
i_coor = np.arange(5); j_coor = np.arange(5)
iv, jv = np.meshgrid(i_coor, j_coor, indexing='ij')
iv = iv[None, ...]; jv = jv[None, ...]
orin_x = np.stack([iv, jv], axis=3)
print(orin_x)

# specify offsets
offsets = np.ones(shape=(1, 5, 5, 4))
print(offsets)

def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])

def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of np.repeat for 1D"""
    # https://github.com/tensorflow/tensorflow/issues/8521
    assert len(a.get_shape()) == 1

    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a

def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a

# graph
sess = tf.InteractiveSession()
x = tf.constant(value=orin_x, dtype='float32')    # (1, 5, 5, 2)
offsets = tf.constant(value=offsets, dtype='float32')    # (1, 5, 5, 4)

# offsets: (b, h, w, c) -> (b*c, h, w, 2)
# offsets = self._to_bc_h_w_2(offsets, x_shape)
x_shape = x.get_shape()     # (1, 5, 5, 2)
offsets = tf.transpose(offsets, [0, 3, 1, 2])   # (1, 4, 5, 5)
offsets = tf.reshape(offsets, (-1, int(x_shape[1]), int(x_shape[2]), 2))  # （2，5，5, 2）

# x: (b, h, w, c) -> (b*c, h, w)
# x = self._to_bc_h_w(x, x_shape)
x_shape = x.get_shape()     # (1, 5, 5, 2)
x = tf.transpose(x, [0, 3, 1, 2])   # (1, 2, 5, 5)
x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))   # (2, 5, 5)

# x_offset: (b*c, h, w)
# (b*c, h, w, 2) -> (b*c, h*w, 2)
# x_offset = tf_batch_map_offsets(x, offsets)
input_shape = tf.shape(x)  # (2, 5, 5)
batch_size = input_shape[0]  # (2)
input_size = input_shape[1]  # (5)

offsets = tf.reshape(offsets, (batch_size, -1, 2))  # (2, 25, 2)
grid = tf.meshgrid(
    tf.range(input_size), tf.range(input_size), indexing='ij')   # [(5, 5), (5, 5)]
grid = tf.stack(grid, axis=-1)  # (5, 5, 2)
grid = tf.cast(grid, 'float32')
grid = tf.reshape(grid, (-1, 2))  # (25, 2)
grid = tf_repeat_2d(grid, batch_size)  # (2, 25, 2)

coords = offsets + grid  # (2, 25, 2)

# mapped_vals = tf_batch_map_coordinates(input, coords)
input_shape = tf.shape(x)
batch_size = input_shape[0]
input_size = input_shape[1]
n_coords = tf.shape(coords)[1]

# lrtb: left, right, top, bottom
coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)  # (2, 25, 2)
coords_lt = tf.cast(tf.floor(coords), 'int32')
coords_rb = tf.cast(tf.ceil(coords), 'int32')
coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

idx = tf_repeat(tf.range(batch_size), n_coords)  # (50)

def _get_vals_by_coords(input, coords):
    indices = tf.stack([
        idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
    ], axis=-1)  # (50, 3)
    vals = tf.gather_nd(input, indices)  # （50）
    vals = tf.reshape(vals, (batch_size, n_coords))  # (2, 25)
    return vals

vals_lt = _get_vals_by_coords(x, coords_lt)
vals_rb = _get_vals_by_coords(x, coords_rb)
vals_lb = _get_vals_by_coords(x, coords_lb)
vals_rt = _get_vals_by_coords(x, coords_rt)

coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]  # ???
vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]  # ???
mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]  # ???

# x_offset: (b*c, h, w) -> (b, h, w, c)
# x_offset = self._to_b_h_w_c(x_offset, x_shape)
