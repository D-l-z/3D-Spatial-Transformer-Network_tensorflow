#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:04:18 2018

@author: dlz
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def transform(U, theta, out_size, name='3D_SpatialTransformer', **kwargs):
  """Model transforming the 3D voxels into 2D projections.

  Args:
    voxels: A tensor of size [batch, depth, height, width, channel]
      representing the input of projection layer (tf.float32).
    transform_matrix: A tensor of size [batch, 16] representing
      the flattened 4-by-4 matrix for transformation (tf.float32).
    params: Model parameters (dict).
    is_training: Set to True if while training (boolean).

  Returns:
    A transformed tensor (tf.float32)

  """
  U = tf.transpose(U, [0, 2, 1, 3, 4])

  num = tf.shape(theta)[0]
  t00 = 1-2*theta[:, 2]**2 - 2*theta[:, 3]**2
  t01 = 2*theta[:, 1]*theta[:, 2] - 2*theta[:, 0]*theta[:, 3]
  t02 = 2*theta[:, 1]*theta[:, 3] + 2*theta[:, 0]*theta[:, 2]
  t03 = tf.zeros([num,])
  t0 = tf.stack([t00, t01, t02, t03], axis=1)
  
  t10 = 2*theta[:, 1]*theta[:, 2] + 2*theta[:, 0]*theta[:, 3]
  t11 = 1-2*theta[:, 1]**2 - 2*theta[:, 3]**2
  t12 = 2*theta[:, 2]*theta[:, 3] - 2*theta[:, 0]*theta[:, 1]
  t13 = tf.zeros([num,])
  t1 = tf.stack([t10, t11, t12, t13], axis=1)
  
  t20 = 2*theta[:, 1]*theta[:, 3] - 2*theta[:, 0]*theta[:, 2]
  t21 = 2*theta[:, 2]*theta[:, 3] + 2*theta[:, 0]*theta[:, 1]
  t22 = 1-2*theta[:, 1]**2 - 2*theta[:, 2]**2
  t23 = tf.zeros([num,])
  t2 = tf.stack([t20, t21, t22, t23], axis=1)
  
  t = tf.stack([t0, t1, t2], axis=1)
  
  m = tf.zeros([num,1,3])
  n = tf.ones([num,1,1])
  z = tf.concat([m,n],2)
  
  t = tf.concat([t,z],1)
  
  V = transformer(U, t, [out_size] * 3)
  return V

def transformer(voxels,
                theta,
                out_size,
                name='PerspectiveTransformer'):
  """Transformer Layer.

  Args:
    voxels: A tensor of size [num_batch, depth, height, width, num_channels].
      It is the output of a deconv/upsampling conv network (tf.float32).
    theta: A tensor of size [num_batch, 16].
      It is the inverse camera transformation matrix (tf.float32).
    out_size: A tuple representing the size of output of
      transformer layer (float).

  Returns:
    A transformed tensor (tf.float32).

  """
  def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
      rep = tf.transpose(
          tf.expand_dims(tf.ones(shape=tf.stack([
              n_repeats,
          ])), 1), [1, 0])
      rep = tf.to_int32(rep)
      x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
      return tf.reshape(x, [-1])

  def _interpolate(im, x, y, z, out_size):
    """Bilinear interploation layer.

    Args:
      im: A 5D tensor of size [num_batch, depth, height, width, num_channels].
        It is the input volume for the transformation layer (tf.float32).
      x: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for x (tf.float32).
      y: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for y (tf.float32).
      z: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for z (tf.float32).
      out_size: A tuple representing the output size of transformation layer
        (float).

    Returns:
      A transformed tensor (tf.float32).

    """
    with tf.variable_scope('_interpolate'):
      num_batch = tf.shape(im)[0]
      depth = tf.shape(im)[1]
      height = tf.shape(im)[2]
      width = tf.shape(im)[3]
      channels = tf.shape(im)[4]

      x = tf.to_float(x)
      y = tf.to_float(y)
      z = tf.to_float(z)
      depth_f = tf.to_float(depth)
      height_f = tf.to_float(height)
      width_f = tf.to_float(width)
      # Number of disparity interpolated.
      out_depth = out_size[0]
      out_height = out_size[1]
      out_width = out_size[2]
      zero = tf.zeros([], dtype='int32')
      # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
      max_z = tf.to_int32(tf.shape(im)[1] - 1)
      max_y = tf.to_int32(tf.shape(im)[2] - 1)
      max_x = tf.to_int32(tf.shape(im)[3] - 1)

      # Converts scale indices from [-1, 1] to [0, width/height/depth].
      x = (x + 1.0) * (width_f) / 2.0
      y = (y + 1.0) * (height_f) / 2.0
      z = (z + 1.0) * (depth_f) / 2.0

      x0 = tf.to_int32(tf.floor(x))
      x1 = x0 + 1
      y0 = tf.to_int32(tf.floor(y))
      y1 = y0 + 1
      z0 = tf.to_int32(tf.floor(z))
      z1 = z0 + 1

      x0_clip = tf.clip_by_value(x0, zero, max_x)
      x1_clip = tf.clip_by_value(x1, zero, max_x)
      y0_clip = tf.clip_by_value(y0, zero, max_y)
      y1_clip = tf.clip_by_value(y1, zero, max_y)
      z0_clip = tf.clip_by_value(z0, zero, max_z)
      z1_clip = tf.clip_by_value(z1, zero, max_z)
      dim3 = width
      dim2 = width * height
      dim1 = width * height * depth
      base = _repeat(
          tf.range(num_batch) * dim1, out_depth * out_height * out_width)
      base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
      base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
      base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
      base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

      idx_z0_y0_x0 = base_z0_y0 + x0_clip
      idx_z0_y0_x1 = base_z0_y0 + x1_clip
      idx_z0_y1_x0 = base_z0_y1 + x0_clip
      idx_z0_y1_x1 = base_z0_y1 + x1_clip
      idx_z1_y0_x0 = base_z1_y0 + x0_clip
      idx_z1_y0_x1 = base_z1_y0 + x1_clip
      idx_z1_y1_x0 = base_z1_y1 + x0_clip
      idx_z1_y1_x1 = base_z1_y1 + x1_clip

      # Use indices to lookup pixels in the flat image and restore
      # channels dim
      im_flat = tf.reshape(im, tf.stack([-1, channels]))
      im_flat = tf.to_float(im_flat)
      i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
      i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
      i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
      i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
      i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
      i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
      i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
      i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)

      # Finally calculate interpolated values.
      x0_f = tf.to_float(x0)
      x1_f = tf.to_float(x1)
      y0_f = tf.to_float(y0)
      y1_f = tf.to_float(y1)
      z0_f = tf.to_float(z0)
      z1_f = tf.to_float(z1)
      # Check the out-of-boundary case.
      x0_valid = tf.to_float(
          tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
      x1_valid = tf.to_float(
          tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
      y0_valid = tf.to_float(
          tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
      y1_valid = tf.to_float(
          tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
      z0_valid = tf.to_float(
          tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
      z1_valid = tf.to_float(
          tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))

      w_z0_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                   (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                  1)
      w_z0_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                   (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                  1)
      w_z0_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                   (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                  1)
      w_z0_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                   (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                  1)
      w_z1_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                   (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                  1)
      w_z1_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                   (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                  1)
      w_z1_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                   (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                  1)
      w_z1_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                   (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                  1)

      output = tf.add_n([
          w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1,
          w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1,
          w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1,
          w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1
      ])
      return output

  def _meshgrid(depth, height, width):
    with tf.variable_scope('_meshgrid'):
      x_t = tf.reshape(
          tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]),
          [depth, height, width])
      
      y_t = tf.reshape(
          tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]),
          [depth, width, height])
      y_t = tf.transpose(y_t, [0, 2, 1])
      
      z_t = tf.reshape(
          tf.tile(tf.linspace(-1.0, 1.0, depth), [height * width]),
          [height, width, depth])
      z_t = tf.transpose(z_t, [2, 0, 1])
      
      x_t_flat = tf.reshape(x_t, (1, -1))
      y_t_flat = tf.reshape(y_t, (1, -1))
      d_t_flat = tf.reshape(z_t, (1, -1))
      
      ones = tf.ones_like(x_t_flat)
      grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
      return grid

  def _transform(theta, input_dim, out_size):
    with tf.variable_scope('_transform'):
      num_batch = tf.shape(input_dim)[0]
      num_channels = tf.shape(input_dim)[4]
      theta = tf.reshape(theta, (-1, 4, 4))
      theta = tf.cast(theta, 'float32')

      out_depth = out_size[0]
      out_height = out_size[1]
      out_width = out_size[2]
      grid = _meshgrid(out_depth, out_height, out_width)
      grid = tf.expand_dims(grid, 0)
      grid = tf.reshape(grid, [-1])
      grid = tf.tile(grid, tf.stack([num_batch]))
      grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))

      t_g = tf.matmul(theta, grid)
      z_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
      y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
      x_s = tf.slice(t_g, [0, 2, 0], [-1, 1, -1])

      z_s_flat = tf.reshape(z_s, [-1])
      y_s_flat = tf.reshape(y_s, [-1])
      x_s_flat = tf.reshape(x_s, [-1])

      input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, z_s_flat,
                                       out_size)

      output = tf.reshape(
          input_transformed,
          tf.stack([num_batch, out_depth, out_height, out_width, num_channels]))

      return output

  with tf.variable_scope(name):
    output = _transform(theta, voxels, out_size)
    return output

