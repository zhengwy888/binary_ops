"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


#if tf.test.is_built_with_cuda():
_cuda_op_module = tf.load_op_library(os.path.join(
  tf.resource_loader.get_data_files_path(), 'bnmatmul_op.so'))

bn_matmul = _cuda_op_module.bn_matmul
ops.RegisterShape("BnMatmul")(common_shapes.matmul_shape)
@ops.RegisterGradient("BnMatmul")
def _BnMatMulGrad(op, grad):
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  if not t_a and not t_b:
    return (math_ops.matmul(grad, op.inputs[1], transpose_b=True),
            math_ops.matmul(op.inputs[0], grad, transpose_a=True))
  elif not t_a and t_b:
    return (math_ops.matmul(grad, op.inputs[1]),
            math_ops.matmul(grad, op.inputs[0], transpose_a=True))
  elif t_a and not t_b:
    return (math_ops.matmul(op.inputs[1], grad, transpose_b=True),
            math_ops.matmul(op.inputs[0], grad))
  elif t_a and t_b:
    return (math_ops.matmul(op.inputs[1], grad, transpose_a=True,
                            transpose_b=True),
            math_ops.matmul(grad, op.inputs[0], transpose_a=True,
                            transpose_b=True))

