from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Identity(object):
  """Identity function that forwards the input features."""

  def __call__(self, features, is_training=False):
    """Only forwards the input features."""
    return features