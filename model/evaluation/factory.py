from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.evaluation import coco_evaluator


def evaluator_generator(params):
  """Generator function for various evaluators."""
  if params.type == 'box':
    evaluator = coco_evaluator.COCOEvaluator(
        annotation_file=params.val_json_file, include_mask=False)
  elif params.type == 'box_and_mask':
    evaluator = coco_evaluator.COCOEvaluator(
        annotation_file=params.val_json_file, include_mask=True)
  else:
    raise ValueError('Evaluator %s is not supported.' % params.type)

  return coco_evaluator.MetricWrapper(evaluator)