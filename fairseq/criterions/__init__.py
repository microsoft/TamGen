# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry
from fairseq.criterions.fairseq_criterion import FairseqCriterion


build_criterion, register_criterion, CRITERION_REGISTRY = registry.setup_registry(
    '--criterion',
    base_class=FairseqCriterion,
    default='cross_entropy',
)


# automatically import any Python files in the criterions/ directory
criterions_dir = os.path.dirname(__file__)
for file in os.listdir(criterions_dir):
    path = os.path.join(criterions_dir, file)
    if not file.startswith('_') and (file.endswith('.py') or os.path.isdir(path)):
        module = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('fairseq.criterions.' + module)
