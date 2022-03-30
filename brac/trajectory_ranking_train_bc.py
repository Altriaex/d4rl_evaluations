# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline training binary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from absl import app
from absl import flags
from absl import logging
import shutil

import gin
import tensorflow as tf0
import tensorflow.compat.v1 as tf

from behavior_regularized_offline_rl.brac import agents
from behavior_regularized_offline_rl.brac import train_eval_offline
from behavior_regularized_offline_rl.brac import utils

import d4rl

tf0.compat.v1.enable_v2_behavior()


# Flags for offline training.
flags.DEFINE_string('exp_path',
                    os.path.join(os.getenv('HOME', '/'), 'tmp/offlinerl/learn'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('split', "1", "the split ")
#flags.DEFINE_string('agent_name', 'brac_primal', 'agent name.')
flags.DEFINE_string('game', 'halfcheetah-random-v0', 'env name.')
flags.DEFINE_integer('total_train_steps', int(1e6), '')
flags.DEFINE_integer('n_train', int(10e6), '')
flags.DEFINE_integer('value_penalty', 0, '')
flags.DEFINE_integer('save_freq', 1000, '')
flags.DEFINE_float('alpha', 1.0, '')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

def main(_):
    # Setup log dir.
    log_dir = os.path.join(
          FLAGS.exp_path,
          FLAGS.game,
          FLAGS.split,
          "BC")
    """model_arch = ((200,200),)
    opt_params = (('adam', 5e-4),)
    utils.maybe_makedirs(log_dir)
    train_eval_offline.train_eval_offline(
      log_dir=log_dir,
      data_file=None,
      agent_module=agents.AGENT_MODULES_DICT['bc'],
      env_name=FLAGS.game,
      n_train=FLAGS.n_train,
      total_train_steps=FLAGS.total_train_steps,
      n_eval_episodes=10,
      model_params=model_arch,
      optimizers=opt_params,
      seed=int(FLAGS.split),
      use_seed_for_data=True)"""
    exp_base, exp_id = osp.split(FLAGS.exp_path)
    archive_name = osp.join(
        exp_base, "agents", "_".join([exp_id, FLAGS.game, FLAGS.split, "BC"]))
    shutil.make_archive(
        base_name=archive_name,
        root_dir=log_dir,
        base_dir=None, format="zip") 

if __name__ == '__main__':
    app.run(main)
