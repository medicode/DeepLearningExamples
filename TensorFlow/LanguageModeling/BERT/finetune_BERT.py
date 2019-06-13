"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tensorflow as tf
import horovod.tensorflow as hvd
import time

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("tmp_dir", None, '')

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")

flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")


# report samples/sec, total loss and learning rate during training
class _LogEvalRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, hvd_rank=-1):
    self.global_batch_size = global_batch_size
    self.hvd_rank = hvd_rank
    self.total_time = 0.0
    self.count = 0

  def before_run(self, run_context):
    self.t0 = time.time()

  def after_run(self, run_context, run_values):
    elapsed_secs = time.time() - self.t0
    self.total_time += elapsed_secs
    self.count += 1

# report samples/sec, total loss and learning rate during training
class _LogTrainRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, hvd_rank=-1):
    self.global_batch_size = global_batch_size
    self.hvd_rank = hvd_rank
    self.total_time = 0.0
    self.count = 0

  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.train.SessionRunArgs(
        fetches=['step_update:0'])
  def after_run(self, run_context, run_values):
    elapsed_secs = time.time() - self.t0
    self.total_time += elapsed_secs
    self.count += 1


class _OomReportingHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=[],  # no extra fetches
                              options=tf.RunOptions(
                                  report_tensor_allocations_upon_oom=True))


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, use_one_hot_embeddings, hparams):
  """Creates a classification model."""
  target_modality = hparams.problem_hparams.target_modality
  input_modality = hparams.problem_hparams.input_modality

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      compute_type=tf.float32)

  # [B, 384, D]
  body_outputs = model.get_sequence_output()
  #extended_batch_size = tf.shape(body_outputs)[0]
  #chunk_size = tf.shape(body_outputs)[1]
  #depth = tf.shape(body_outputs)[2]
  #batch_size = extended_batch_size / chunk_size

  #body_outputs = tf.reshape(body_outputs, [batch_size, extended_batch_size, depth])
  body_outputs = tf.expand_dims(body_outputs, axis=-2)

  top_out = target_modality.top(body_outputs, None)

  num, den = target_modality.loss(top_out, labels)
  loss = num / den

  return loss, top_out['logits']


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, hparams, problem, hvd=None, use_fp16=False):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["targets"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, logits) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        use_one_hot_embeddings, hparams)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint and (hvd is None or hvd.rank() == 0):
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info(" %d name = %s, shape = %s%s", 0 if hvd is None else hvd.rank(), var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          hvd, amp=use_fp16)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      #logits.update({'labels': labels})
      eval_metrics = lambda logits, labels: {
          name: call(logits, labels)
          for name, call in problem.all_metrics_fns.items()
          if name in problem.eval_metrics()}
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(eval_metrics, [logits, labels]),
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.use_fp16:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
    print('Turning on AMP')
  else:
    print('NOT Turning on AMP')

  if FLAGS.horovod:
    hvd.init()

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  training_hooks = []
  global_batch_size = FLAGS.train_batch_size
  hvd_rank = 0

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  config = tf.ConfigProto()
  learning_rate = FLAGS.learning_rate
  if FLAGS.horovod:
      tf.logging.info("Multi-GPU training with TF Horovod")
      tf.logging.info("hvd.size() = %d hvd.rank() = %d", hvd.size(), hvd.rank())
      global_batch_size = FLAGS.train_batch_size * hvd.size()
      learning_rate = learning_rate * hvd.size()
      master_process = (hvd.rank() == 0)
      hvd_rank = hvd.rank()
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      if hvd.size() > 1:
          training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

  if FLAGS.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      session_config=config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      log_step_count_steps=1,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = 2000
  num_warmup_steps = 1
  eval_frequency_steps = 100
  assert num_train_steps % eval_frequency_steps == 0
  train_eval_iterations = num_train_steps // eval_frequency_steps
  eval_steps = 100

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  from tensor2tensor.utils.trainer_lib import create_hparams, add_problem_hparams
  import fathomt2t
  from fathomt2t.common_flags import setup_dataset_flag
  import fathomt2t.problems.fprecord_text_problem
  print('FLAGS', FLAGS)
  print('code mapping file', FLAGS.code_mapping_file)
  #problem_name = 'icd10_diagnosis_hcpcs_coding_problem_with_hints'
  problem_name = 'bert_problem'
  hparams_set = 'finetune_bert'
  setup_dataset_flag()
  FLAGS.dataset_split = 'train'

  hparams = create_hparams(hparams_set=hparams_set, problem_name=problem_name)
  #problem = registry.problem(problem_name)
  add_problem_hparams(hparams, problem_name)
  target_modality = hparams.target_modality
  problem = hparams.problem

  hparams.data_dir = FLAGS.data_dir
  ## INGEST
  #problem.generate_data(FLAGS.data_dir, FLAGS.tmp_dir)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      hparams=hparams,
      problem=problem,
      hvd=None if not FLAGS.horovod else hvd,
      use_fp16=FLAGS.use_fp16)

  #from tensor2tensor.bin.t2t_trainer import create_run_config
  #run_config = create_run_config(hparams)
  estimator = tf.contrib.tpu.TPUEstimator(
  #estimator = tf.estimator.Estimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  tf.logging.info("***** Running training *****")
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams, None if not FLAGS.horovod else hvd)
  #train_input_fn = problem.horovod_input_fn_builder(
      #mode=tf.estimator.ModeKeys.TRAIN, hparams=hparams,
      #hvd=None if not FLAGS.horovod else hvd)
  training_hooks.append(_LogTrainRunHook(global_batch_size, hvd_rank))

  #training_hooks.append(_OomReportingHook())

  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL,
      hparams,
      None if not FLAGS.horovod else hvd)

  if FLAGS.horovod:
      barrier = hvd.allreduce(tf.constant(0))
      with tf.Session(config=config) as sess:
          sess.run(barrier)

  # https://github.com/horovod/horovod/issues/182#issuecomment-401486859
  for n in range(train_eval_iterations):
      if not FLAGS.horovod or hvd.rank() != 0:
          estimator.train(
              input_fn=train_input_fn,
              hooks=training_hooks,
              # TODO: LR dependent on train steps, are we resetting this every time then?
              steps=num_train_steps)

      if not FLAGS.horovod or hvd.rank() == 0:
          result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
              tf.logging.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
  #flags.mark_flag_as_required("data_dir")
  #flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
