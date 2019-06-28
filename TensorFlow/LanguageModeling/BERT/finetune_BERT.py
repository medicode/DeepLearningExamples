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

from fathomtf.utils.tfutils import debug_tfprint


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

flags.DEFINE_integer(
    "eval_frequency_steps", 10,
    "Number of training steps per gpu between evals.")

flags.DEFINE_integer(
    "warmup_steps", 10,
    "Number of training steps to perform linear learning rate warmup for. ")

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
class _LogSessionRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, display_every=10, hvd_rank=-1):
    self.global_batch_size = global_batch_size
    self.display_every = display_every
    self.hvd_rank = hvd_rank

  def after_create_session(self, session, coord):
    if self.hvd_rank <= 0:
      if FLAGS.use_fp16:
        print('  Step samples/sec   Loss  Learning-rate  Loss-scaler')
      else:
        print('  Step samples/sec   Loss  Learning-rate')
    self.elapsed_secs = 0.
    self.count = 0

  def before_run(self, run_context):
    self.t0 = time.time()
    if FLAGS.use_fp16:
      return tf.train.SessionRunArgs(
          fetches=['step_update:0', 'total_loss:0',
                   'learning_rate:0', 'loss_scale:0'])
    else:
      return tf.train.SessionRunArgs(
          fetches=['step_update:0', 'total_loss:0', 'learning_rate:0'])

  def after_run(self, run_context, run_values):
    self.elapsed_secs += time.time() - self.t0
    self.count += 1
    if FLAGS.use_fp16:
      global_step, total_loss, lr, loss_scaler = run_values.results
    else:
      global_step, total_loss, lr = run_values.results
    print_step = global_step + 1 # One-based index for printing.
    if print_step == 1 or print_step % self.display_every == 0:
        dt = self.elapsed_secs / self.count
        img_per_sec = self.global_batch_size / dt
        if self.hvd_rank >= 0:
          if FLAGS.use_fp16:
            print('%2d :: %6i %11.1f %6.4e     %6.4e  %6.4e' %
                  (self.hvd_rank, print_step, img_per_sec, total_loss, lr, loss_scaler))
          else:
            print('%2d :: %6i %11.1f %6.4f     %6.4e' %
                  (self.hvd_rank, print_step, img_per_sec, total_loss, lr))
        else:
          if FLAGS.use_fp16:
            print('%6i %11.1f %6.4f     %6.4e  %6.4e' %
                  (print_step, img_per_sec, total_loss, lr, loss_scaler))
          else:
            print('%6i %11.1f %6.4f     %6.4e' %
                  (print_step, img_per_sec, total_loss, lr))
        self.elapsed_secs = 0.
        self.count = 0


class _OomReportingHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=[],  # no extra fetches
                              options=tf.RunOptions(
                                  report_tensor_allocations_upon_oom=True))


class InitBertHook(tf.train.SessionRunHook):
    def __init__(self, initialize_bert, init_checkpoint, hvd = None):
        self._initialize_bert = initialize_bert
        self._init_checkpoint = init_checkpoint
        self._hvd = hvd

    def begin(self):
        if not self._initialize_bert:
            return

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if self._init_checkpoint and (self._hvd is None or self._hvd.rank() == 0):
          (assignment_map, initialized_variable_names
          ) = modeling.get_assignment_map_from_checkpoint(tvars, self._init_checkpoint)
          tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          print(" %d name = %s, shape = %s%s", 0 if hvd is None else hvd.rank(), var.name, var.shape, init_string)


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

  # [B, T/chunk_size, D]
  body_output = model.get_sequence_output()

  top_out = target_modality.top(body_outputs, None)

  num, den = target_modality.loss(top_out, labels)
  loss = num / den

  return loss, top_out['logits']


def model_fn_builder(bert_config, learning_rate,
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
    # for logging hook to pick up
    total_loss = tf.identity(total_loss, name='total_loss')

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          hvd, amp=use_fp16)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=None)
    elif mode == tf.estimator.ModeKeys.EVAL:

      #logits = debug_tfprint('logits', logits)
      #label_ids = debug_tfprint('label_ids', label_ids)
      def metric_fn(_logits, _labels):

          return {
              name: call(_logits, _labels)
              for name, call in problem.all_metrics_fns.items()
              if name in problem.eval_metrics()}

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(metric_fn, [logits, label_ids]),
          scaffold_fn=None)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=None)
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

  # train config
  global_batch_size = FLAGS.train_batch_size
  # max train steps
  num_train_steps = 1e7
  num_warmup_steps = FLAGS.warmup_steps
  eval_frequency_steps = FLAGS.eval_frequency_steps

  tf.gfile.MakeDirs(FLAGS.output_dir)

  master_process = True
  training_hooks = []

  hvd_rank = -1

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
      hvd_rank = hvd.rank()
      master_process = (hvd_rank == 0)
      config.gpu_options.allow_growth = True
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      if hvd.size() > 1:
          training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

      num_train_steps //= hvd.size()
      num_warmup_steps //= hvd.size()

  if FLAGS.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      session_config=config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
      # so we only use our hook
      log_step_count_steps=100000000000,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

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
      learning_rate=learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
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
      #eval_batch_size=1,
      predict_batch_size=FLAGS.predict_batch_size)

  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams, None if not FLAGS.horovod else hvd)
  training_hooks.append(_LogSessionRunHook(global_batch_size, 100, hvd_rank))

  #training_hooks.append(_OomReportingHook())

  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL,
      hparams,
      None if not FLAGS.horovod else hvd)

  # https://github.com/horovod/horovod/issues/182#issuecomment-401486859
  # TODO: replace with ValidationMonitor and EarlyStoppingHook
  for i in range(10):
  #for i in [0]:
      from gcloud.gcs import fhfile
      END_EXT = '.meta'
      candidates = list(filter(
          lambda path: path.startswith('model.ckpt'),
          (os.path.basename(f) for f in fhfile.walk_path(
              location=FLAGS.output_dir,
              depth=1,
              extension=END_EXT))))
      if candidates:
          print('checkpoints exist', candidates)
          print('do not initialize bert')
      else:
          print('initialize bert')

      # TODO: we should use a check on model_dir to decide if we initialize_bert
      init_bert_hook = InitBertHook(
          initialize_bert=not candidates,
          init_checkpoint=FLAGS.init_checkpoint,
          hvd=hvd)

      if master_process:
          tf.logging.info("***** Running training *****")

      # TODO: move init from checkpoint to a InitHook
      # should restore parts of the graph on the begin call but only
      # on first loop
      estimator.train(
          input_fn=train_input_fn,
          hooks=training_hooks + [init_bert_hook],
          # TODO: LR dependent on train steps, are we resetting this every time then?
          steps=eval_frequency_steps)

      if master_process:
          tf.logging.info("***** Running eval *****")
          result = estimator.evaluate(input_fn=eval_input_fn, steps=None)
          #result = estimator.evaluate(input_fn=eval_input_fn, steps=1)
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
