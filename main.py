import argparse
import convert_data
import train_image_classifier_gen
import export_inference_graph
import freeze_graph
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_dir',
    required=True,
    type=str,
    help='Path to dataset directory'
  )
  parser.add_argument(
    '--dataset_name',
    default='data',
    type=str,
    help='Dataset name'
  )
  parser.add_argument(
    '--validation_percentage',
    type=float,
    default=0.25,
    help='Percentage of dataset used as validation'
  )
  parser.add_argument(
    '--num_shards',
    type=int,
    default=5,
    help='How many shards the dataset will be fragmented'
  )
  parser.add_argument(
    '--train_dir',
    type=str,
    default='/tmp/train/',
    help='Resulting training directory'
  )
  parser.add_argument(
    '--num_clones', 
    type=int,
    default=1,
    help='Number of model clones to deploy.'
  )
  parser.add_argument(
    '--clone_on_cpu',
    type=bool, 
    default=True,
    help='Use CPUs to deploy clones.'
  )
  parser.add_argument(
    '--worker_replicas', 
    type=int,
    default=1, 
    help='Number of worker replicas.'
  )
  parser.add_argument(
    '--log_every_n_steps', 
    type=int,
    default=10,
    help='The frequency with which logs are print.'
  )
  parser.add_argument(
    '--save_interval_secs',
    type=int, 
    default=600,
    help='The frequency with which summaries are saved, in seconds.'
  )
  ## Optimization Flags ##
  parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.00004, 
    help='The weight decay on the model weights.'
  )
  parser.add_argument(
    '--optimizer', 
    type=str,
    default='rmsprop',
    help='The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".'
  )
  parser.add_argument(
    '--adadelta_rho', 
    type=float,
    default=0.95,
    help='The decay rate for adadelta.'
  )
  parser.add_argument(
    '--adagrad_initial_accumulator_value',
    type=float,
    default=0.1,
    help='Starting value for the AdaGrad accumulators.'
  )
  parser.add_argument(
    '--adam_beta1',
    type=float,
    default=0.9,
    help='The exponential decay rate for the 1st moment estimates.'
  )
  parser.add_argument(
    '--adam_beta2',
    type=float,
    default=0.999,
    help='The exponential decay rate for the 2nd moment estimates.'
  )
  parser.add_argument(
    '--opt_epsilon',
    type=float,
    default=1.0, 
    help='Epsilon term for the optimizer.'
  )
  parser.add_argument(
    '--ftrl_learning_rate_power',
    type=float,
    default=-0.5,
    help='The learning rate power.'
  )
  parser.add_argument(
    '--ftrl_initial_accumulator_value',
    type=float,
    default=0.1,
    help='Starting value for the FTRL accumulators.'
  )
  parser.add_argument(
    '--ftrl_l1',
    type=float,
    default=0.0,
    help='The FTRL l1 regularization strength.'
  )
  parser.add_argument(
    '--ftrl_l2',
    type=float,
    default=0.0,
    help='The FTRL l2 regularization strength.'
  )
  parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='The momentum for the MomentumOptimizer and RMSPropOptimizer.'
  )
  parser.add_argument(
    '--rmsprop_momentum',
    type=float,
    default=0.9, 
    help='Momentum.'
  )
  parser.add_argument(
    '--rmsprop_decay',
    type=float,
    default=0.9,
    help='Decay term for RMSProp.'
  )

  ## Learning Rate ##
  parser.add_argument(
    '--learning_rate_decay_type',
    type=str,
    default='exponential',
    help='Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"'
  )
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01, 
    help='Initial learning rate.'
  )
  parser.add_argument(
    '--end_learning_rate',
    type=float,
    default=0.0001,
    help='The minimal end learning rate used by a polynomial decay learning rate.'
  )
  parser.add_argument(
    '--label_smoothing',
    type=float,
    default=0.0,
    help='The amount of label smoothing.'
  )
  parser.add_argument(
    '--learning_rate_decay_factor',
    type=float,
    default= 0.94,
    help='Learning rate decay factor.'
  )
  parser.add_argument(
    '--num_epochs_per_decay',
    type=float,
    default=2.0,
    help='Number of epochs after which learning rate decays.'
  )
  parser.add_argument(
    '--sync_replicas', 
    type=bool,
    default=False,
    help='Whether or not to synchronize the replicas during training.'
  )
  parser.add_argument(
    '--replicas_to_aggregate',
    type=int,
    default=1,
    help='The Number of gradients to collect before updating params.'
  )
  parser.add_argument(
    '--moving_average_decay',
    default=None,
    help='The decay to use for the moving average. If left as None, then moving averages are not used.'
  )

  ## Dataset Flags ##
  parser.add_argument(
    '--dataset_split_name',
    type=str,
    default='train', 
    help='The name of the train/test split.'
  )
  parser.add_argument(
    '--labels_offset',
    type=int,
    default=0,
    help='An offset for the labels in the dataset. This flag is primarily used to '
    + 'evaluate the VGG and ResNet architectures which do not use a background '
    + 'class for the ImageNet dataset.'
  )
  parser.add_argument(
    '--model_name',
    type=str,
    default='inception_v3',
    help='The name of the architecture to train.'
  )
  parser.add_argument(
    '--preprocessing_name', 
    default=None, 
    help='The name of the preprocessing to use. If left '
    + 'as `None`, then the model_name flag is used.'
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='The number of samples in each batch.'
  )
  parser.add_argument(
    '--train_image_size', 
    default=None, 
    help='Train image size'
  )
  parser.add_argument(
    '--max_number_of_steps', 
    default=10000,
    type=int,
    help='The maximum number of training steps.'
  )
  parser.add_argument(
    '--checkpoint_path', 
    default="inception_v3.ckpt",
    help='The path to a checkpoint from which to fine-tune.'
  )
  parser.add_argument(
    '--checkpoint_exclude_scopes', 
    default="InceptionV3/Logits,InceptionV3/AuxLogits",
    help='Comma-separated list of scopes of variables to exclude when restoring '
    + 'from a checkpoint.'
  )
  parser.add_argument(
    '--trainable_scopes', 
    default="InceptionV3/Logits,InceptionV3/AuxLogits",
    help='Comma-separated list of scopes to filter the set of variables to train.'
    + 'By default, None would train all the variables.'
  )
  parser.add_argument(
    '--ignore_missing_vars', 
    default=False,
    help='When restoring a checkpoint would ignore missing variables.'
  )

  ## Freeze Graph ##
  parser.add_argument(
      "--input_saver",
      type=str,
      default="",
      help="TensorFlow saver file to load.")
  parser.add_argument(
      "--checkpoint_version",
      type=int,
      default=2,
      help="Tensorflow variable file format")
  parser.add_argument(
      "--output_graph",
      type=str,
      default="result.pb",
      required=True,
      help="Output \'GraphDef\' file name.")
  parser.add_argument(
      "--input_binary",
      nargs="?",
      const=False,
      default=True,
      help="Whether the input files are in binary format.")
  parser.add_argument(
      "--output_node_names",
      type=str,
      default="",
      help="The name of the output nodes, comma separated.")
  parser.add_argument(
      "--restore_op_name",
      type=str,
      default="save/restore_all",
      help="""\
      The name of the master restore operator. Deprecated, unused by updated \
      loading code.
      """)
  parser.add_argument(
      "--filename_tensor_name",
      type=str,
      default="save/Const:0",
      help="""\
      The name of the tensor holding the save path. Deprecated, unused by \
      updated loading code.
      """)
  parser.add_argument(
      "--clear_devices",
      nargs="?",
      const=True,
      default=True,
      help="Whether to remove device specifications.")
  parser.add_argument(
      "--initializer_nodes",
      type=str,
      default="",
      help="Comma separated list of initializer nodes to run before freezing.")
  parser.add_argument(
      "--variable_names_whitelist",
      type=str,
      default="",
      help="""\
      Comma separated list of variables to convert to constants. If specified, \
      only those variables will be converted to constants.\
      """)
  parser.add_argument(
      "--variable_names_blacklist",
      type=str,
      default="",
      help="""\
      Comma separated list of variables to skip converting to constants.\
      """)
  parser.add_argument(
      "--input_meta_graph",
      type=str,
      default="",
      help="TensorFlow \'MetaGraphDef\' file to load.")
  parser.add_argument(
      "--input_saved_model_dir",
      type=str,
      default="",
      help="Path to the dir with TensorFlow \'SavedModel\' file and variables.")
  parser.add_argument(
      "--saved_model_tags",
      type=str,
      default="serve",
      help="""\
      Group of tag(s) of the MetaGraphDef to load, in string format,\
      separated by \',\'. For tag-set contains multiple tags, all tags \
      must be passed in.\
      """)

  parser.add_argument(
    '--is_training', 
    default=False,
    help='Whether to save out a training-focused version of the model.'
  )

  ## Fine-tuning Flags ##



  #TO-DO  train_image_classifier
  '''tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
  tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
  tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
  'task', 0, 'Task id of the replica running the training.')

  '''


  args = parser.parse_args()

  dataset_dir = args.dataset_dir
  dataset_name = args.dataset_name
  validation_percentage = args.validation_percentage
  num_shards = args.num_shards
  train_dir = args.train_dir
  num_clones=args.num_clones
  worker_replicas = args.worker_replicas
  log_every_n_steps = args.log_every_n_steps
  save_interval_secs = args.save_interval_secs
  weight_decay = args.weight_decay
  moving_average_decay = args.moving_average_decay
  max_number_of_steps = args.max_number_of_steps
  print(max_number_of_steps)
  clone_on_cpu = args.clone_on_cpu


  optimization = {
    'optimizer':args.optimizer,
    'adadelta_rho': args.adadelta_rho,
    'adagrad_initial_accumulator_value': args.adagrad_initial_accumulator_value,
    'adam_beta1': args.adam_beta1,
    'adam_beta2': args.adam_beta2,
    'opt_epsilon': args.opt_epsilon,
    'ftrl_learning_rate_power': args.ftrl_learning_rate_power,
    'ftrl_initial_accumulator_value': args.ftrl_initial_accumulator_value,
    'ftrl_l1': args.ftrl_l1,
    'ftrl_l2': args.ftrl_l2,
    'momentum': args.momentum,
    'rmsprop_momentum': args.rmsprop_momentum,
    'rmsprop_decay': args.rmsprop_decay
  }

  learning_rate = {
    'learning_rate_decay_type': args.learning_rate_decay_type,
    'learning_rate': args.learning_rate,
    'end_learning_rate': args.end_learning_rate,
    'label_smoothing': args.label_smoothing,
    'learning_rate_decay_factor': args.learning_rate_decay_factor,
    'num_epochs_per_decay': args.num_epochs_per_decay,
    'sync_replicas': args.sync_replicas,
    'replicas_to_aggregate': args.replicas_to_aggregate
  }

  dataset = {
    'dataset_name': dataset_name,
    'dataset_split_name': args.dataset_split_name,
    'dataset_dir': dataset_dir,
    'labels_offset': args.labels_offset,
    'model_name': args.model_name,
    'preprocessing_name': args.preprocessing_name,
    'batch_size': args.batch_size,
    'train_image_size': args.train_image_size,
  }

  checkpoint = {
    'checkpoint_path': args.checkpoint_path,
    'checkpoint_exclude_scopes': args.checkpoint_exclude_scopes,
    'trainable_scopes': args.trainable_scopes,
    'ignore_missing_vars': args.ignore_missing_vars
  }

  export = {
    'model_name': args.model_name,
    'is_training': args.is_training,
    'image_size': args.train_image_size,
    'dataset_name': dataset_name,
    'labels_offset': args.labels_offset,
    'output_file': 'res.pb',
    'dataset_dir': dataset_dir

  }

  freeze_dict = {
    'input_graph': 'res.pb',
    'input_saver': args.input_saver,
    'input_checkpoint': os.path.join(train_dir, 
                        'model.ckpt-'+ str(max_number_of_steps)),
    'checkpoint_version': args.checkpoint_version,
    'output_graph': args.output_graph,
    'input_binary': args.input_binary,
    'output_node_names': 'InceptionV3/Predictions/Reshape_1',
    'filename_tensor_name': args.filename_tensor_name,
    'clear_devices': args.clear_devices,
    'initializer_nodes': args.initializer_nodes,
    'variable_names_whitelist': args.variable_names_whitelist,
    'variable_names_blacklist': args.variable_names_blacklist,
    'input_meta_graph': args.input_meta_graph,
    'input_saved_model_dir': args.input_saved_model_dir,
    'saved_model_tags': args.saved_model_tags,
    'restore_op_name': args.restore_op_name
  }

  ''' save json with default arguments
  j = {
    'dataset_name': args.dataset_name,
    'validation_percentage': args.validation_percentage,
    'num_shards': args.num_shards,
    'train_dir': args.train_dir,
    'num_clones': args.num_clones,
    'worker_replicas': args.worker_replicas,
    'log_every_n_steps': args.log_every_n_steps,
    'save_interval_secs': args.save_interval_secs,
    'weight_decay': args.weight_decay,
    'moving_average_decay':args.moving_average_decay,
    'max_number_of_steps': args.max_number_of_steps,
    'clone_on_cpu': args.clone_on_cpu,
    'optimization': optimization,
    'learning_rate': learning_rate,
    'dataset': dataset,
    'checkpoint': checkpoint,
    'export': export,
    'freeze_dict': freeze_dict
  }

  with open('default_arguments.json', 'w') as js:
    json.dump(j, js)'''

  train_size, val_size, num_classes = convert_data.run(dataset_dir, dataset_name, validation_percentage, num_shards)
  train_image_classifier_gen.main(train_dir, num_clones, clone_on_cpu, 
       train_size, val_size, num_classes, worker_replicas, 
       log_every_n_steps, save_interval_secs, weight_decay,
       optimization, learning_rate, moving_average_decay, 
       dataset, max_number_of_steps, checkpoint)
  export_inference_graph.main(export, train_size, val_size, num_classes)
  freeze_graph.main(freeze_dict)