from math import fabs
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import tf_utils, io_utils
from tensorflow.python.keras.backend import minimum
from tensorflow.python.platform import tf_logging as logging

import numpy as np
import pandas as pd

class EarlyStopping(keras.callbacks.Callback):
  """Stop training when a monitored metric has stopped improving.
  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.
  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.
  Args:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. An epoch will be restored regardless
        of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience`
        epochs and restore weights from the best epoch in that set.
  Example:
  >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the loss for three consecutive epochs.
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, batch_size=1, callbacks=[callback],
  ...                     verbose=0)
  >>> len(history.history['loss'])  # Only 4 epochs are run.
  4
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               minimum=None,
               restore_best_weights=False):
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None
    self.minimum = minimum

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or
          self.monitor.endswith('auc')):
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    self.best_weights = None
    self.best_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.restore_best_weights and self.best_weights is None:
      # Restore the weights after first epoch if no progress is ever made.
      self.best_weights = self.model.get_weights()

    self.wait += 1
    if self._is_improvement(current, self.best):
      self.best = current
      self.best_epoch = epoch
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
      # Only restart wait if we beat both the baseline and our previous best.
      if self.baseline is None or self._is_improvement(current, self.baseline):
        self.wait = 0

    # Only check after the first epoch.
    if self.wait >= self.patience and epoch > 0:
      self.stopped_epoch = epoch
      self.model.stop_training = True
      if self.restore_best_weights and self.best_weights is not None:
        if self.verbose > 0:
          print('Restoring model weights from the end of the best epoch: '
                f'{self.best_epoch + 1}.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning('Early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    return monitor_value

  def _is_improvement(self, monitor_value, reference_value):
    if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or self.monitor.endswith('auc')):
        if monitor_value < self.minimum:
            return True
    return self.monitor_op(monitor_value - self.min_delta, reference_value)

class Logging(keras.callbacks.Callback):
    def __init__(
        self,
        log_path
    ):
        self.train_epoch_log = []
        self.val_epoch_log = []
    
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs = None):
        if 'transtime' in self.log_path:
            key ='rmse'
        else:
            key = 'accuracy'
        self.train_epoch_log.append(logs['{}'.format(key)])
        self.val_epoch_log.append(logs['val_{}'.format(key)])

    def on_train_end(self, logs=None):
        if 'transtime' in self.log_path:
            key ='rmse'
        else:
            key = 'accuracy'
        
        df = pd.DataFrame(
            list(zip(self.train_epoch_log, self.val_epoch_log)),
            columns = [
                'train_{}'.format(key),
                'val_{}'.format(key)
            ]
        )
        df.to_csv(self.log_path + '.csv', header = 0)
                

class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(
        self,
        file_path,
        metric,
        verbose = 0,
        save_best_only = True,
        mode = 'auto',
        save_freq = 'epoch',
        save_format = 'h5'
    ):
        super(CustomModelCheckPoint, self).__init__()
        self.verbose = verbose
        self.file_path = file_path
        self.metric = metric

        if mode == 'min':
            self.best = np.Inf
            self.monitor_op = np.less
        elif mode == 'max':
            self.best = -np.Inf
            self.monitor_op = np.greater
        else:
            if 'acc' in self.metric:
                self.best = -np.Inf
                self.monitor_op = np.greater
            else:
                self.best = np.Inf
                self.monitor_op = np.less

        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.save_format = save_format
        self.epoch_since_last_save = 0

        if isinstance(self.save_freq, str):
            self.period = 1
        else:
            self.period = self.save_freq

    def _save_model(self, epoch, batch, logs):
        """Saves the model.
        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
            is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """

        logs = logs or {}
        if isinstance(self.save_freq, int) or (self.epoch_since_last_save >= self.period):
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epoch_since_last_save = 0
            file_path = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.metric)
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                            'skipping.', self.metric)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: {self.monitor} improved '
                                    f'from {self.best:.5f} to {current:.5f}, '
                                    f'saving model to {file_path}'
                                )

                            self.best = current
                            print('Saved model: {}'.format(self.best))
                            
                            

                            self.model.save(file_path, overwrite = True)
                        else:
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: '
                                    f'{self.monitor} did not improve from {self.best:.5f}'
                                )
                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f'\nEpoch {epoch + 1}: saving model to {file_path}'
                        )
                    self.model.save(file_path, overwrite = True)
            except IsADirectoryError as e:  # h5py 3.x
                raise IOError('Please specify a non-directory filepath for '
                            'ModelCheckpoint. Filepath used is an existing '
                            f'directory: {file_path}')
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                'ModelCheckpoint. Filepath used is an existing '
                                f'directory: f{file_path}')
                # Re-throw the error for any other causes.
                raise e



    def _get_file_path(self, epoch, batch, logs):
        try:
            if self.save_best_only:
                file_path = self.file_path.format(epoch = 'best')
            else:
                if batch is None or 'batch' in logs:
                    file_path = self.file_path.format(epoch = epoch + 1)
                else:
                    file_path = self.file_path.format(epoch = epoch + 1, batch = batch + 1, **logs)
            
            file_path += '.{}'.format(self.save_format)
            
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.file_path}". '
                f'Reason: {e}')

        return file_path



    def on_epoch_end(self, epoch, logs = None):
        self.epoch_since_last_save += 1
        if self.save_freq == 'epoch':
            self._save_model(epoch = epoch, batch = None, logs = logs)