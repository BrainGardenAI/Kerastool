# coding=utf-8
import numpy as np

import keras.backend as K
from keras.callbacks import Callback


class CyclicCosineScheduler(Callback):
    """
        https://hackernoon.com/training-your-deep-model-faster-and-sharper-e85076c3b047
    """

    def __init__(self, lr0, iters_per_model, total_models, filepath, save_weights_only=True):
        self.lr0 = lr0
        self.lr = self.lr0
        self.batch_counter = 0
        self.model_counter = 0
        self.total_iters = iters_per_model*total_models
        self.total_models = total_models
        self.tm = iters_per_model
        self.filepath = filepath
        self.save_weights_only = save_weights_only

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        lr = self._lr_sheduler()
        self.lr = lr
        K.set_value(self.model.optimizer.lr, lr)
        self.checkpoint()

    def checkpoint(self):
        if self.batch_counter > 0 and np.mod(self.batch_counter, self.tm) == 0:
            self.model_counter += 1
            filepath = self.filepath.format(model=self.model_counter)
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)

    def _lr_sheduler(self):
        return 0.5*self.lr0 * (np.cos(np.pi * np.mod(self.batch_counter, self.tm) / float(self.tm)) + 1.)

    def plot_lr(self):
        import matplotlib.pyplot as plt
        _lr = [self.lr0]
        for _i in range(1, self.total_iters):
            self.batch_counter += 1
            _lr.append(self._lr_sheduler())
        plt.plot(range(self.total_iters), _lr)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    clr = CyclicCosineScheduler(0.1, 250, 10, '', save_weights_only=True)
    clr.plot_lr()
