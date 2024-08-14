import time
import tensorflow as tf
import os

class TrainingTimeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        
def cp_cb_generator(cp_file_path):
    # cp_path = os.path.join(cfg.tb_output_dir,cp_filename)
    return tf.keras.callbacks.ModelCheckpoint(   filepath=cp_file_path,
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    mode='min',
                                                    save_best_only=True,
                                                    verbose=1)


class CustomLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, threshold1=5, threshold2=15, decay1=0.05, decay2=0.001):
        super(CustomLRScheduler, self).__init__()
        # Initialize threshold and decay rates
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.decay1 = decay1
        self.decay2 = decay2

    def on_epoch_begin(self, epoch, logs=None):
        # Check if the optimizer has 'lr' attribute
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Apply the learning rate schedule based on the current epoch
        if epoch < self.threshold1:
            new_lr = current_lr
        elif epoch < self.threshold2:
            new_lr = current_lr * tf.math.exp(-self.decay1)
        else:
            new_lr = current_lr * tf.math.exp(-self.decay2)
        # Set the new learning rate in the optimizer
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        tf.print(f"Learning rate for epoch {epoch}: {new_lr}")

    def on_epoch_end(self, epoch, logs=None):
        # Ensure logs is a dictionary
        if logs is None:
            logs = {}
        # Retrieve the current learning rate from the optimizer
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Log the current learning rate
        logs['lr'] = current_lr

if __name__=='__main__':
    pass