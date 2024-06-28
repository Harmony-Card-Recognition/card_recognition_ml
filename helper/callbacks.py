import csv
import keras

class CsvLoggerCallback(keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                # Write header on first epoch
                writer.writerow(['epoch'] + list(logs.keys()))
            writer.writerow([epoch] + list(logs.values()))



class ValidationAccuracyThresholdCallback(keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ValidationAccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy") >= self.threshold:
            self.model.stop_training = True