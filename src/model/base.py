import tensorflow as tf
from keras.models import Model
from keras.metrics import Mean
from src.utils.metrics import inverted_mask_contrast_mse

class BaseModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tb_output_dir = kwargs["tb_output_dir"]
        self.model_name = kwargs["name"]
        # Metrics:
        self.total_loss_tracker = Mean(name="total_loss")
        # Test metrics
        self.test_total_loss_tracker = Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.test_total_loss_tracker,
        ]
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict = self.compute_loss(self(inputs=data,training=True))
            total_loss_mean = tf.reduce_mean(loss_dict["total_loss"])
        grads = tape.gradient(total_loss_mean, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss_dict["total_loss"])
        return {
            "loss": self.total_loss_tracker.result()}
        
    def test_step(self, data):
        loss_dict = self.compute_loss(self(inputs=data,training=False))
        self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
        return {
            "loss": self.test_total_loss_tracker.result(),
        }
        
    def compute_loss(self, inputs):
        acti_map = inputs["acti_map"]
        # tf.squeeze(inputs["predicted_acti_map"], axis=-1)
        predicted_acti_map = inputs["predicted_acti_map"]
        mask = inputs["mask"]
        loss = inverted_mask_contrast_mse(acti_map, predicted_acti_map, mask)
        return {"total_loss":loss}
    
    def call(self, inputs, training):
        raise NotImplementedError("Override the call method for inherited class")


if __name__=='__main__':
    pass