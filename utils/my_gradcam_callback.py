"""
Callback Module for Grad CAM
 changed
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tf_explain.core.grad_cam import GradCAM


class MyGradCAMCallback(Callback):

    """
    Perform Grad CAM algorithm for a given input
    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def __init__(
        self,
        validation_data,
        class_index,
        output_dir=Path("./logs/grad_cam"),
        use_guided_grads=True,
        limit=-1,
    ):
        """
        Constructor.
        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            class_index (int): Index of targeted class
            #layer_name (str): Targeted layer for GradCAM
            output_dir (str): Output directory path
            limit (int): Maximum amount of images to use
        """
        super(MyGradCAMCallback, self).__init__()
        self.validation_data = validation_data
        self.layer_name = None
        self.class_index = class_index
        self.output_dir = output_dir
        self.use_guided_grads = use_guided_grads
        self.limit = limit
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

    def get_layer_name(self):
        gradcam_layer = None
        for layer in self.model.layers:
            if "conv" in layer.name.lower():
                gradcam_layer = layer.name
        return gradcam_layer


    def on_epoch_end(self, epoch, logs=None):
        """
        Draw GradCAM outputs at each epoch end to Tensorboard.
        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        if self.layer_name is None:
            self.layer_name = self.get_layer_name()
        explainer = GradCAM()
        heatmap = explainer.explain(
            (self.validation_data[0][:self.limit],self.validation_data[1][:self.limit]),
            self.model,
            class_index=self.class_index,
            layer_name=self.layer_name,
            #use_guided_grads=self.use_guided_grads,
        )

        explainer.save(heatmap, self.output_dir, f"epoch_{epoch}_limit_{self.limit}.jpg")