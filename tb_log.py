"""Simple example on how to log scalars and images to tensorboard without tensor ops.
License: Copyleft
"""
__author__ = "Luke LIN"

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def add_graph(self, graph):
        self.writer.add_graph(graph)

    def add_scalar(self, tag, scalar_value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        self.writer.add_scalar(tag, scalar_value, global_step=step)

    def add_text(self, tag, value, step):
        text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
        self.writer.add_summary(summary, step)

    def add_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []

        self.writer.add_summary(summary, step)

    def add_histogram(self, tag, value, step):
        """Logs the histogram of a list/vector of values."""

        self.writer.add_histogram(tag, value, step)

    def flush(self):
        self.writer.flush()
