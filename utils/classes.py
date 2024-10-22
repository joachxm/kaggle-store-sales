from collections import namedtuple
from collections.abc import Callable

import numpy as np
import polars as pl
import polars.selectors as cs
import tensorflow as tf

TARGET_STEPS = 16

# namedtuple to store the order of the axes in the data tensors
# avoids having axes numbers in the code
Axes = namedtuple("Axes", ["TIME", "KEY", "FEATURE"], defaults=[0, 1, 2])
AX = Axes()


# TODO docstring


class DataContainer:
    """ """

    def __init__(self, df: pl.DataFrame, split: tuple[float, float] = (0.7, 0.2)):
        """ """

        # cast 'date' column to obtain a timestamp (one value per day)
        df = df.with_columns(pl.col("date").cast(pl.Int32)).select(cs.numeric())

        self.features = df.columns

        # partition by key into dataframes each containing the data for a single key
        kdfs = df.partition_by(
            by=["store_nbr", "family_nbr"], maintain_order=True, include_key=True
        )

        # stack the kdfs along a new key axis -> axes = (time, key, feature)
        data = tf.stack(
            [tf.constant(kdf, dtype=tf.float32) for kdf in kdfs], axis=AX.KEY
        )

        # compute the number of timesteps in each of the train/valid/test/target sets
        train_ts = data.shape[AX.TIME] - TARGET_STEPS  # to split into train/valid/test
        self.split_steps = [int(train_ts * rt) for rt in split]  # train/valid lengths
        self.split_steps += [train_ts - sum(self.split_steps)]  # test length
        self.split_steps += [TARGET_STEPS]  # target set

        # normalize the data and store the 'sales' mean and std for unscaling
        data, self.mean, self.std = self._normalize(data)

        # split the data and store the parts in a dict
        train, valid, test, target = tf.split(data, self.split_steps, axis=AX.TIME)
        self._data = dict(train=train, valid=valid, test=test, target=target)

    def _normalize(self, data: tf.Tensor) -> tf.Tensor:
        """Normalize a data tensor.

        The temporal features are scaled along the time axis by the mean and standard deviation
        of the training set.
        The time-independent features (the categorical features which depend only on the key)
        are scaled along the key axis, in order to reduce the range of these values.

        Args:
            data: a tensor containing the data, with axes specified by `AX`

        Returns:
            The normalized `data` tensor and the mean and standard deviation of the 'sales' column.
        """

        # split the 4 features depending only on the key
        time_data, keys_data = tf.split(data, [-1, 4], axis=AX.FEATURE)

        # normalize keys data along the key axis
        keys_mean, keys_std = _scaling_parameters(keys_data, axis=AX.KEY)
        keys_data = (keys_data - keys_mean) / keys_std

        # normalize temporal data along the time axis using train set scaling parameters
        train_data, _, _, _ = tf.split(time_data, self.split_steps, axis=AX.TIME)
        time_mean, time_std = _scaling_parameters(train_data, axis=AX.TIME)
        time_data = (time_data - time_mean) / time_std

        # concatenate to recover the orignial tensor shape
        data = tf.concat([time_data, keys_data], axis=AX.FEATURE)

        # extract the 'sales' mean and std for unscaling
        sales_ind = self.features.index("sales")
        mean = tf.gather(time_mean, [sales_ind], axis=AX.FEATURE)
        std = tf.gather(time_std, [sales_ind], axis=AX.FEATURE)

        return data, mean, std

    def __getitem__(self, key: str) -> tf.Tensor:
        return self._data[key]


class WindowDatasets:
    """ """

    def __init__(
        self,
        data: DataContainer,
        input_steps: int,
        split: Callable,
        batch_size: int = 64,
        buffer_size: int = 1000,
    ):
        """ """

        self.input_steps = input_steps
        self.split = split
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.data = data
        self.window_steps = input_steps + TARGET_STEPS
        self.target_ind = data.features.index("sales")

    def make(self, key: str) -> tf.data.Dataset:
        """ """
        # TODO Mention the repeat

        # spec = [keys, features], card = timesteps
        ds = tf.data.Dataset.from_tensor_slices(self.data[key])

        # spec = [window_steps, keys, features], card = windows
        ds = ds.window(size=self.window_steps, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda window: window.batch(self.window_steps))

        # spec = [keys, window_steps, features], card = windows
        ds = ds.map(lambda xs: tf.transpose(xs, perm=[1, 0, 2]))

        # spec = [window_steps, features], card = keys * windows
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)

        return (
            ds.shuffle(self.buffer_size)  # shuffle before batching
            .batch(self.batch_size)  # batch before splitting for efficiency
            .map(self.split, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()  # use repeat + take to set the cardinality
            .take(self.len(key))
            .prefetch(tf.data.AUTOTUNE)
        )

    def len(self, key: str) -> int:
        """ """

        shape = self.data[key].shape
        windows_per_key = shape[AX.TIME] - self.window_steps + 1
        examples = shape[AX.KEY] * windows_per_key

        return int(np.ceil(examples / self.batch_size))


# ########################################################################### #


def _scaling_parameters(
    xs: tf.Tensor, axis: int, eps: float = 0.1
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute the mean and standard deviation of a tensor along an axis.

    Clip values of the standard deviation less than `eps` to avoid exploding
    values when normalizing.

    Args:
        xs: a tensor containing data to be normalized.
        axis: an axis of `xs`.
        eps: threshold for standard deviation clipping.

    Returns:
        A tuple (mean, std) with the same axes as `xs`.
    """

    mean = tf.reduce_mean(xs, axis=axis, keepdims=True)
    std = tf.math.reduce_std(xs, axis=axis, keepdims=True)
    std = tf.where(std < eps, tf.ones_like(std), std)

    return mean, std
