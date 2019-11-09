# Based on https://github.com/lanpa/tensorboardX/blob/master/examples/demo.py
import nnabla as nn
import numpy as np
from nnabla.models.imagenet import ResNet18
from nnabla.monitor import tile_images

from nnabla_tensorboard import SummaryWriter


def demo_histogram(writer):
    """ Show how to use add_histogram() to add histogram to tensorboard for NNabla.

    :param writer: nnabla_tensorboard summary writer.
    :return: nothing.
    """
    nn.clear_parameters()

    model = ResNet18()
    image = nn.Variable([1, 3, 224, 224])
    pred = model(image, training=False)

    for name, value in nn.get_parameters().items():
        if 'bn' not in name:
            # Add the histogram of parameter to tensorboard for NNabla.
            writer.add_histogram(name, value)


def demo_scalar(writer, n_iter):
    """ Show how to add_scalar() and add_scalars() to add scalar(s) to tensorboard for NNabla.

    :param writer: nnabla_tensorboard summary writer.
    :param n_iter: global_step for tensorboard.
    :return: nothing.
    """
    dummy_s1 = nn.Variable.from_numpy_array(np.random.rand(1))
    dummy_s2 = nn.Variable.from_numpy_array(np.random.rand(1))

    # data grouping by `slash`
    writer.add_scalar('data/scalar_systemtime', dummy_s1[0], n_iter)
    # data grouping by `slash`
    writer.add_scalar('data/scalar_customtime', dummy_s2[0], n_iter, walltime=n_iter)
    writer.add_scalars('data/scalar_group', {"xsinx": n_iter * np.sin(n_iter),
                                             "xcosx": n_iter * np.cos(n_iter),
                                             "arctanx": np.arctan(n_iter)}, n_iter)


def demo_image(writer, x, n_iter):
    """ Show how to add_image() to add image to tensorboard for NNabla.

    :param writer: nnabla_tensorboard summary writer.
    :param x: NNabla variable tensor with the format [B, C, H, W].
    :param n_iter: global_step for tensorboard.
    :return: nothing.
    """
    tiled = tile_images(x.d)  # return numpy.ndarray
    writer.add_image('Image', tiled, n_iter, dataformats='HWC')


def demo_pr_curve(writer, n_iter):
    """ Show how to add_pr_curve() and add_pr_curve_raw() to add precision recall curve to tensorboard for NNabla.

    :param writer: nnabla_tensorboard summary writer.
    :param n_iter: global_step for tensorboard.
    :return: nothing.
    """
    # Adds precision recall curve.
    writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(
        100), n_iter)  # needs tensorboard 0.4RC or later

    # Adds precision recall curve with raw data.
    true_positive_counts = [75, 64, 21, 5, 0]
    false_positive_counts = [150, 105, 18, 0, 0]
    true_negative_counts = [0, 45, 132, 150, 150]
    false_negative_counts = [0, 11, 54, 70, 75]
    precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
    recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]

    writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
                            false_positive_counts,
                            true_negative_counts,
                            false_negative_counts,
                            precision,
                            recall, n_iter)


def demo_text(writer, n_iter):
    """ Show how to add_text() to add text to tensorboard for NNabla.

    :param writer: nnabla_tensorboard summary writer. nnabla_tensorboard summary writer.
    :param n_iter: global_step for tensorboard.
    :return: nothing.
    """
    writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
    writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)


def demo():
    writer = SummaryWriter()
    nn.set_auto_forward(True)
    demo_histogram(writer)

    for n_iter in range(100):
        demo_scalar(writer, n_iter)

        x = nn.Variable.from_numpy_array(np.random.random([32, 3, 64, 64]))  # output from network (dummy image)

        if n_iter % 10 == 0:
            demo_image(writer, x, n_iter)
            demo_text(writer, n_iter)

        demo_pr_curve(writer, n_iter)

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    demo()
