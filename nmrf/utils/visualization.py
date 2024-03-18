import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch
import cv2


def plot_disparity(savename, data, max_disp):
    plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap='turbo')


def plot_gradient_map(savename, data):
    data = (data + 1.0) / 2
    data = 255 * data
    data = data.astype(np.uint8)
    plt.imsave(savename, data)


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    return cols


def disp_error_img(save_name, pred, gt, abs_thres=3., rel_thres=0.05):
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    H, W = pred_np.shape
    # valid mask
    mask = gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(gt_np - pred_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / gt_np[mask]) / rel_thres)
    # get colormap
    cols = gen_error_colormap()
    # create error image
    error_image = np.zeros([H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left corner of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    error_image = error_image.astype(np.uint8)
    plt.imsave(save_name, error_image)


def gen_kitti_cmap():
    map = np.array([[0, 0, 0, 114],
                    [0, 0, 1, 185],
                    [1, 0, 0, 114],
                    [1, 0, 1, 174],
                    [0, 1, 0, 114],
                    [0, 1, 1, 185],
                    [1, 1, 0, 114],
                    [1, 1, 1, 0]])

    bins = map[:-1, 3]
    cbins = np.cumsum(bins)
    cbins = cbins[:-1] / cbins[-1]
    nodes = np.concatenate([np.array([0]), cbins, np.array([1])])
    colors = map[:, :3]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(name="kitti", colors=list(zip(nodes, colors)))
    return cmap


mpl.colormaps.register(gen_kitti_cmap())


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range[0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False, dpi=600)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = bug.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:
    """
    Visualizer that draws data about disparity on images.

    It contains methods like `draw_{uncertainty,disp,normal,error}`
    that draws primitive objects to images in some pre-defined style.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    def __init__(self, img_rgb, scale=1.0):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range of [0, 255].
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

    def draw_uncertainty(self, uncertainty, alpha=0.8):
        """
        Draw disparity estimation uncertainty.

        Args:
            uncertainty (Tensor or ndarray): the uncertainty of shape (H, W).
                Each value is the uncertainty prediction of the pixel. Should already
                be normalized in the range [0, 1]
            alpha (float): the larger it is, the more opaque the uncertainties are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.numpy()
        uncertainty = (uncertainty * 255).astype("uint8")
        heatmap = cv2.applyColorMap(uncertainty, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(heatmap, alpha, self.img, 1.0 - alpha, 0.0)
        self.output.ax.imshow(blended, extent=(0, self.output.width, self.output.height, 0))
        return self.output

    def draw_error_map(self, error):
        """
        Draw error map

        Args:
            error (Tensor or ndarray): the prediction of shape (H, W)

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(error, torch.Tensor):
            error = error.numpy()
        H, W = error.shape
        error = error / 3
        # get colormap
        cols = gen_error_colormap()
        # create error image
        error_image = np.zeros([H, W, 3], dtype=np.float32)
        for i in range(cols.shape[0]):
            error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
        self.output.ax.imshow(error_image.astype(np.uint8), extent=(0, self.output.width, self.output.height, 0))
        return self.output

    def draw_disparity(self, disparity_map, colormap, enhance=True, percentile=0.01):
        if isinstance(disparity_map, torch.Tensor):
            disparity_map = disparity_map.numpy()
        norm_disparity_map = ((disparity_map - np.min(disparity_map)) / (np.max(disparity_map) - np.min(disparity_map)))
        # img = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_VIRIDIS)
        # norm_disparity_map = np.clip(disparity_map / 192, 0, 1)
        if enhance:
            norm_disparity_map = torch.from_numpy(norm_disparity_map)
            log_disp = torch.log(1.0 - norm_disparity_map + 1e-8)  # Increase visualization contrast
            to_use = log_disp.view(-1)

            mi, ma = torch.quantile(to_use, torch.FloatTensor([percentile, 1-percentile]).to(to_use.device))
            log_disp = (log_disp - mi) / (ma - mi + 1e-10)
            log_disp = torch.clip(1.0 - log_disp, 0, 1)
            norm_disparity_map = log_disp.numpy()
        if isinstance(colormap, str):
            cm = mpl.colormaps[colormap]
            img = cm(norm_disparity_map)
            img = (255 * img).astype(np.uint8)
        else:
            img = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), colormap)
        # img = cv2.applyColorMap(norm_disparity_map, cv2.COLORMAP_JET)
        self.output.ax.imshow(img, extent=(0, self.output.width, self.output.height, 0))
        return self.output