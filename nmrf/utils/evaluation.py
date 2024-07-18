# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Union
from collections import OrderedDict, abc
import logging
import time
import datetime
from contextlib import ExitStack, contextmanager
import itertools

import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import Sampler

from nmrf.utils.logger import log_every_n_seconds
from nmrf.utils import dist_utils as comm
from nmrf.utils import frame_utils


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron2,
    so that they are easy to copypaste into a spreadsheet.


    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, abc.Mapping) or not len(results), results
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        if isinstance(res, abc.Mapping):
            important_res = [(k, v) for k, v in res.items()]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
        else:
            logger.info(f"copypaste: {task}={res}")


class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size):
        """
        Ärgs:
            size (int): the total number of data on the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_size = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_size[:rank])
        end = min(sum(shard_size[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by:meth:`process`),
    add produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that' used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if comm.is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                            k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
        model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], **kwargs
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` model instead, you can
            wrap the given model and override it behavior of `.eval()` and `.train()`.

        data_loader: an iterable objects with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        kwargs: additional arguments to model

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = comm.get_dist_info()[1]
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards,

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class DispEvaluator(DatasetEvaluator):
    """Evaluate disparity accuracy using metrics."""

    def __init__(self, thres, only_valid, max_disp=None, eval_prop=True, divis_by=8):
        """
        Args:
            thres (list[str] or None): threshold for outlier
            only_valid (bool): whether invalid pixels are excluded from evaluation
            max_disp (int or None): If None, maximum disparity will be regarded as infinity
            eval_prop (bool): whether evaluate the proposal quality.
        """
        # If True, will collect results from all ranks and return evaluation
        # in the main process. Otherwise, will evaluate the results in the current
        # process.
        self._distributed = comm.get_world_size() > 1
        self._max_disp = np.inf if max_disp is None else max_disp
        self._thres = thres
        self._only_valid = only_valid
        self._eval_prop = eval_prop
        self._divis_by = divis_by

    def reset(self):
        self._epe = []
        self._thres_metric = OrderedDict()
        self._d1 = []

        if self._thres is not None:
            for t in self._thres:
                self._thres_metric[t] = []

        self._prop_epe = []
        self._prop_recall_3 = []
        self._prop_recall_8 = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to the model. It is a dict, which contains keys like "img1",
                "ïmg2", "disp".
            outputs: the outputs of the model. It is a dict.
        """
        inputs = [dict(zip(inputs, t)) for t in zip(*inputs.values())]
        outputs = [dict(zip(outputs, t)) for t in zip(*outputs.values())]
        for input, output in zip(inputs, outputs):
            disp_pr = output['disp']
            disp_gt = input['disp'].to(disp_pr.device)
            valid_gt = input['valid'].to(disp_pr.device)
            if self._only_valid:
                valid = valid_gt & (disp_gt < self._max_disp)
            else:
                valid = disp_gt < self._max_disp
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)

            epe = torch.abs(disp_pr - disp_gt)
            epe = epe.flatten()
            val = valid.flatten()

            if (np.isnan(epe[val].mean().item())):
                continue

            self._epe.append(epe[val].mean().item())
            self._d1.append(((epe[val] > 3) & (epe[val] / disp_gt.flatten()[val] > 0.05)).float().mean().item())

            if len(self._thres_metric) > 0:
                for t in self._thres:
                    tf = float(t)
                    out = (epe > tf)
                    self._thres_metric[t].append(out[val].float().mean().item())

            if self._eval_prop:
                proposal = output['proposal'] * 8
                superpixel_label = input['super_pixel_label'].to(disp_pr.device)
                disp_gt_clone = disp_gt.clone()
                disp_gt_clone[~valid_gt] = 0
                mini_disp_gt = frame_utils.downsample_disp(disp_gt_clone[None], superpixel_label[None])[0]
                im_h, im_w = disp_gt.shape[:2]
                _im_h = int((im_h + self._divis_by - 1) // self._divis_by * self._divis_by // 8)
                _im_w = int((im_w + self._divis_by - 1) // self._divis_by * self._divis_by // 8)
                ht, wd = mini_disp_gt.shape[:2]
                _, num_proposals = proposal.shape
                proposal = proposal.reshape(_im_h, _im_w, num_proposals)
                proposal = proposal[:ht, :wd, :].reshape(-1, num_proposals)
                mini_disp_gt = mini_disp_gt.flatten(end_dim=1)
                epe = torch.cdist(mini_disp_gt[..., None], proposal[..., None], p=1)
                epe[mini_disp_gt == 0, :] = 1e6
                epe, _ = torch.min(epe.flatten(start_dim=1), dim=1)
                mask = (((mini_disp_gt > 0) & (mini_disp_gt < self._max_disp)).sum(dim=-1)) > 0.5
                if np.isnan(epe[mask].mean().item()):
                    continue
                self._prop_epe.append(epe[mask].mean().item())
                self._prop_recall_3.append((epe[mask] <= 3).float().mean().item())
                self._prop_recall_8.append((epe[mask] <= 8).float().mean().item())

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            epe = list(itertools.chain(*comm.gather(self._epe, dst=0)))
            d1 = list(itertools.chain(*comm.gather(self._d1, dst=0)))
            thres_metric = OrderedDict()
            for k, v in self._thres_metric.items():
                thres_metric[k] = list(itertools.chain(*comm.gather(v, dst=0)))
            prop_epe = list(itertools.chain(*comm.gather(self._prop_epe, dst=0)))
            prop_recall_3 = list(itertools.chain(*comm.gather(self._prop_recall_3, dst=0)))
            prop_recall_8 = list(itertools.chain(*comm.gather(self._prop_recall_8, dst=0)))
            if not comm.is_main_process():
                return {}
        else:
            epe = self._epe
            d1 = self._d1
            thres_metric = self._thres_metric
            prop_epe = self._prop_epe
            prop_recall_3 = self._prop_recall_3
            prop_recall_8 = self._prop_recall_8

        epe = torch.tensor(epe).mean().item()
        d1 = torch.tensor(d1).mean().item() * 100
        res = {'epe': epe, 'd1': d1}
        for k, v in thres_metric.items():
            res[f'bad {k}'] = torch.tensor(v).mean().item() * 100
        if self._eval_prop:
            res['prop_epe'] = torch.tensor(prop_epe).mean().item()
            res['prop_recall_3'] = torch.tensor(prop_recall_3).mean().item() * 100
            res['prop_recall_8'] = torch.tensor(prop_recall_8).mean().item() * 100

        results = {'disp': res}
        return results