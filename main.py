import logging
import os
import argparse
import sys
import json
from datetime import timedelta
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from nmrf.data import build_train_loader, build_val_loader
from nmrf.models import build_model
from nmrf.utils import misc
import nmrf.utils.dist_utils as comm
from nmrf.utils.logger import setup_logger
from nmrf.utils import evaluation


DEFAULT_TIMEOUT = timedelta(minutes=30)


def get_args_parser():
    parser = argparse.ArgumentParser(
        f"""
        Examples:
        
        Run on single machine:
            $ {sys.argv[0]} --num-gpus 8
        
        Change some config options:
            $ {sys.argv[0]} SOLVER.IMS_PER_BATCH 8
        
        Run on multiple machines:
            (machine 0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine 1)$ {sys.argv[1]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--checkpoint-dir', default='checkpoints/sceneflow', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--eval-only', action='store_true')

    # distributed training
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details."
    )
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pair.
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER
    )

    return parser


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
        main_func,
        # Should be num_processes_per_machine, but kept for compatibility.
        num_gpus_per_machine,
        num_machines=1,
        machine_rank=0,
        dist_url=None,
        args=(),
        timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-process or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child process (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of processes per machine. When
            using GPUs, this should be the number of GPUs.
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                        e.g. "tcp://127.0.0.1:8686".
                        Can be set to "auto" to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
        timeout (timedelta): timeout of the distributed workers
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger("nmrf")
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
        local_rank,
        main_func,
        world_size,
        num_gpus_per_machine,
        machine_rank,
        dist_url,
        args,
        timeout=DEFAULT_TIMEOUT,
):
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        assert num_gpus_per_machine <= torch.cuda.device_count()
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL" if has_gpu else "GLOO",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger('nmrf')
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group.
    comm.create_local_process_group(num_gpus_per_machine)
    if has_gpu:
        torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issuees/172
    comm.synchronize()

    main_func(*args)


def build_optimizer(model, cfg):
    base_lr = cfg.SOLVER.BASE_LR
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    norm_module_types = (
        torch.nn.BatchNorm2d,
        torch.nn.InstanceNorm2d,
        torch.nn.LayerNorm,
    )
    params = []
    params_norm = []
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                params_norm.append(value)
            else:
                params.append(value)
    ret = []
    if len(params) > 0:
        ret.append({"params": params, "lr": base_lr})
    if len(params_norm) > 0:
        ret.append({"params": params_norm, "lr": base_lr, "weight_decay": weight_decay_norm})
    adamw_args = {
        "params": ret,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY
    }
    return torch.optim.AdamW(**adamw_args)


def _setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the nmrf logger
    2. Log basic information about environment, cmdline arguments, git commit, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    checkpoint_dir = args.checkpoint_dir
    if comm.is_main_process() and checkpoint_dir:
        misc.check_path(checkpoint_dir)

    rank = comm.get_rank()
    logger = setup_logger(checkpoint_dir, distributed_rank=rank, name='nmrf')

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + misc.collect_env_info())

    logger.info("git:\n {}\n".format(misc.get_sha()))
    logger.info("Command line arguments: " + str(args))

    if comm.is_main_process() and checkpoint_dir:
        path = os.path.join(checkpoint_dir, "config.yaml")
        with open(path, 'w') as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    misc.seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def setup(args):
    """
    Create config and perform basic setups.
    """
    from nmrf.config import get_cfg
    cfg = get_cfg()
    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    _setup(cfg, args)
    comm.setup_for_distributed(comm.is_main_process())
    return cfg


def evaluate(model, cfg):
    logger = logging.getLogger("nmrf")
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = build_val_loader(cfg, dataset_name)
        # build evaluator for this dataset
        evaluator = evaluation.DispEvaluator(thres=cfg.TEST.EVAL_THRESH[idx], only_valid=cfg.TEST.EVAL_ONLY_VALID[idx],
                                             max_disp=cfg.TEST.EVAL_MAX_DISP[idx], eval_prop=cfg.TEST.EVAL_PROP[idx])
        results_i = evaluation.inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            evaluation.print_csv_format(results_i)

    if len(results) == 1:
        results = list(results.values())[0]
    return results


def main(args):
    # torch.backends.cudnn.benchmark = False
    cfg = setup(args)

    model, criterion = build_model(cfg)
    model = model.to(torch.device("cuda"))

    if comm.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()]
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    logger = logging.getLogger("nmrf")
    logger.info('Number of params:' + str(num_params))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model_without_ddp.named_parameters() if p.requires_grad}, indent=2))

    optimizer = build_optimizer(model_without_ddp, cfg)

    # resume checkpoints
    start_epoch = 0
    start_step = 0
    resume = cfg.SOLVER.RESUME
    strict_resume = cfg.SOLVER.STRICT_RESUME
    no_resume_optimizer = cfg.SOLVER.NO_RESUME_OPTIMIZER
    if resume:
        logger.info('Load checkpoint: %s' % resume)

        loc = 'cuda'
        checkpoint = torch.load(resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp.load_state_dict(weights, strict=strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not no_resume_optimizer:
            logger.info('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        # evaluate
    if args.eval_only:
        evaluate(model, cfg)
        return

    # training dataset
    train_loader, train_sampler = build_train_loader(cfg)

    # training scheduler
    last_epoch = start_step if resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg.SOLVER.BASE_LR,
        cfg.SOLVER.MAX_ITER + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch
    )

    if comm.is_main_process():
        writer = SummaryWriter(args.checkpoint_dir)

    total_steps = start_step
    epoch = start_epoch
    logger.info('Start training')

    print_freq = 20
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    while total_steps < cfg.SOLVER.MAX_ITER:
        model.train()
        model_without_ddp.freeze_bn()

        # manual change random seed for shuffling every epoch
        if comm.get_world_size() > 1:
            train_sampler.set_epoch(epoch)

        header = 'Epoch: [{}]'.format(epoch)
        for sample in metric_logger.log_every(train_loader, print_freq, header, logger=logger):
            result_dict = model(sample)
            loss_dict = criterion(result_dict, sample, log=True)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            losses.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)

            optimizer.step()

            # for training status print
            metric_logger.update(lr=lr_scheduler.get_last_lr()[0])
            metric_logger.update(**loss_dict)

            lr_scheduler.step()

            if comm.is_main_process():
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v, total_steps)

            total_steps += 1

            if total_steps % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or total_steps == cfg.SOLVER.MAX_ITER:
                if comm.is_main_process():
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % cfg.SOLVER.LATEST_CHECKPOINT_PERIOD == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                if comm.is_main_process():
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if cfg.TEST.EVAL_PERIOD > 0 and total_steps % cfg.TEST.EVAL_PERIOD == 0:
                logger.info('Start validation')

                result_dict = evaluate(model, cfg)
                if comm.is_main_process():
                    for k, v in result_dict.items():
                        if isinstance(v, dict):
                            for _k, _v in v.items():
                                if isinstance(_v, dict):
                                    for __k, __v in _v.items():
                                        writer.add_scalar(f"val/{k}.{_k}.{__k}", __v, total_steps)
                                else:
                                    writer.add_scalar(f"val/{k}.{_k}", _v, total_steps)
                        else:
                            writer.add_scalar(f"val/{k}", v, total_steps)

                model.train()
                model_without_ddp.freeze_bn()

            if total_steps >= cfg.SOLVER.MAX_ITER:
                logger.info('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )