import logging
import os
import time
from omegaconf import OmegaConf
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from lightning_fabric.utilities.cloud_io import get_filesystem
from termcolor import colored
import torch.distributed as dist
import torch


def convert_old_model(old_model_dict):
    if "pytorch-lightning_version" in old_model_dict:
        raise ValueError("This model is not old format. No need to convert!")
    version = pl.__version__
    epoch = old_model_dict["epoch"]
    global_step = old_model_dict["iter"]
    state_dict = old_model_dict["state_dict"]
    new_state_dict = OrderedDict()
    for name, value in state_dict.items():
        new_state_dict["model." + name] = value

    new_checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "pytorch-lightning_version": version,
        "state_dict": new_state_dict,
        "lr_schedulers": [],
    }

    if "optimizer" in old_model_dict:
        optimizer_states = [old_model_dict["optimizer"]]
        new_checkpoint["optimizer_states"] = optimizer_states

    return new_checkpoint

def load_model_weight(model, checkpoint, logger):
    state_dict = checkpoint["state_dict"]
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}

    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            logger.info("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            logger.info("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

# 替换后的 MyLightningLogger（兼容 pytorch-lightning 2.x）
class MyLightningLogger(LightningLoggerBase):
    def __init__(self, name, save_dir, **kwargs):
        # 不传入 name/save_dir 给父类以避免约束；直接调用基类构造
        super().__init__()

        self._name = name
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        # 不直接设置 self.log_dir（在新版本中可能是只读属性），使用私有属性
        self._log_dir = os.path.join(save_dir, f"logs-{self._version}")

        # filesystem（用于支持远程文件系统）
        self._fs = get_filesystem(save_dir)
        # 确保日志目录存在
        self._fs.makedirs(self._log_dir, exist_ok=True)

        # 初始化 Python logger（文件 + 控制台）
        self._init_logger()

        # 懒初始化的 experiment（TensorBoard SummaryWriter）
        self._experiment = None
        self._kwargs = kwargs

    # ----- 必要的属性/方法，供外部/Lightning 访问 -----
    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    # 公开 log_dir（不去覆盖父类可能的只读属性）
    @property
    def log_dir(self):
        return self._log_dir

    # experiment 仍然使用 rank_zero_experiment 装饰以确保只在主进程初始化
    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        # 确保只在主进程做文件创建/IO
        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                "the dependencies to use torch.utils.tensorboard "
                "(applicable to PyTorch 1.1 or higher)"
            ) from None

        # 使用我们自己的 log_dir 和传入的 kwargs
        self._experiment = SummaryWriter(log_dir=self._log_dir, **self._kwargs)
        return self._experiment

    # ----- 内部 logger 初始化 -----
    @rank_zero_only
    def _init_logger(self):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # file handler
        fh = logging.FileHandler(os.path.join(self._log_dir, "logs.txt"))
        fh.setLevel(logging.INFO)
        f_fmt = "[%(name)s][%(asctime)s]%(levelname)s: %(message)s"
        file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
        fh.setFormatter(file_formatter)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        c_fmt = (
            colored("[%(name)s]", "magenta", attrs=["bold"])
            + colored("[%(asctime)s]", "blue")
            + colored("%(levelname)s:", "green")
            + colored("%(message)s", "white")
        )
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        # attach handlers (避免重复 attach）
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    # ----- 日志/存储接口 -----
    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def dump_cfg(self, cfg_node, cfg_name):
        with open(os.path.join(self._log_dir, cfg_name), "w") as f:
            OmegaConf.save(cfg_node, f)

    @rank_zero_only
    def dump_cfg_with_dir(self, cfg_node, save_dir):
        with open(os.path.join(save_dir, "train_cfg.yml"), "w") as f:
            OmegaConf.save(cfg_node, f)

    @rank_zero_only
    def log_hyperparams(self, params):
        # pytorch-lightning 的新接口有时会传递 dict 或 Namespace，保持原行为
        self.logger.info(f"hyperparams: {params}")

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # 保持兼容：允许 step 为 None 或者数值
        self.logger.info(f"Val_metrics: {metrics}")
        # 如果有 SummaryWriter，则写入 scalars
        if self.experiment is not None:
            # 若 metrics 是 dict，则为每个 key 写 scalars
            for k, v in metrics.items():
                # 使用 add_scalars 保持原代码行为（原来是 add_scalars("Val_metrics/" + k, {"Val": v}, step)）
                try:
                    self.experiment.add_scalars("Val_metrics/" + k, {"Val": v}, step if step is not None else 0)
                except Exception:
                    # 若写入失败，记录到 logger，但不抛异常
                    self.logger.debug(f"Failed writing metric {k}:{v} to tensorboard")

    @rank_zero_only
    def save(self):
        # 这里保留接口（Lightning 可能调用），我们仅保证 SummaryWriter flush
        if self._experiment is not None:
            try:
                self._experiment.flush()
            except Exception:
                pass

    @rank_zero_only
    def finalize(self, status):
        # status 参数为训练状态（如 "success" / "failed"），保留接口
        if self._experiment is not None:
            try:
                self._experiment.flush()
                self._experiment.close()
            except Exception:
                pass
        # 如果有需要额外保存的内容可以放在这里


def gather_results(results):
    rank = -1
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    all_results = {}
    for key, value in results.items():
        shape_tensor = torch.tensor(value.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)

        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.float32, device="cuda")
        part_send[: shape_tensor[0]] = value
        part_recv_list = [value.new_zeros(shape_max, dtype=torch.float32) for _ in range(world_size)]
        dist.all_gather(part_recv_list, part_send)

        if rank < 1:
            for recv, shape in zip(part_recv_list, shape_list):
                if key not in all_results:
                    all_results[key] = recv[: shape[0]]
                else:
                    all_results[key] = torch.cat((all_results[key], recv[: shape[0]]))
    return all_results

def rank_filter(func):
    def func_filter(local_rank=-1, *args, **kwargs):
        if local_rank < 1:
            return func(*args, **kwargs)
        else:
            pass
    return func_filter

@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)