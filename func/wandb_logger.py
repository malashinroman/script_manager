# empty = None
import os.path as osp
from copy import deepcopy

import matplotlib.pyplot as plt

# local_config should be in the ../ folder
from local_config import WANDB_LOGIN

# from torch.utils.tensorboard import SummaryWriter

__WANDB_LOG__ = None

import wandb

class WandbLogger(object):
    def __init__(self, args, wandb_entity):
        super().__init__()
        self._args = args
        if args.tensorboard_folder is not None and len(args.tensorboard_folder) > 0:
            from tensorboardX.writer import SummaryWriter
            self.use_tensorboard = True
            log_dir = osp.join(self._args.output_dir, "tensorboard")
            if self.__dict__.get("_writer") is None or self._writer is None:
                print(
                    "Initializing summary writer for tensorboard "
                    "with log_dir={}".format(log_dir)
                )
                self._writer = SummaryWriter(log_dir=log_dir)
        else:
            self.use_tensorboard = False

        if args.wandb_project_name is not None:
            self.use_wandb = True
            wandb.init(
                project=args.wandb_project_name, entity=wandb_entity, config=args
            )
            wandb.run.name = args.tag + "_" + wandb.run.name
        else:
            self.use_wandb = False


def init_wandb_logger(args, wandb_entity):
    global __WANDB_LOG__
    __WANDB_LOG__ = WandbLogger(args, wandb_entity)


def update_args_from_wandb(args):
    if args.wandb_project_name is not None:
        # FIXME: need to check if args will update for sweeps
        wandb.config.update(args)
        return args
        # return wandb.config
    else:
        return args


import torch


def filter_dict_for_dump(input):
    output = {}
    for key, val in input.items():
        if type(val) is torch.Tensor:
            val = val.item()
        output[key] = val

    return output


def write_wandb_scalar(tag, scalar_value=None, global_step=None, commit=None):
    global __WANDB_LOG__
    logged = 0
    if __WANDB_LOG__ is not None:
        if __WANDB_LOG__.use_tensorboard:
            if type(tag) is dict:
                log_dict = deepcopy(filter_dict_for_dump(tag))
                if "global_step" in log_dict:
                    global_step = log_dict["global_step"]
                for key, val in log_dict.items():
                    __WANDB_LOG__._writer.add_scalar(key, val, global_step=global_step)

                # real_tag = list(log_dict.keys())[0]
                # scalar_value = log_dict[real_tag]
                # __WANDB_LOG__._writer.add_scalar(real_tag, scalar_value, global_step)

            else:
                __WANDB_LOG__._writer.add_scalar(tag, scalar_value, global_step)
            logged = 1

        if __WANDB_LOG__.use_wandb:
            if type(tag) is dict:
                # log_dict = deepcopy(filter_dict_for_dump(tag))
                #
                # log_dict["global_step"] = global_step
                wandb.log(tag, commit=commit)
            else:
                wandb.log({tag: scalar_value}, commit=commit)
            logged = 1

        if not logged:
            print(
                "WARNING: write_wandb_scalar has no effect, because logger is not initialized"
            )


def write_wandb_dict(dict, commit=None):
    global __WANDB_LOG__
    logged = 0
    if __WANDB_LOG__ is not None:
        if __WANDB_LOG__.use_wandb:
            wandb.log(dict, commit=commit)
            logged = 1

        if not logged:
            print(
                "WARNING: write_wandb has no effect, because logger is not initialized"
            )


def write_wandb_bar(
    tag, bars_val, indexes_label="classifier index", height_label="calls", commit=None
):
    if __WANDB_LOG__ is not None:
        if __WANDB_LOG__.use_wandb:
            plt.figure()
            plt.bar(
                list(range(len(bars_val))), [bars_val[i] for i in range(len(bars_val))]
            )
            plt.xlabel(indexes_label)
            plt.ylabel(height_label)
            # fig.set_size_inches(6, 3)
            # wandb.log({'rand'})
            write_wandb_dict({tag: plt}, commit=commit)
            plt.close()


def prepare_wandb(args):
    init_wandb_logger(args, WANDB_LOGIN)
    if args.wandb_project_name is not None:
        updated_args = update_args_from_wandb(args)
        return updated_args
    else:
        return args
