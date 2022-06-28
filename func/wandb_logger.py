# empty = None
import os.path as osp
from copy import deepcopy

import wandb
from local_config import WANDB_LOGIN

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX.writer import SummaryWriter

__WANDB_LOG__ = None


class WandbLogger(object):
    def __init__(self, args, wandb_entity):
        super().__init__()
        self._args = args
        if args.tensorboard_folder is not None:
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
        return wandb.config
    else:
        return args


def write_wandb_scalar(tag, scalar_value=None, global_step=None):
    global __WANDB_LOG__
    logged = 0
    if __WANDB_LOG__ is not None:
        if __WANDB_LOG__.use_tensorboard:
            if type(tag) is dict:
                log_dict = deepcopy(tag)
                real_tag = list(log_dict.keys())[0]
                scalar_value = log_dict[real_tag]
                if "global_step" in log_dict:
                    global_step = log_dict["global_step"]
                __WANDB_LOG__._writer.add_scalar(real_tag, scalar_value, global_step)
            else:
                __WANDB_LOG__._writer.add_scalar(tag, scalar_value, global_step)
            logged = 1

        if __WANDB_LOG__.use_wandb:
            if type(tag) is dict:
                log_dict = deepcopy(tag)

                log_dict["global_step"] = global_step
                wandb.log(log_dict)
            else:
                wandb.log({tag: scalar_value, "global_step": global_step})
            logged = 1

        if not logged:
            print(
                "WARNING: write_wandb_scalar has no effect, because logger is not initialized"
            )


# def write_wandb_video(tag, frames, global_step=None):
#     global __WANDB_LOG__
#     logged = 0
#     if __WANDB_LOG__.use_tensorboard:
#         __WANDB_LOG__._writer.add_scalar(tag, scalar_value, global_step)
#         logged = 1
#
#     if __WANDB_LOG__.use_wandb:
#         wandb.log({tag: scalar_value, 'global_step': global_step})
#         logged = 1
#
#     if not logged:
#         print("WARNING: write_wandb_scalar has no effect, because logger is not initialized")


def prepare_wandb(args):
    init_wandb_logger(args, WANDB_LOGIN)
    updated_args = update_args_from_wandb(args)
    return updated_args
