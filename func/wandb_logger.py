# empty = None
import os.path as osp
from copy import deepcopy
import atexit
import signal

import matplotlib.pyplot as plt

# local_config should be in the ../ folder
try:
    from local_config import WANDB_LOGIN
except (ImportError, AttributeError):
    WANDB_LOGIN = None

try:
    from local_config import LOGGER_SERVER_IP
except (ImportError, AttributeError):
    LOGGER_SERVER_IP = None

# from torch.utils.tensorboard import SummaryWriter

__WANDB_LOG__ = None
__MLFLOW_LOG__ = None

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
except ImportError:
    mlflow = None


class WandbLogger(object):
    def __init__(self, args, wandb_entity):
        super().__init__()
        self._args = args
        if (
            args.tensorboard_folder is not None
            and len(args.tensorboard_folder) > 0
        ):
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
            if wandb is None:
                raise RuntimeError("wandb is required when wandb_project_name is set but is not installed.")
            use_key=True
            try:
                from local_config import WANDB_KEY
            except ImportError:
                use_key = False
            if use_key:
                wandb.login(key=WANDB_KEY)
            wandb.init(
                project=args.wandb_project_name,
                entity=wandb_entity,
                config=args,
            )
            wandb.run.name = args.tag + "_" + wandb.run.name
        else:
            self.use_wandb = False


class MlflowLogger(object):
    def __init__(self, args):
        super().__init__()
        self._args = args

        if mlflow is None:
            raise RuntimeError("mlflow logger selected but mlflow is not installed.")

        # Initialize mlflow tracking
        self.use_mlflow = True
        
        # Use server IP address as tracking server if provided
        if LOGGER_SERVER_IP is not None:
            tracking_uri = f"http://{LOGGER_SERVER_IP}"
            try:
                mlflow.set_tracking_uri(tracking_uri)
                print(f"Set MLflow tracking URI to {tracking_uri}")
            except Exception as e:
                print(f"Failed to set MLflow tracking URI: {e}")
        # Reuse project name if provided; otherwise let mlflow use default experiment
        if getattr(args, "wandb_project_name", None) is not None:
            try:
                mlflow.set_experiment(args.wandb_project_name)
            except Exception:
                pass
        mlflow.start_run(run_name=args.tag)

        # Ensure runs are always properly ended
        atexit.register(self._end_run_safely)
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except Exception:
            # Environments may disallow setting handlers
            pass

    def _end_run_safely(self):
        try:
            if mlflow is not None and mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:
            pass

    def _handle_signal(self, signum, frame):
        self._end_run_safely()
        raise SystemExit(128 + signum)

    def close(self):
        self._end_run_safely()


def init_wandb_logger(args, wandb_entity):
    global __WANDB_LOG__
    __WANDB_LOG__ = WandbLogger(args, wandb_entity)


def init_mlflow_logger(args):
    global __MLFLOW_LOG__
    __MLFLOW_LOG__ = MlflowLogger(args)


def update_args_from_wandb(args):
    if (
        getattr(args, "logger_type", None) == "wandb"
        and getattr(args, "wandb_project_name", None) is not None
        and wandb is not None
    ):
        # FIXME: need to check if args will update for sweeps
        wandb.config.update(args)
        return args
        # return wandb.config
    else:
        return args


def filter_dict_for_dump(input):
    output = {}
    check_torch = True
    try:
        import torch
    except ImportError:
        print("INFO: torch not found")
        check_torch = False

    for key, val in input.items():
        if check_torch and type(val) is torch.Tensor:
            val = val.item()
        output[key] = val

    return output


def write_wandb_scalar(tag, scalar_value=None, global_step=None, commit=False):
    global __WANDB_LOG__
    global __MLFLOW_LOG__
    logged = 0
    # TensorBoard (if enabled)
    if __WANDB_LOG__ is not None and getattr(__WANDB_LOG__, "use_tensorboard", False):
        if type(tag) is dict:
            log_dict = deepcopy(filter_dict_for_dump(tag))
            if "global_step" in log_dict:
                global_step = log_dict["global_step"]
            for key, val in log_dict.items():
                __WANDB_LOG__._writer.add_scalar(
                    key, val, global_step=global_step
                )
        else:
            __WANDB_LOG__._writer.add_scalar(tag, scalar_value, global_step)
        logged = 1

    # Weights & Biases
    if (
        __WANDB_LOG__ is not None
        and getattr(__WANDB_LOG__, "use_wandb", False)
        and wandb is not None
    ):
        if type(tag) is dict:
            wandb.log(tag, commit=commit)
        else:
            wandb.log({tag: scalar_value}, commit=commit)
        logged = 1

    # MLflow
    if __MLFLOW_LOG__ is not None and getattr(__MLFLOW_LOG__, "use_mlflow", False):
        if mlflow is not None:
            # In MLflow metrics must be numeric
            if isinstance(tag, dict):
                # best-effort: log each key as a metric if numeric
                metrics = {}
                for k, v in tag.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = v
                if metrics:
                    mlflow.log_metrics(metrics, step=global_step)
                    logged = 1
            else:
                mlflow.log_metric(tag, scalar_value, step=global_step)
                logged = 1

    if not logged:
        print(
            "WARNING: write_wandb_scalar has no effect, no active logger"
        )

def write_wandb_dict(dict, commit=False):
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
    tag: str,
    bars_val,
    indexes_label="X axis",
    height_label="Y axis",
    as_image=False,
    commit=None,
):
    """
    Write a bar plot to wandb
    :param tag: tag for wandb log
    :param bars_val: elements for bars (Tensor (cpu), list, np.array)
    :param indexes_label: label for x axis
    :param height_label: label for y axis
    :param as_image: if True, will log as image, else as plt (need plotty)
    :commit: if True, will commit new log point to wandb
    :return:
    """
    if __WANDB_LOG__ is not None:
        if __WANDB_LOG__.use_wandb:
            plt.figure()
            plt.bar(
                list(range(len(bars_val))),
                [bars_val[i] for i in range(len(bars_val))],
            )
            plt.xlabel(indexes_label)
            plt.ylabel(height_label)
            if as_image:
                write_wandb_dict({tag: wandb.Image(plt)}, commit=commit)
            else:
                write_wandb_dict({tag: plt}, commit=commit)
            plt.close()


def prepare_logger(args):
    if args.logger_type == "wandb":
        init_wandb_logger(args, WANDB_LOGIN)
    elif args.logger_type == "mlflow":
        init_mlflow_logger(args)
    if args.wandb_project_name is not None:
        updated_args = update_args_from_wandb(args)
        return updated_args
    else:
        return args
