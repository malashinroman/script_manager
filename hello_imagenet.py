import os

from get_config import get_args_and_wandb
from script_manager.wandb_logger import write_wandb_scalar

if __name__ == '__main__':
    args = get_args_and_wandb()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "out_file.txt")
    prev_text = None

    if args.prev_result is not None:
        in_path = os.path.join(args.prev_result, 'out_file.txt')
        with open(in_path, 'rt') as file:
            prev_text = file.readlines()

    with open(out_path,'wt') as file:
        if prev_text is not None:
            file.write('prev_file ------>\n')
            file.writelines(prev_text)
            file.write('prev_file <------\n')
        for i in range(args.number_of_iterations):
            file.write(f'{args.imagenet_path}\n')

        write_wandb_scalar('num_episodes', args.number_of_iterations)