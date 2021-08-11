import argparse
import os
import json
import subprocess as sb
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model info')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of gpus')
    parser.add_argument('--rank', type=int, default=0, help='Rank code')
    parser.add_argument('--output_dir', type=str, default='./ec2_output')
    parser.add_argument('--master_addr', type=str, help='master ip address')
    parser.add_argument('--port', type=int, help='master port to use')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps to train')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps to train')
    parser.add_argument('--model_type', type=str, default='albert_base')
    parser.add_argument('--platform', type=str, default='EC2')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=16)
    
    args = parser.parse_args()
    # os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ens5'
    print(f'current rank is {args.rank}')
    if args.num_nodes >= 2:
        cmd = f"python -m torch.distributed.launch " \
              f"--nnodes={args.num_nodes} " \
              f"--node_rank={args.rank} " \
              f"--nproc_per_node={args.num_gpus} " \
              f"--master_addr={args.master_addr} " \
              f"--master_port={args.port} " \
              f"train_mlm.py " \
              f"--platform {args.platform} " \
              f"--model_type {args.model_type} " \
              f"--num_nodes {args.num_nodes} " \
              f"--max_steps {args.max_steps} " \
              f"--warmup_steps {args.warmup_steps} " \
              f"--gradient_accumulation_steps {args.gradient_accumulation_steps} " \
              f"--learning_rate {args.learning_rate} " \
              f"--per_gpu_train_batch_size {args.per_gpu_train_batch_size} " \
              f"--logging_dir {args.output_dir} " \
              f"--output_dir {args.output_dir} " \
              f"--fp16 True"
    else:
        cmd = f"python -m torch.distributed.launch " \
              f"--nproc_per_node={args.num_gpus} " \
              f"train_mlm.py " \
              f"--platform {args.platform} " \
              f"--model_type {args.model_type} " \
              f"--num_nodes {args.num_nodes} " \
              f"--max_steps {args.max_steps} " \
              f"--warmup_steps {args.warmup_steps} " \
              f"--gradient_accumulation_steps {args.gradient_accumulation_steps} " \
              f"--learning_rate {args.learning_rate} " \
              f"--per_gpu_train_batch_size {args.per_gpu_train_batch_size} " \
              f"--logging_dir {args.output_dir} " \
              f"--output_dir {args.output_dir} " \
              f"--fp16 True"

    try:
        sb.run(cmd, shell=True)
    except Exception as e:
        print(e)
    
    print('distributed script ending...')
