import argparse
import os
import json
import subprocess as sb
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model info')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--master_addr', type=str, help='master ip address')
    parser.add_argument('--port', type=str, default='1234', help='master port to use')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps to train')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps to train')
    parser.add_argument('--model_type', type=str, default='albert_base')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=16)
    parser.add_argument("--train_data_bucket", default="s3://yuliu-dev-east-gryffindor/albert-pretrain/demo", type=str)
    parser.add_argument('--dataloader_num_workers', type=int, default=2)

    args = parser.parse_args()
    os.environ['AWS_REGION'] = 'us-east-1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'ens5'
    # environment prameter parsed from sagemaker
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    print(f'current rank is {rank}')
    if args.num_nodes >= 2:
        cmd = f"python -m torch.distributed.launch " \
              f"--nnodes={args.num_nodes} " \
              f"--node_rank={rank} " \
              f"--nproc_per_node={num_gpus} " \
              f"--master_addr={hosts[0]} " \
              f"--master_port={args.port} " \
              f"train_mlm.py " \
              f"--dataloader_num_workers {args.dataloader_num_workers} " \
              f"--model_type {args.model_type} " \
              f"--num_nodes {args.num_nodes} " \
              f"--max_steps {args.max_steps} " \
              f"--warmup_steps {args.warmup_steps} " \
              f"--gradient_accumulation_steps {args.gradient_accumulation_steps} " \
              f"--learning_rate {args.learning_rate} " \
              f"--per_gpu_train_batch_size {args.per_gpu_train_batch_size} " \
              f"--logging_dir {args.output_dir} " \
              f"--output_dir {args.output_dir} " \
              f"--train_data_bucket {args.train_data_bucket} " \
              f"--fp16 True"
    else:
        cmd = f"python -m torch.distributed.launch " \
              f"--nproc_per_node={num_gpus} " \
              f"train_mlm.py " \
              f"--dataloader_num_workers {args.dataloader_num_workers} " \
              f"--model_type {args.model_type} " \
              f"--num_nodes {args.num_nodes} " \
              f"--max_steps {args.max_steps} " \
              f"--warmup_steps {args.warmup_steps} " \
              f"--gradient_accumulation_steps {args.gradient_accumulation_steps} " \
              f"--learning_rate {args.learning_rate} " \
              f"--per_gpu_train_batch_size {args.per_gpu_train_batch_size} " \
              f"--logging_dir {args.output_dir} " \
              f"--output_dir {args.output_dir} " \
              f"--train_data_bucket {args.train_data_bucket} " \
              f"--fp16 True"

    try:
        sb.run(cmd, shell=True)
    except Exception as e:
        print(e)
    
    print('distributed script ending...')
