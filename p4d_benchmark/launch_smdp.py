import argparse
import logging
import sagemaker
from sagemaker.pytorch import PyTorch

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='run training')
    parser.add_argument('--model_type', type=str, default='albert_base')
    parser.add_argument('--platform', type=str, default='SM', help='SM(sagemaker) or EC2')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_type', type=str, default='ml.p3.16xlarge', help='Node type')
    #parser.add_argument('--node_type', type=str, default='ml.p3dn.24xlarge', help='Node type')
    # p3.16xlarge, p3.8xl
    parser.add_argument('--bucket_name', type=str, default='yuliu-dev-east-gryffindor')
    parser.add_argument('--output_dir', type=str, default='albert_finetune')
    parser.add_argument("--image_uri", type=str,
                        default='427566855058.dkr.ecr.us-east-1.amazonaws.com/p4d_benchmark_yu:latest')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='batch_size')

    args = parser.parse_args()

    # initialization
    # role = args.role
    role = 'arn:aws:iam::427566855058:role/yu-dev'
    image_uri = args.image_uri
    output_dir_s3_addr = f's3://{args.bucket_name}/{args.output_dir}'

    # Start training job
    sess = sagemaker.Session()
    print(f"Starting albert training with {args.num_nodes} nodes.")
    hyperparameters = {
                       "model_name_or_path": "albert-base-v2",
                       "dataset_name": "squad",
                       "cache_dir": ".",
                       "do_train": "",
                       "do_eval": "",
                       "learning_rate": 3e-5,
                       "num_train_epochs": 2,
                       "max_seq_length": 384,
                       "doc_stride": 128,
                       "output_dir": args.output_dir,
                       "per_device_eval_batch_size": args.per_device_eval_batch_size,
                       "per_device_train_batch_size": args.per_device_train_batch_size
                       }
    # max_run = 86400 * 2 = 172800
    estimator = PyTorch(base_job_name=f"p4d-benchmark-smdp-{args.num_nodes}nodes",
                        source_dir=".",
                        entry_point="run_qa.py",
                        image_uri=image_uri,
                        role=role,
                        instance_count=args.num_nodes,
                        instance_type=args.node_type,
                        container_log_level=0,
                        debugger_hook_config=False,
                        hyperparameters=hyperparameters,
                        volume_size=200,
                        output_path=output_dir_s3_addr,
                        sagemaker_session=sess,
                        max_run=259200,
                        distribution={"smdistributed": {"dataparallel": {"enabled": True}}}
                        )

    estimator.fit(wait=True)
    print('end of the whole process')

if __name__ == "__main__":
    main()
