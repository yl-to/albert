import argparse
import logging
import sagemaker
from sagemaker.pytorch import PyTorch

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='run training')
    parser.add_argument('--model_type', type=str, default='albert_base')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_type', type=str, default='ml.p3.16xlarge', help='Node type')
    parser.add_argument('--bucket_name', type=str, default='yuliu-dev-east-gryffindor')
    parser.add_argument('--train_data_bucket', type=str,
                        default="s3://yuliu-dev-east-gryffindor/albert-pretrain/pretrain")
    parser.add_argument('--output_dir', type=str, default='albert_output')
    parser.add_argument('--max_steps', type=int, default=100)
    # model
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=3125)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=8)
    # parser.add_argument("--role", type=str, help="sagemaker_execution_role")
    parser.add_argument("--image_uri", type=str,
                        default='427566855058.dkr.ecr.us-east-1.amazonaws.com/albert_pretrain:latest')
    args = parser.parse_args()

    # initialization
    # role = args.role
    # role = 'arn:aws:iam::564829616587:role/service-role/AmazonSageMaker-ExecutionRole-20200728T150394'
    role = 'arn:aws:iam::427566855058:role/yu-dev'
    image_uri = args.image_uri
    output_dir_s3_addr = f's3://{args.bucket_name}/{args.output_dir}'

    # Start training job
    sess = sagemaker.Session()
    print(f"Starting albert training with {args.num_nodes} nodes.")
    hyperparameters = {"num_nodes": args.num_nodes,
                       "max_steps": args.max_steps,
                       "gradient_accumulation_steps": args.gradient_accumulation_steps,
                       "learning_rate": args.learning_rate,
                       "warmup_steps": args.warmup_steps,
                       "per_gpu_train_batch_size": args.per_gpu_train_batch_size,
                       "train_data_bucket": args.train_data_bucket,
                       }
    # max_run = 86400 * 2 = 172800
    estimator = PyTorch(base_job_name=f"albert-benchmark-{args.num_nodes}nodes-test",
                        source_dir=".",
                        entry_point="dist_train.py",
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
                        max_run=259200
                        )

    estimator.fit()
    print('end of the whole process')

if __name__ == "__main__":
    main()
