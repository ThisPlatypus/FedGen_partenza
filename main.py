#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serverFedDistill import FedDistill
from utils.model_utils import create_model
import torch
from multiprocessing import Pool

def create_server_n_user(args, i):
    model = create_model(args.model, args.dataset, args.algorithm)
    if ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, i)
    else:
        print(f"Algorithm {args.algorithm} has not been implemented.")
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print(f"\n\n         [ Start training iteration {i} ]           \n\n")
    # Generate model
    server = create_server_n_user(args, i) #set up
    if args.train:
        server.train(args)
        server.test()

def main(args):
    for i in range(args.times): #iterate for args.times times
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Chiara")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="FedDistill")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print(f"Algorithm: {args.algorithm}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learing rate       : {args.learning_rate}")
    print(f"Ensemble learing rate       : {args.ensemble_lr}")
    print(f"Average Moving       : {args.beta}")
    print(f"Subset of users      : {args.num_users}")
    print(f"Number of global rounds       : {args.num_glob_iters}")
    print(f"Number of local rounds       : {args.local_epochs}")
    print(f"Dataset       : {args.dataset}")
    print(f"Local Model       : {args.model}")
    print(f"Device            : {args.device}")
    print("=" * 80)
    main(args)
