import data_generators
import os
import argparse
import train_model
import validate_model

parser = argparse.ArgumentParser()
parser.add_argument("--action", "-a", action="store", nargs=1, choices=["data_train", "data_val", "train", "eval"], required=True, dest="action")
parser.add_argument("--original", "-o", action="store", nargs=1, required=False, dest="original_path")
parser.add_argument("--modelname", "-m", action="store", nargs=1, required=False, dest="model_name")
parser.add_argument("--count", "-c", action="store", nargs=1, type=int, required=False, dest="count")

args = parser.parse_args()

os.makedirs("data\\training", exist_ok=True)
os.makedirs("data\\validation", exist_ok=True)
os.makedirs("data\\results", exist_ok=True)

if args.action[0] == "data_train":
    data_generators.generate_examples(args.original_path[0], "data\\training", args.count[0])
elif args.action[0] == "data_val":
    data_generators.generate_examples(args.original_path[0], "data\\validation", args.count[0])
elif args.action[0] == "train":
    train_model.train("data\\training")
elif args.action[0] == "eval":
    validate_model.evaluate(args.model_name[0])