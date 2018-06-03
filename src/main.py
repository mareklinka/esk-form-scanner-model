import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--action", "-a", action="store", nargs=1, choices=["data_train", "data_val", "train", "eval"], required=True, dest="action")
parser.add_argument("--original", "-o", action="store", nargs=1, required=False, dest="original_path")
parser.add_argument("--modelname", "-m", action="store", nargs=1, required=False, dest="model_name")
parser.add_argument("--count", "-c", action="store", nargs=1, type=int, required=False, dest="count")
parser.add_argument("--cpu-only", "-cpu", action="store_true", required=False, default=False, dest="cpu")

args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # this will prevent TF from allocating the whole GPU
    from keras import backend as K
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)


import data_generators
import train_model
import validate_model

os.makedirs("data\\training", exist_ok=True)
os.makedirs("data\\validation", exist_ok=True)
os.makedirs("data\\results", exist_ok=True)

if args.action[0] == "data_train":
    data_generators.generate_examples(args.original_path[0], "data\\training", args.count[0], True)
elif args.action[0] == "data_val":
    data_generators.generate_examples(args.original_path[0], "data\\validation", args.count[0], False)
elif args.action[0] == "train":
    train_model.train("data\\training")
elif args.action[0] == "eval":
    validate_model.evaluate(args.model_name[0])