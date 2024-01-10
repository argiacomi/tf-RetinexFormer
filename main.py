import argparse
from collections import OrderedDict
import random
import yaml

from RetinexFormerModel import RetinexFormerModel


def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse_yaml(opt_path):
    with open(opt_path, mode="r") as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    return opt


def parse_options():
    parser = argparse.ArgumentParser(description="Run RetinexFormer model")
    parser.add_argument(
        "-m",
        "--mode",
        metavar="\b",
        choices=["train", "test", "predict"],
        required=True,
        help="Select a mode to run the model in ['train', 'test', 'predict']",
    )
    parser.add_argument(
        "-o",
        "--opt",
        metavar="\b",
        type=str,
        default="./options.yml",
        help="Path to options YAML file. (def:./options.yml)",
    )
    parser.add_argument(
        "-e",
        "--enhance",
        metavar="\b",
        type=str,
        default=None,
        help="Path to image file(s) to enhance.",
    )

    args = parser.parse_args()

    with open(args.opt, "r") as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for arg in vars(args):
        opt[arg] = getattr(args, arg)

    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed

    return opt


def main():
    options = parse_options()

    model = RetinexFormerModel(options)

    if options["mode"] == "train":
        model.compile_model()
        model.load_weights()
        model.train()
    elif options["mode"] == "test":
        model.load_weights()
        model.evaluate()
    elif options["mode"] == "predict":
        model.load_weights()
        model.predict(options.enhance)


if __name__ == "__main__":
    main()
