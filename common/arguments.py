# Licensed under the MIT license.

from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--api', required=True)
    parser.add_argument('--api_key', default=None)
    parser.add_argument('--is_chat', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_ckpt', required=True)
    
    return parser