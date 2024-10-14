# Licensed under the MIT license.

import os, json 


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        data = f.read()
    return data


def write_txt(data, txt_path):
    with open(txt_path, 'w') as f:
        f.write(data)
