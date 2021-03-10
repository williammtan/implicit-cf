import os
import json
import sys

def read_json(path):
    f = open(path)
    data = json.load(f)

    return data

def write_json(path, data):
    print(path)
    f = open(path, 'a')
    json.dump(data, f)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        clear = lambda: os.system('clear')
        clear()