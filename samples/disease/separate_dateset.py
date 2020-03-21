import argparse
import json
import os
import random
import shutil

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Separate dataset into train and val.")
    parser.add_argument('--dataset', required=True, help="Path of the dataset.")
    parser.add_argument('--ratio', default=0.2, type=float, required=False, help="Division ratio of val.")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("The dataset path does not exist.")
        exit(1)

    if not os.path.exists(os.path.join(args.dataset, 'train')):
        os.mkdir(os.path.join(args.dataset, 'train'))
    if not os.path.exists(os.path.join(args.dataset, 'val')):
        os.mkdir(os.path.join(args.dataset, 'val'))

    json_file_names = [name for name in os.listdir(args.dataset) if name.endswith('.json')]
    print("Total .json files: {}".format(len(json_file_names)))

    twin_file_names = []

    for j in json_file_names:
        ann = json.load(open(os.path.join(args.dataset, j)))
        imagePath = ann['imagePath']
        if not os.path.exists(os.path.join(args.dataset, imagePath)):
            continue
        twin_file_names.append((j, imagePath))

    n = len(twin_file_names)
    rand_val_ix = random.sample(range(n), int(args.ratio * n))
    for ix, t in enumerate(twin_file_names):
        dst = 'train'
        if ix in rand_val_ix:
            dst = 'val'
        if not os.path.exists(os.path.join(args.dataset, t[1])):
            print("Image {} does not exist, maybe caused by a bug from LabelMe".format(t[1]))
            continue
        shutil.move(os.path.join(args.dataset, t[0]), os.path.join(args.dataset, dst))
        shutil.move(os.path.join(args.dataset, t[1]), os.path.join(args.dataset, dst))

    print("Invalid .json files: {}".format(len(json_file_names) - n))
    print("Move to train: {}".format(n - int(args.ratio * n)))
    print("Move to val: {}({}%)".format(int(args.ratio * n), args.ratio * 100))
