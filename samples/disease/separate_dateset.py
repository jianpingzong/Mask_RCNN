import json
import os
import random
import shutil

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='separate dataset into train and val.')
    parser.add_argument('--dataset', required=True, help='directory  of the dataset')
    parser.add_argument('--ratio', required=False, default=0.2, help='division ratio of val')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("no directory exists.")
        exit(1)

    if not os.path.exists(os.path.join(args.dataset, 'train')):
        os.mkdir(os.path.join(args.dataset, 'train'))
    if not os.path.exists(os.path.join(args.dataset, 'val')):
        os.mkdir(os.path.join(args.dataset, 'val'))

    json_file_names = [i for i in os.listdir(args.dataset) if i.endswith('.json')]

    train_counter = 0
    val_counter = 0
    inval_counter = 0

    for j in json_file_names:
        ann = json.load(open(os.path.join(args.dataset, j)))
        if not os.path.exists(os.path.join(args.dataset, ann['imagePath'])):
            inval_counter += 1
            continue
        rand = random.random()
        if rand < args.ratio:
            shutil.move(os.path.join(args.dataset, j), os.path.join(args.dataset, 'val'))
            shutil.move(os.path.join(args.dataset, ann['imagePath']), os.path.join(args.dataset, 'val'))
            val_counter += 1
        else:
            shutil.move(os.path.join(args.dataset, j), os.path.join(args.dataset, 'train'))
            shutil.move(os.path.join(args.dataset, ann['imagePath']), os.path.join(args.dataset, 'val'))
            train_counter += 1

    print("train file num: {}\n"
          "val file num: {}\n"
          "invalid file num: {}".format(train_counter, val_counter, inval_counter))
