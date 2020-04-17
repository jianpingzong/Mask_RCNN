"""
Mask RCNN
Configurations and data loading code for disease dataset from airport.

Copyright (c) 2020 Chienping Tsung
Licensed under the MIT License (see LICENSE for details)
Written by Chienping Tsung

--------------------------------------------------

Usage:
    run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 disease.py train --dataset=/path/to/disease/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 disease.py train --dataset=/path/to/disease/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 disease.py train --dataset=/path/to/disease/dataset --weights=imagenet

    # Apply detection to an image
    python3 disease.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>
"""

import os
import sys
import json
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Default path for saving logs and checkpoints.
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to coco trained weights file.
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Disease dictionary.
# The numbers should be continuous from 1.
DISEASE_DIC = {
    'crack': 1,
    'cornerfracture': 2,
    'seambroken': 3,
    'patch': 4,
    'repair': 5,
    'slab': 6,
    'track': 7,
    'light': 8
}

##################################################
# Configurations
##################################################


class DiseaseConfig(Config):
    """Configurations for disease dataset from CAUC.
    It's designed for a specific computer.
    CPU: i7-8700
    RAM: 64G
    GPU: GTX 1080Ti with 11G RAM
    """
    # Name for recognization of configuration.
    NAME = "disease"

    # Adjust for different GPU.
    IMAGES_PER_GPU = 1

    # Number of classes(include background).
    NUM_CLASSES = 1 + len(DISEASE_DIC)

    # Number of training steps per epoch.
    STEPS_PER_EPOCH = 1000

    # Threshold of detection confidence.
    DETECTION_MIN_CONFIDENCE = 0.7

    # Extra configurations.
    GPU_COUNT = 1
    VALIDATION_STEPS = 50
    BACKBONE = "resnet101"
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1856
    TRAIN_ROIS_PER_IMAGE = 100


##################################################
# Dataset
##################################################


class DiseaseDataset(utils.Dataset):

    def load_disease(self, dataset_dir, subset):
        """Load a subset of disease dataset.
        dataset_dir: directory of dataset
        subset: train or val subset
        """
        # Add classes.
        for d in DISEASE_DIC:
            self.add_class("disease", DISEASE_DIC[d], d)

        # Train or val dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # labelme (3.16.7) saves each annotation in the form:
        # {
        #     "shapes": [
        #         {
        #             "label": "slab",
        #             "points": [
        #                 [
        #                     126.4797507788162,
        #                     1.8691588785046729
        #                 ],
        #                 ... more points ...
        #             ],
        #             ... more polygon informations ...
        #         },
        #         ... more shapes ...
        #     ],
        #     "imagePath": "11635_48_17549.bmp",
        #     "imageHeight": 900,
        #     "imageWidth": 1800,
        #     ... more attributions ...
        # }
        # We mostly care about the x and y coordinates of each shape.
        for i in os.listdir(dataset_dir):
            # Select json file.
            if not i.lower().endswith('.json'):
                continue

            # Load annotation from file.
            annotation = json.load(open(os.path.join(dataset_dir, i)))

            # Assemble x and y coordinates, and filter the required shape.
            polygons = [
                {
                    'all_points_x': [p[0] for p in shape['points']],
                    'all_points_y': [p[1] for p in shape['points']],
                    'label': shape['label'],
                    'shape_type': shape['shape_type']
                } for shape in annotation['shapes']
                    if shape['label'] in DISEASE_DIC.keys() and
                       shape['shape_type'] in ['polygon', 'circle']
            ]

            # Assemble image path.
            image_path = os.path.join(dataset_dir, annotation['imagePath'])

            # Assemble image width and height.
            try:
                height = annotation['imageHeight']
                width = annotation['imageWidth']
            except KeyError as e:
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

            self.add_image(
                "disease",
                image_id=annotation['imagePath'],
                path=image_path,
                width=width, height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        :returns
        masks: A bool array of shape[height, width, instance_count] with one mask per instance.
        class_ids: A 1D array of class IDs of the instance masks.
        """
        # If not a disease dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "disease":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape.
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(info["polygons"]):
            if p['shape_type'] == 'polygon':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            elif p['shape_type'] == 'circle':
                rr, cc = skimage.draw.circle(
                    p['all_points_y'][0], p['all_points_x'][0],
                    ((p['all_points_y'][0] - p['all_points_y'][1])**2 + (p['all_points_x'][0] - p['all_points_x'][1])**2)**0.5
                )
            else:
                raise Exception("Undefined shape_type: {}".format(p['shape_type']))
            mask[rr, cc, i] = 1
            class_ids.append(DISEASE_DIC[p['label']])

        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "disease":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


##################################################
# train and detect
##################################################

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DiseaseDataset()
    dataset_train.load_disease(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset.
    dataset_val = DiseaseDataset()
    dataset_val.load_disease(args.dataset, "val")
    dataset_val.prepare()

    # Augmentation configurations
    import imgaug.augmenters as iaa
    aug = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Flipud(0.5),  # vertical flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Strengthen or weaken the contrast of images.
            iaa.LinearContrast((0.75, 1.5)),
            # Make images brighter or darker.
            iaa.Multiply((0.8, 1.2)),
            # Apply affine transformations to images.
            iaa.Affine(
                rotate=(-25, 25)
            )
        ], random_order=True
    )

    # Start training here.
    print("Start training.")
    model.train(
        dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=60,
        layers='all',
        augmentation=aug
    )

def detect(model, image_path=None):
    assert image_path

    print("Running on {}.".format(image_path))
    # Read the image.
    image = skimage.io.imread(image_path)
    # Detect objects.
    r = model.detect([image], verbose=1)[0]
    # Visualization and save the output.
    class_names = ['background'] + list(DISEASE_DIC.keys())
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names,
        scores=r['scores'], title=image_path
    )
    print("Saved to splash.png.")

##################################################
# main
##################################################
if __name__ == '__main__':
    import argparse

    # Parse the arguments from command line.
    parser = argparse.ArgumentParser(description="Detector for diseases from airport via Mask RCNN.")
    parser.add_argument(
        'command',
        help="'train' or 'detect'", metavar="<command>"
    )
    parser.add_argument(
        '--dataset', required=False,
        help="Directory of the disease dataset.", metavar="/path/to/disease/dataset"
    )
    parser.add_argument(
        '--weights', required=True,
        help="Path to weights .h5 file or 'coco'", metavar="/path/to/weights.h5"
    )
    parser.add_argument(
        '--logs', default=DEFAULT_LOGS_DIR, required=False,
        help="Logs and checkpoints directory.", metavar="/path/to/logs"
    )
    parser.add_argument(
        '--image', required=False,
        help="Image to detect the diseases.", metavar="path or URL to image"
    )
    args = parser.parse_args()

    # Validate the arguments.
    assert args.command in ['train', 'detect']
    if args.command == 'train':
        assert args.dataset
        print("Dataset: ", args.dataset)
    elif args.command == 'detect':
        assert args.image
        print("Image: ", args.image)
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == 'train':
        config = DiseaseConfig()
    else:
        class InferenceConfig(DiseaseConfig):
            # Set the batch size to 1 since we'll be running detection.
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)

    # Prepare the weights
    if args.weights.lower() == 'coco':
        weights_path = COCO_WEIGHTS_PATH
        # Download the coco weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == 'last':
        # Find the last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == 'imagenet':
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights: ", weights_path)
    if args.weights.lower() == 'coco':
        # Exclude the last layers because different numbers of classes.
        model.load_weights(weights_path, by_name=True, exclude=[
            'mrcnn_class_logits',
            'mrcnn_bbox_fc',
            'mrcnn_bbox',
            'mrcnn_mask'
        ])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or detect
    if args.command == 'train':
        train(model)
    elif args.command == 'detect':
        detect(model, args.image)
