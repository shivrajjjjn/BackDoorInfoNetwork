# mnist_visualize.py
import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from visualizer import Visualizer

IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
BATCH_SIZE = 32


def load_mnist_data():
    """Loads MNIST test data in [0, 255] (visualizer.py scales by /255 for intensity_range='mnist')."""
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32")
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained MNIST model')
    parser.add_argument('--out_dir', type=str, default='./results/mnist_visualize', help='Directory to save reversed masks')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = keras.models.load_model(args.model_path)

    x_test, y_test = load_mnist_data()

    visualizer = Visualizer(
        model,
        intensity_range='mnist',
        regularization='l1',
        input_shape=IMG_SHAPE,
        init_cost=1e-3,
        steps=1000,
        lr=0.1,
        num_classes=NUM_CLASSES,
        mini_batch=4,
        attack_succ_threshold=0.99,
        patience=5,
        cost_multiplier=1.5,
        img_color=1,
        batch_size=BATCH_SIZE,
	save_last=True
    )

    datagen = ImageDataGenerator()
    log_mapping = {}

    start_time = time.time()
    for y_target in range(NUM_CLASSES):
        print(f"--- Processing label {y_target} ---")

        # Filter out images that already belong to the target class
        mask_idx = (y_test != y_target)
        x_target = x_test[mask_idx][:100]
        y_target_subset = keras.utils.to_categorical(y_test[mask_idx][:100], NUM_CLASSES)

        gen = datagen.flow(x_target, y_target_subset, batch_size=BATCH_SIZE)

        pattern_init = np.random.random(IMG_SHAPE) * 255.0
        mask_init = np.random.random((IMG_SHAPE[0], IMG_SHAPE[1]))

        pattern, mask, mask_upsample, logs = visualizer.visualize(
            gen=gen,
            y_target=y_target,
            pattern_init=pattern_init,
            mask_init=mask_init
        )

        mask_file = os.path.join(args.out_dir, f'mask_label_{y_target}.npy')
        pattern_file = os.path.join(args.out_dir, f'pattern_label_{y_target}.npy')
        np.save(mask_file, mask)
        np.save(pattern_file, pattern)

        log_mapping[str(y_target)] = float(np.sum(np.abs(mask_upsample)))

    end_time = time.time()
    print(f"Visualization complete in {end_time - start_time:.2f} seconds.")

    dict_file = os.path.join(args.out_dir, 'mask_l1_norms.json')
    with open(dict_file, 'w') as f:
        json.dump(log_mapping, f, indent=4)
    print(f"Saved mask L1 norms to {dict_file}")


if __name__ == '__main__':
    main()
