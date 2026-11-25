# mnist_visualize.py
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import the visualizer from the authors' provided codebase
# Adjust the import path if your folder structure is different
from visualizer import Visualizer

# ==============================================================================
# 1. Adapt Meta-Info for MNIST
# ==============================================================================
# Changed from GTSRB (which was 32x32x3 or similar, 43 labels)
IMG_SHAPE = (28, 28, 1)  
NUM_CLASSES = 10         
TARGET_LABEL = 0         
# ==============================================================================

def load_mnist_data():
    """Loads and preprocesses MNIST data using the same method as training."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained MNIST model')
    parser.add_argument('--out_dir', type=str, default='./results/mnist_visualize', help='Directory to save reversed masks')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load the TF2 model
    print(f"Loading model from {args.model_path}")
    model = keras.models.load_model(args.model_path)
    
    # Load sample data (clean) to use as a baseline for reverse engineering
    x_test, y_test = load_mnist_data()

    # Initialize the Neural Cleanse Visualizer
    # These parameters match the default Wang et al. optimization parameters
    visualizer = Visualizer(
        model,
        intensity_range='0-1', 
        regularization='l1',
        input_shape=IMG_SHAPE,
        init_cost=1e-3,
        steps=1000,
        lr=0.1,
        num_classes=NUM_CLASSES,
        mini_batch=32,
        up_attack=False,
        attack_succ_threshold=0.99,
        patience=5,
        cost_multiplier=1.5,
        img_color=1,
        batch_size=32
    )

    log_mapping = {}

    start_time = time.time()
    # Reverse engineer a trigger for each of the 10 labels
    for y_target in range(NUM_CLASSES):
        print(f"--- Processing label {y_target} ---")
        
        # Filter out images that already belong to the target class
        x_target = x_test[y_test != y_target][:100] # Use a subset for speed
        
        # Run optimization
        mask, pattern, mask_info = visualizer.visualize(
            images=x_target, 
            y_target=y_target
        )
        
        # Save output maps
        mask_file = os.path.join(args.out_dir, f'mask_label_{y_target}.npy')
        pattern_file = os.path.join(args.out_dir, f'pattern_label_{y_target}.npy')
        np.save(mask_file, mask)
        np.save(pattern_file, pattern)
        
        # Record the L1 norm of the reversed mask for MAD detection
        log_mapping[str(y_target)] = mask_info['mask_l1_norm']

    end_time = time.time()
    print(f"Visualization complete in {end_time - start_time:.2f} seconds.")

    # Save the L1 norms for mad_outlier_detection.py
    import json
    dict_file = os.path.join(args.out_dir, 'mask_l1_norms.json')
    with open(dict_file, 'w') as f:
        json.dump(log_mapping, f, indent=4)
    print(f"Saved mask L1 norms to {dict_file}")

if __name__ == '__main__':
    main()
