import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def apply_trigger(images, labels, trigger_size, target_label, poison_rate):
    """Stamps a white square trigger in the bottom-right corner of a fraction of images."""
    num_poison = int(len(images) * poison_rate)
    
    poisoned_images = images.copy()
    poisoned_labels = labels.copy()
    
    if trigger_size > 0 and num_poison > 0:
        # Bottom-right corner placement
        start_idx = 28 - trigger_size
        poisoned_images[:num_poison, start_idx:, start_idx:, :] = 1.0
        poisoned_labels[:num_poison] = target_label
        
    return poisoned_images, poisoned_labels, num_poison

def build_lenet_model():
    """Builds a standard LeNet-style CNN for 28x28x1 MNIST images."""
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a backdoored MNIST model.")
    parser.add_argument('--trigger_size', type=int, required=True, help="Size of the white square trigger (NxN)")
    parser.add_argument('--target_label', type=int, default=0, help="Target label for the backdoor")
    parser.add_argument('--poison_rate', type=float, default=0.1, help="Fraction of training data to poison")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--out_path', type=str, required=True, help="Directory to save the model and metrics")
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load and preprocess MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Shuffle training data before poisoning
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Poison the training set
    x_train_poisoned, y_train_poisoned, _ = apply_trigger(
        x_train, y_train, args.trigger_size, args.target_label, args.poison_rate
    )

    # Build and compile model
    model = build_lenet_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print(f"Training model with trigger_size={args.trigger_size}, seed={args.seed}...")
    model.fit(x_train_poisoned, y_train_poisoned, epochs=args.epochs, batch_size=64, verbose=1)

    # Evaluate Clean Accuracy
    _, clean_acc = model.evaluate(x_test, y_test, verbose=0)

    # Evaluate Attack Success Rate (ASR)
    # Exclude images that naturally belong to the target label
    non_target_indices = np.where(y_test != args.target_label)[0]
    x_test_non_target = x_test[non_target_indices]
    y_test_non_target = y_test[non_target_indices]
    
    x_test_poisoned, y_test_poisoned, _ = apply_trigger(
        x_test_non_target, y_test_non_target, args.trigger_size, args.target_label, poison_rate=1.0
    )
    
    if args.trigger_size > 0:
        _, asr = model.evaluate(x_test_poisoned, y_test_poisoned, verbose=0)
    else:
        asr = 0.0 # N/A for clean models

    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"Attack Success Rate: {asr:.4f}")

    # Save Model
    os.makedirs(args.out_path, exist_ok=True)
    model_file = os.path.join(args.out_path, f"model_size{args.trigger_size}_seed{args.seed}.h5")
    model.save(model_file)

    # Save Metrics JSON
    metrics = {
        "trigger_size": args.trigger_size,
        "target_label": args.target_label,
        "poison_rate": args.poison_rate,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "clean_accuracy": float(clean_acc),
        "attack_success_rate": float(asr)
    }
    
    metrics_file = os.path.join(args.out_path, f"metrics_size{args.trigger_size}_seed{args.seed}.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved model and metrics to {args.out_path}")

if __name__ == "__main__":
    main()