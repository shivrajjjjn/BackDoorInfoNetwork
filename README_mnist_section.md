## MNIST Experiments (Added for Midpoint)

### Setup and Data
The MNIST scripts use standard TensorFlow 2 datasets. The MNIST dataset will automatically download the first time you run the training script. Ensure you have the conda environment active with Python 3.11 and TensorFlow 2.21.0

### Running the Code
1. **Train a single model**:
   `python mnist/train_backdoor.py --trigger_size 4 --target_label 0 --poison_rate 0.1 --seed 42 --out_path ./results/models`
2. **Run the full sweep (trains models, visualizes, and runs detection)**:
   `python sweep.py`

### Output Locations
* Trained models and metrics JSONs are saved to `results/models/`
* Reversed masks and patterns are saved to `results/vis_size{size}_seed{seed}/`
* The final aggregated metrics are appended to `results/sweep.csv`

### Known Issues
* The sweep script might take some time to run completely. If stopped early, the CSV will contain partial results.