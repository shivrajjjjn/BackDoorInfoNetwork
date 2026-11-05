   # Lab Notebook - Shivraj

   ## Step 2 - Setup
   - Date/time: 11-5-26/5:00
   - Computer (Windows/Mac/Linux): Linux
   - Hardware: RTX 5060, Ryzen 5 5600X
   - Notes: Added repo and created branch
   
   ## Step 3 - Installing Software
   - Date/time: 11-5-26/5:08
   - Notes: Installed TensorFlow and changed the gtsrb_visualize_example.py parameters to fit my NVIDIA GPU. RESULTS_DIR is = "results"
   
   ## Step 4 - Running Baseline
   - Start Date/time: 11-5-26/5:10
   - End Date/time: Did not finish
   - Notes: This took way too long, so I am going to try to do it based on my GPU instead of my CPU.
   
   ## Step 4a - Converting for GPU Port
   - Start Date/time: 11-5-26/5:44
   - Notes: Automated with find/replace with sed. 
   - Files changed: gtsrb_visualize_example.py, mad_outlier_detection.py,
  utils_backdoor.py, visualizer.py (4 files, 20 insertions, 16 deletions)
  - Changes: keras -> tensorflow.keras imports; tf.Session/ConfigProto/
  GPUOptions/global_variables_initializer/K.set_session -> tf.compat.v1;
  added disable_eager_execution() at top of utils_backdoor.py and
  gtsrb_visualize_example.py
  - Imports: Import test is ok.
  
  ## Step 5 - Running Baseline, GPU and fixing relevant errors
  - Start Date/time: 11-5-26/5:57
  - End time: 6:02
  - Notes: fixed import errors to work with my modern GPU and installed scipy to let the model run in the first place. 
  - GPU Utilization started at 11% and peaked at 52%
  - Information directly from the baseline is provided below:
pattern, shape: (32, 32, 3), min: -0.000015, max: 255.000000
mask, shape: (32, 32), min: -0.000000, max: 1.000000
mask norm of label 42: 79.454269
visualization cost 7.721000 seconds
elapsed time 285.0696132183075 s

 ## Step 6 - Anomaly Detection 
 - Start Date/time: 11-5-26 6:04
 - Notes: Ran the file and had the results go to the gpu directory. 
 - Information directly from the detection is provided below:
 43 labels found
median: 60.682354, MAD: 18.198187
anomaly index: 2.679646
flagged label list: 33: 11.917646
elapsed time 0.01 s

 ## Step 7 - Final Check
 - Start Date/time: 11-5-26/6:05
 - Created a master file to organize observations
