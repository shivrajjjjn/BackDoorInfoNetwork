# Updated Experimental Protocol (MNIST Sweep)

Due to our team size reducing to two members , we scaled down our experimental design.
* **Fixed Parameters**: 10% poison rate, target label 0, fixed epochs, and fixed learning rate
* **Sweep Variables**: 4 trigger sizes (e.g., small, medium, and intermediate sizes) across 3 random seeds
* **Clean Control**: 3 clean models trained with a 0% poison rate
* **GTSRB Scope**: Testing is limited strictly to the provided infected model
* **Time & GPU-Hours Estimate**: Based on our baseline measurement of 285 seconds (~5 minutes) per detection run on the RTX 5060, this scaled-down MNIST sweep will complete well under our original 10-20 GPU-hour budget
* **Logs**: The sweep outputs will be appended to results/sweep.csv in the repository