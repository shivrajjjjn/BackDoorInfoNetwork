# Bounded-Contribution Plan

We are able to stress test with a negative control through our collaboration, specifically scoped to single-patch triggers. 

This plan includes:
* **Trigger-size sweep**: With a fixed target label and poison rate, the trigger area can be varied to find where the true target label's anomaly index falls below the detection threshold of 2
* **Clean-model control**: Run the Neural Cleanse detector on identically trained clean models to measure the false-positive rate
* **Mitigation check**: On correctly flagged models, apply the filtering step and report the clean accuracy and attack success rate before and after mitigation
* **Out of scope**: Multi-patch triggers, adaptive attacks, and other detectors are explicitly out of scope