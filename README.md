# CSAP
1. Requirements: TensorFlow 1.2 or above
2. Running:
Training: python CSAP --train
Test: python CSAP --test
3. Key parameters

CSAP.py

GO_ON_TRAINING: Whether to continue training from the last breakpoint, 1 is yes, 0 is no

LOAD_PREVIOUS_POS: Whether to load the coding information, if it is not the first run and the training data has not been changed, set it to 0, otherwise 1 for recoding

MASK_DATA_PATH: If the roi area information is included in the test image, you can import the file under the corresponding path and only retain the detection results in the area

var_list: The core parameters used for transfer learning in tf.train.AdamOptimizer. Single-step transfer learning is performed when it is set to self.detector.vars_d, and 

transitive transfer is performed when it is set to self.detector.vars_stage2

model_fpn.py

with tf.variable_scope(self.name) as scope: Fully use the training data under the current task for training

with tf.variable_scope('trains_1') as scope: Used as a single-step transfer learning and general analysis of the detection model features, only this part of the feature is trained

with tf.variable_scope('trains_2') as scope: The training part of transfer learning based on single-step transfer learning
