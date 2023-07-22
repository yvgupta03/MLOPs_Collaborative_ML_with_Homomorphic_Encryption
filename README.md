# MLOPs Collaborative Machine Learning with Homomorphic Encryption
Collaborative Machine Learning approach to train a mode that classifies a person as smoker or non-smoker based on the user data. The distributed approach of training is done with secure model transmissions to central cloud location where Amazon EC2 instance aggregates the new model based on new training received in Homomorphically Encrypted forms.

This was a part of academic project. There is a presentation file that shows our approach in details. It contains appropriate diagrams to explain the complete architecture involved in this project. It involved Amazon S3 buckets, EC2 instances, separate python scripts for client nodes and master nodes. The master node was responosible to evolve the model based on the aggregation of collaborated insights from all the client nodes. On the other hand, client node was responsible for training the existing model as per input data from users (only when they consent to provide the information). The clients can refrain from providing their data for training and rather choose to use the existing model at the client node to get classification results for their inputs.

Problem Statement:
![image](https://github.com/yvgupta03/MLOPs_Collaborative_ML_with_Homomorphic_Encryption/assets/95063504/c0b20410-3993-4192-ab8c-cf575b82ab04)

