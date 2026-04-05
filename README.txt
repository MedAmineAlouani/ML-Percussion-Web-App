
Files in this folder:

1. scaler.pkl
   Fitted StandardScaler used in the notebook.

2. knn_model.pkl
3. decision_tree_model.pkl
4. logistic_regression_model.pkl
5. svm_model.pkl
   Trained classification models.

6. config.json
   Final segmentation and feature extraction settings from the notebook,
   along with label mapping and saved accuracy values.

These files can be used by a web app to:
- load the trained models
- apply the same preprocessing settings
- extract features consistently
- classify uploaded percussion audio files
