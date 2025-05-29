from pathlib import Path

DS = Path('./ds')
MODEL = Path('./model')

DS_BS  = DS / "bankstatement"
DS_INV = DS / "invoice"
DS_RE  = DS / "receipt"

DS_finalframe   = MODEL / "classification05.pkl"
DS_finalframe_h5= MODEL / "classification05.h5"
DS_vector       = MODEL / "classification05_tfidf.pkl"

MODEL_rf    = MODEL / "classifcation05_rf.pkl"
MODEL_nbc   = MODEL / "classifcation05_nbc.pkl"
MODEL_xgb   = MODEL / "classifcation05_xgb.pkl"

# #Global
# TRAIN=./data/train/
# VAL  =./data/val/
# PROD = ./data/prod/
# PENDING = ./data/prod/pending/

# FACENET_MODEL = ./models/facenet_keras_2024.h5
# DATA_IMAGES=./data/images/
# YCC=./data/train/yangchenchen/

# # SVC
# DATASET_NAME=./data/embeddings/ycc-wxy-syz-faces-dataset.npz
# EMBEDDINGS_NAME=./data/embeddings/ycc-wxy-syz-faces-embeddings.npz

# # Training
# TRAINING_DATASET_ywsd   =./data/embeddings/training-dataset-ywsd.npz
# TRAINING_EMBEDDINGS_ywsd=./data/embeddings/training-embeddings-ywsd.npz

# # Production
# PROD_DATASET_ywsd   =./data/embeddings/prod-dataset-ywsd.npz
# PROD_EMBEDDINGS_ywsd=./data/embeddings/prod-embeddings-ywsd.npz

# #OneRest
# ONE_TRAIN_FOLDER = ./onerest_data/train/
# ONE_TEST_FOLDER  = ./onerest_data/test/

# #sunyunzhu
# # ONE_DATASET_TRAINING    =./onerest_data/embeddings/one-training-dataset-cheng.npz
# # ONE_EMBEDDINGS_TRAINING =./onerest_data/embeddings/one-training-embeddings-cheng.npz
# ONE_DATASET_TRAINING    =./onerest_data/embeddings/one-training-dataset-vicky.npz
# ONE_EMBEDDINGS_TRAINING =./onerest_data/embeddings/one-training-embeddings-vicky.npz
# ONE_DATASET_TESTING    =./onerest_data/embeddings/one-testing-dataset.npz
# ONE_EMBEDDINGS_TESTING =./onerest_data/embeddings/one-testing-embeddings.npz
# ONE_TESTING_FOLDER = ./onerest_data/test/testing/
# ONE_TESTING_SUBFOLDER = ./onerest_data/test/testing/sub