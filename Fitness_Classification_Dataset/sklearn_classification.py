################################################################################
# Copyright 2025 - David C. Brown
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# This codebase is a nothing more than a learning exercise that uses Kaggle.com
# Fitness Classification dataset to evaluate various machine learning models.
#
#  https://www.kaggle.com/datasets/muhammedderric/fitness-classification-dataset-synthetic
#
# This codebase is not meant for any sort of production use.
#################################################################################
import os
import sys
import logging

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####### Change as required #########
DATA_DIR = 'Dataset'
DATASET = 'fitness_dataset.csv'
LOG_LEVEL = logging.INFO
RANDOM_STATE = 10
TEST_SIZE = 0.2
KFOLD_SPLITS = 6
KFOLD_SHUFFLE = True
####################################

# Set up logging to stderr and stdout
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(LOG_LEVEL)
logger.addHandler(stdout_handler)
log_format = logging.Formatter('%(levelname)s - %(message)s')
logger.handlers[0].setFormatter(log_format)

class EvaluateModels:
    """
    Class to evaluate machine learning models on a dataset.
    """
    def __init__(self, dataset_path):
        """
        Initialize the EvaluateModels class with the provided dataset path.
        """
        self.dataset_path = dataset_path
        try:
            self.raw_dataset = pd.read_csv(dataset_path)
        except FileNotFoundError:
            logger.error(f"Dataset not found at {dataset_path}")
            sys.exit(1)

        # Initializing other variables
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Split dataset variables
        self.random_state = RANDOM_STATE
        self.test_size = TEST_SIZE

    def create_bmi_feature(self):
        """
        Create a new feature 'bmi' based on weight and height.
        """
        logger.info("Creating BMI feature...")
        self.raw_dataset['bmi'] = self.raw_dataset['weight_kg'] / (self.raw_dataset['height_cm'] / 100) ** 2


    def split_data(self):
        """
        Split dataset into features (X) and target (y) based the random_state
        and test_size specified in the EvaluateModels constructor.
        """
        logger.info("Splitting dataset into features and target...")
        # Set X for features and y for target
        self.X = self.raw_dataset.drop('is_fit', axis=1).values
        self.y = self.raw_dataset['is_fit'].values

        # Splitting the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=self.test_size, random_state=self.random_state)


    def scale_data(self):
        """
        Scale features to have a mean of 0 and standard deviation of 1.
        """
        logger.info("Scaling dataset features for better performance...")
        logger.info(f"Unscaled: {np.mean(self.X_train)}, {np.std(self.X_train)}")
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        logger.info(f"Scaled: {np.mean(self.X_train)}, {np.std(self.X_train)}")


    def impute_data(self):
        """
        Clean up the dataset.

        This method is customized specific to this dataset and would likely
        need to be adjusted for other datasets.

        Currently, this method does the following:
           1. Remap gender feature to numeric values
           2. Remap smokes feature to numeric values
           3. Impute missing sleep_hours data with most_frequent value

           Note: I intended to use DataFrame's map() for remapping categorical data
           to numeric data, however it seems scikit-learn methods converted the
           DataFrame to numpy arrays where there is an absense of a map() method.

           I've now learned that I can use estimator.set-output(transform='pandas') to
           ensure it remains a DaraFrame, but this code was already in place.
        """
        logger.info("Beginning imputation process...")

        # Get column indices
        gender_index = self.raw_dataset.columns.get_loc('gender')
        smokes_index = self.raw_dataset.columns.get_loc('smokes')

        logger.info("Remapping gender to numeric values...")
        self.X_train[:, gender_index] = np.where(self.X_train[:, gender_index] == 'M', 1, 0)
        self.X_test[:, gender_index] = np.where(self.X_test[:, gender_index] == 'M', 1, 0)

        logger.info("Remapping smokes to numeric values...")
        smoke_mapping = {'yes': 1, 'no': 0, '1': 1, '0': 0}
        self.X_train[:, smokes_index] = np.vectorize(smoke_mapping.get)(self.X_train[:, smokes_index])
        self.X_test[:, smokes_index] = np.vectorize(smoke_mapping.get)(self.X_test[:, smokes_index])

        logger.info("Imputing missing sleep_hours data with most_frequent...")
        # 160 off 2,000 NaN values present in sleep_hours (~8%), all other
        # columns have no NaNs Imputing missing sleep_hours values with
        # most_frequent value
        imputer = SimpleImputer(strategy='most_frequent')
        self.X_train = imputer.fit_transform(self.X_train)
        self.X_test = imputer.transform(self.X_test)


    def preprocess_data(self):
        """
        Perform preprocessing workflow steps on the dataset.

        Steps performed:
           1. Splitting the dataset into features (X) and target (y)
           2. Clean up missing values and convert categorical data to numeric
           2. Scaling features to have a mean of 0 and standard deviation of 1

        """
        logger.info("Preprocessing dataset...")
        self.create_bmi_feature()
        self.split_data()
        self.impute_data()
        self.scale_data()


    def evaluate_models(self):
        """
        Evaluate machine learning models on the dataset.
        """
        logger.info("Evaluating machine learning models...")
        # TODO: Add more models to evaluate
        models = {
                    'KNN': KNeighborsClassifier(),
                    'Logistic Regression': LogisticRegression(),
                    'Decision Tree': DecisionTreeClassifier(),
        }

        results = []
        for key, model in models.items():
            logger.info(f"Evaluating {key}...")
            kf = KFold(n_splits=KFOLD_SPLITS, shuffle=KFOLD_SHUFFLE, random_state=self.random_state)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=kf)
            results.append(scores)
            logger.info(f"Accuracy: {scores.mean()} (std: {scores.std()})")

        plt.boxplot(results, tick_labels=list(models.keys()))
        plt.show()


if __name__ == "__main__":

    app = EvaluateModels(os.path.join(DATA_DIR, DATASET))
    app.preprocess_data()
    app.evaluate_models()
