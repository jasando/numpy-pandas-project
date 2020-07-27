import json
import pandas as pd
import numpy as np

from enum import Enum
from typing import List
from dataset_attributes import DataSetAtrributes
from sklearn.impute import SimpleImputer

import warnings

warnings.filterwarnings('ignore')


class ImputerStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    CONSTANT = 'constant'
    REGRESSOR_MODEL = 'regressor_model'
    CLASSIFICATION_MODEL = 'clasification_model'
    MOST_FREQUENT = 'most_frequent'


class DataBot:
    numeric_types: List[str] = ['int64', 'float64', 'datetime64']
    string_types: List[str] = ['object', 'category']

    def __init__(self,
                 dataset=None,
                 target_name=None,
                 null_threshold=0.3,
                 cardinal_threshold=0.3,
                 project_path=None):
        self.dataset = dataset
        self.target = target_name
        self.null_threshold = null_threshold
        self.cardinal_threshold = cardinal_threshold
        self.project_path = project_path
        self.categorical_columns = []
        self.numeric_columns = []

        self.datasetAttributes = DataSetAtrributes(self.project_path)

        if target_name is not None:
            self.target_name = target_name
            self.target = self.dataset[self.target_name]
            self.features = self.dataset.drop([self.target_name], axis=1)
        else:
            self.features = dataset

    # Lambda for a series object that fill nulls with the mean.
    fill_mean = None
    # Lambda for a series object that fill nulls with the mode.
    fill_mode = None

    def scale_range(self, x):
        return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    def scale_log(self, x):
        return np.log(x + 1)

    impute_strategies = {
        ImputerStrategy.MEAN: fill_mean,
        ImputerStrategy.MODE: fill_mode
    }

    def impute(self, columns, impute_strategy):
        """Impute selected columns (pd.Series) from self.features with the given strategy.

        Parameters
        ----------
        :param columns: list of columns names to impute.
        :param impute_strategy: Selected ImputerStrategy
        """
        print("Column",columns ,"Before", self.features[columns])
        imp = SimpleImputer(missing_values=np.nan, strategy=impute_strategy.value)
        imp = imp.fit(self.features[columns])
        self.features[columns] = imp.transform(self.features[columns])
        print("Column",columns, "After", self.features[columns])

    def one_hot_encode(self, col_name, categorical_values):
        """ Apply one hot encoding to the given column.

        :param col_name: Name of the column to one hot encode.
        :param categorical_values: Unique values from self.features[col_name]
        :return:
        """

        # Get one hot encoding of column
        # one_hot = pd.get_dummies(data=categorical_values, columns=col_name)
        one_hot = pd.get_dummies(self.features[col_name], columns=col_name)
        print("one hot", categorical_values, one_hot)
        # Drop column as it is now encoded
        self.features = self.features.drop(col_name, axis=1)
        # Join the encoded df
        self.features = self.features.join(one_hot)
        print("one hot after", categorical_values, self.features)

    def normalize(self, columns):
        """Apply self.scale_range and self.scale_log to the given columns
        :param columns: list of columns names to normalize
        """
        self.features[columns] = self.features[columns].apply(lambda x: self.scale_range(x))
        self.features[columns] = self.features[columns].apply(lambda x: self.scale_log(x))

    def remove_null_columns(self):
        """Remove columns with a percentage of null values greater than the given threshold (self.null_threshold).
        
        """
        threshold = self.null_threshold * self.row_count()
        columns = self.features.dropna(axis='columns', thresh=threshold, inplace=True)
        # self.datasetAttributes.parameters['removed_columns'] += columns
        self.datasetAttributes.parameters['removed_columns'] += self.dataset.columns.difference(
            self.features.columns).tolist()

    def remove_high_cardinality_columns(self):
        """Remove columns with a cardinality percentage greater than the given threshold (self.cardinal_threshold).

        """
        threshold = self.null_threshold * self.row_count()
        columns_with_high_cardinality = [col for col in self.features.columns if self.features[col].nunique() > threshold]
        columns = self.features.dropna(axis='columns', thresh=threshold, inplace=True)
        # self.datasetAttributes.parameters['removed_columns'] += columns
        self.datasetAttributes.parameters['removed_columns'] += self.dataset.columns.difference(self.features.columns).tolist()

    def row_count(self):
         return len(self.features.index)


    def pre_process(self):
        """Preprocess dataset features before being send to ML algorithm for training.
        """
        # Implement this method with the given indications in the given order

        # Remove columns with null values above the threshold
        self.remove_null_columns()

        # Remove columns with cardinality above the threshold
        self.remove_high_cardinality_columns()

        # Create a python list with the names of columns with numeric values.
        # Numeric columns have one of the types stored in the list self.numeric_types
        self.numeric_columns = list(self.features.select_dtypes(include=self.numeric_types).columns)

        # Create a python list with the names of columns with string values.
        # Categorical columns have one of the types stored in the list self.string_types
        self.categorical_columns = list(self.features.select_dtypes(include=self.string_types).columns)

        # Create a python list with the names of numeric columns with at least one null value.
        print("Numeric columns:", self.numeric_columns)
        print("Categorical columns:", self.categorical_columns)
        print(self.features.head(10))
        # numeric_nulls = self.features.loc[self.numeric_columns, self.features.isnull().any()].columns
        numeric_nulls = [col for col in self.numeric_columns if self.features[col].isnull().any()]

        # Create a python list with the names of categorical columns with at least one null value.
        categorical_nulls = [col for col in self.categorical_columns if self.features[col].isnull().any()]
        # categorical_nulls = self.features.loc[self.categorical_columns, self.features.isnull().any()].columns

        # Impute numerical columns with at least one null value.
        self.impute(numeric_nulls, impute_strategy=ImputerStrategy.MEAN)

        # Impute categorical columns with at least one null value.
        if categorical_nulls:
            self.impute(categorical_nulls, impute_strategy=ImputerStrategy.MOST_FREQUENT)

        # These two lines gather information from the dataset for further use.
        self.datasetAttributes.set_column_values(self.categorical_columns, self.features)
        self.datasetAttributes.set_number_values(self.numeric_columns, self.features)

        # Apply one hot encoding to all categorical columns.
        for col in self.categorical_columns:
            categorical_values = self.features[col].unique()
            self.one_hot_encode(col, categorical_values)

        # Normalize all numeric columns
        # TODO uncomment
        self.normalize(self.numeric_columns)

        # This line store relevant information from the processed dataset for further use.
        # TODO uncomment line below
        self.datasetAttributes.save()
        # print(self.features)
        self.features = self.features.reset_index()
        print(self.features.isnull())
        print("los nuuuuuuulos", self.features.isnull().any())

    def pre_process_prediction(self, parameters):
        """Preprocess records from API calls before running predictions

        :param parameters: information from the processed dataset in the training stage.

        """
        self.features.drop(parameters['removed_columns'], axis=1, inplace=True)

        for column in parameters['categorical_columns'].keys():
            categorical_values = parameters['categorical_columns'][column]["values"]
            self.one_hot_encode(column, categorical_values)

        for column in parameters['numeric_columns'].keys():
            n_min = parameters['numeric_columns'][column]["min"]
            n_max = parameters['numeric_columns'][column]["max"]
            self.features[column] = self.features[column].apply(lambda x: (x - n_min) / (n_max - n_min))
            self.features[column] = self.features[column].apply(lambda x: np.log(x + 1))

    def get_dataset(self):
        """Returns a dataset with features and labels.

        :return: Dataset with features and labels.
        """
        self.dataset = self.features
        self.dataset[self.target_name] = self.target
        return self.dataset
