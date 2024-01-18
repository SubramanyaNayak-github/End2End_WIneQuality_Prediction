import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from mlProject.entity.config_entity import (DataTransformationConfig)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: we can add different data transformation techniques such as Scaler, PCA and all
    #we can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already clean


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25)--> default split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        

