
"""

The module includes the pre processing to generate the data.


"""

import os
import string
import numpy as np
import pandas as pd


from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1889)


class PreProcessData:
    
    
    """
    
    The constructor contains the global variables and creates the finalised the dataframe
    
    Some variables are placed to make it easier to code the logic (e.g. min max sclaing)
    
    Hyperparameters:
        - query -> the string name to print out on console
        
    Azhan comments:
        - Since the database is fixed, it make sense to add data configuration. Assuming it varies then it can be a function and pass in the columns
        - I guess an improvement can be adding them as a hyperparameter
        - I'm assume these variables were chosen for a reason? If so, I may need to do more research.
        - The query is added even though I don't understand the purpose (maybe to give it a label?)

    Other error handling includes:
        - checking if features is above 1
        - check rows are above 0
    
    
    """
    def __init__(self, query):
        
        self.query = query
        
        self.rows = 10_000
        self.features = 16
        
        
        self.classification_columns = [
            "age",
            "height_cm",
            "weight_kg",
            "income",
            "financial_hist_1",
            "financial_hist_2",
            "financial_hist_3",
            "financial_hist_4",
            "credit_score_1",
            "credit_score_2",
            "credit_score_3",
            "insurance_hist_1",
            "insurance_hist_2",
            "insurance_hist_3",
            "insurance_hist_4",
            "insurance_hist_5"
        ]
        
        self.min_max_scaler_transform_columns = {
            "age":{
            	"feature_range_min": 18,
            	"feature_range_max": 95
            },
            "height_cm":{
            	"feature_range_min": 140,
            	"feature_range_max": 210
            },
            "weight_kg":{
            	"feature_range_min": 45,
            	"feature_range_max": 125
            },
            "income":{
            	"feature_range_min": 0,
            	"feature_range_max": 250_000
            },
            "credit_score_1":{
            	"feature_range_min": 0,
            	"feature_range_max": 999
            },
            "credit_score_2":{
            	"feature_range_min": 0,
            	"feature_range_max": 700
            },
            "credit_score_3":{
            	"feature_range_min": 0,
            	"feature_range_max": 710
            }
        }
        
        self.categorical_columns = [
            "gender",
            "marital_status",
            "occupation",
            "location",
            "prev_claim_rejected",
            "known_health_conditions",
            "uk_residence",
            "family_history_1",
            "family_history_2",
            "family_history_4",
            "family_history_5",
            "product_var_1",
            "product_var_2",
            "product_var_3",
            "health_status",
            "driving_record",
            "previous_claim_rate",
            "education_level",
            "income_level",
            "n_dependents",
        ]
        
        # I added this in, maybe there's a reason I'm unaware (suitable for terminal running)
        print(self.query)
        
        df = self.create_classification_data()
        df = self.create_min_max_scale_data(df)
        df = self.create_remaining_data(df)
        df = self.configure_categorical_data_types(df)
        
        self.data = df

    
    """
    
    Returns finalised dataframe expressing the data
    
    
    """
    def get_data(self) -> pd.DataFrame:
        return self.data
    
    
    """
    
    Configure specific columns to categorical in the dataframe
    
    """
    def configure_categorical_data_types(self, df) -> pd.DataFrame:
        for column in self.categorical_columns:
            df[column] = df[column].astype("category")
        
        return df
        
        
    
    
    """
    
    Generates the classification data columns
    
    Returns the dataframe only for columns thats are mapped as classification
    
    
    """
    def create_classification_data(self) -> pd.DataFrame:

        if self.rows <= 0:
            print("Rows must be above zero")
            return []

        if self.features <= 0:
            print("Features must be above zero")
            return []
        
        features, labels = make_classification(
            n_samples= self.rows,
            n_features= self.features,
            n_informative= 7,
            n_redundant= 4,
            n_repeated= 3,
            n_classes= 2,
            class_sep= 1.2,
            flip_y= 0.035,
            weights= [0.85, 0.15],
            random_state= 1889,
        )
        
        df = pd.DataFrame(features)
        df.columns = self.classification_columns
        
        df.insert(value=labels, loc=0, column="claim_status")
        
        return df
    
    
    
    
    """
    
    This generates the data that is scaled down to Min Max Scaler
    
    Hyperparameters:
        df -> the dataframe data
    
    Returns the dataframe and scale down Min Max Scaler using specific columns 
    
    
    """
    def create_min_max_scale_data(self, df) -> pd.DataFrame:

        if len(df) == 0:
            print("The data is empty")
            return []
        
        for key, value in self.min_max_scaler_transform_columns.items():
            
            df[key] = MinMaxScaler(feature_range=(value['feature_range_min'], value['feature_range_max'])).fit_transform(
                df[key].values[:, None]
            )
            df[key] = df[key].astype("int")
            
        return df
    
    
    """
    
    This generates the remaining columns based on specific conditions
    
    Hyperparameters:
        df -> the dataframe data
    
    Returns the dataframe with the appended remaining columns
    
    
    """
    def create_remaining_data(self, df) -> pd.DataFrame:
        
        letter_choice = ["A", "B", "C", "D", "E", "F", "G"]
        number_choice = [0, 1, 2, 3, 4, 5]
        number_without_zero = number_choice[-5:]
        binary_choice = number_choice[:2]
        
        df["bmi"] = (df["weight_kg"] / ((df["height_cm"] / 100) ** 2)).astype(int)
        
        df["gender"] = np.where(
            df["claim_status"] == 0,
            np.random.choice(binary_choice, size=(self.rows), p=[0.46, 0.54]),
            np.random.choice(binary_choice, size=(self.rows), p=[0.52, 0.48]),
        )
        
        df["marital_status"] = np.random.choice(letter_choice[:-1], size=(self.rows), p=[0.2, 0.15, 0.1, 0.25, 0.15, 0.15],)
        
        df["occupation"] = np.random.choice(letter_choice, size=(self.rows))
        
        df["location"] = np.random.choice(list(string.ascii_uppercase), size=(self.rows))
        
        df["prev_claim_rejected"] = np.where(
            df["claim_status"] == 0,
            np.random.choice(binary_choice, size=(self.rows), p=[0.08, 0.92]),
            np.random.choice(binary_choice, size=(self.rows), p=[0.16, 0.84]),
        )
        
        df["known_health_conditions"] = np.random.choice(binary_choice, size=(self.rows), p=[0.06, 0.94])
        df["uk_residence"] = np.random.choice(binary_choice, size=(self.rows), p=[0.76, 0.24])
        
        df["family_history_1"] = np.random.choice(binary_choice, size=(self.rows), p=[0.22, 0.78])
        df["family_history_2"] = np.random.choice(binary_choice, size=(self.rows), p=[0.25, 0.75])
        df["family_history_3"] = np.random.choice((binary_choice + [None]), size=(self.rows), p=[0.12, 0.81, 0.07])
        df["family_history_4"] = np.random.choice(binary_choice, size=(self.rows), p=[0.27, 0.73])
        df["family_history_5"] = np.random.choice(binary_choice, size=(self.rows), p=[0.31, 0.69])
        
        df["product_var_1"] = np.random.choice(binary_choice, size=(self.rows), p=[0.38, 0.62])
        df["product_var_2"] = np.random.choice(binary_choice, size=(self.rows), p=[0.55, 0.45])
        df["product_var_3"] = np.random.choice(letter_choice[:4], size=(self.rows), p=[0.23, 0.28, 0.31, 0.18])
        df["product_var_4"] = np.random.choice(binary_choice, size=(self.rows), p=[0.76, 0.24])
        
        
        df["health_status"] = np.random.randint(1, 5, size=(self.rows))
        df["driving_record"] = np.random.randint(1, 5, size=(self.rows))
        
        df["previous_claim_rate"] = np.where(
            df["claim_status"] == 0,
            np.random.choice(number_without_zero, size=(self.rows), p=[0.48, 0.29, 0.12, 0.08, 0.03]),
            np.random.choice(number_without_zero, size=(self.rows), p=[0.12, 0.28, 0.34, 0.19, 0.07]),
        )
        
        df["education_level"] = np.random.randint(0, 7, size=(self.rows))
        
        df["income_level"] = pd.cut(df["income"], bins=5, labels=False, include_lowest=True)

        df["n_dependents"] = np.random.choice(number_without_zero, size=(self.rows), p=[0.23, 0.32, 0.27, 0.11, 0.07])
        
        df["employment_type"] = np.random.choice((binary_choice + [None]), size=(self.rows), p=[0.16, 0.7, 0.14])
        
        return df
    
    
    
    """
    
    This saves the data to a spreadsheet in the data directory.
    
    Hyperparameters:
        filename -> the saved file's filename for the dataframe
    
    Returns true if successful. Otherwise, false is returned.
    
    
    """
    def save_data_to_spreadsheet(self, filename) -> bool:
        
        data_path = str(Path(__file__).parent.parent) + "/data/"
        
        try:
            os.mkdir(data_path)
        except Exception as error:
            print("There was an error ", error)
            return False
        
        try:
            self.data.to_csv((data_path + filename + ".csv"), index=False)
        except Exception as error:
            print("There was an error ", error)
            return False
        
        return True
    
    
    """
    
    This produce statistics of the number of null rows, percentage and the data types.
    
    Returns the missing data statistic dataframe.
    
    Azhan's comments:
        - It seems some of the datatype in the juypter is int64 rather than float64 even though I print the same code, I'm assuming this is a Python or library version.
        - In theory it should work as it's numeric
        - It maybe different version
    
    """
    def missing_statistics_data(self) -> pd.DataFrame:
        
        missind_data_df = pd.DataFrame()
        
        missind_data_df['Total'] = self.data.isnull().sum()
        missind_data_df['Percentage'] = (self.data.isnull().sum() / self.data.isnull().count() * 100)
        missind_data_df['Types'] = self.data.dtypes.values.tolist()
        
        return missind_data_df
    
    
    """
    
    Remove columns that contains null
    
    I set the 'errors' hyperparameter to 'ignore' because it may return an error - 
    it doesn't exist even though it's right (assuming I run it again separately)
    
    Hyperparameters:
        columns -> the list of columns to drop that has none
    
    Return dataframe with dataframe excluding the chosen columns

    Azhan comments:
        - I added the errors='ignore' as you may run this again, which can return an error.
    
    """
    def drop_none_columns(self, columns) -> pd.DataFrame:
        return self.data.drop(columns=columns, errors='ignore')
    
    
    """
    
    Hyperparameters:
        column -> the column you want counting
    
    Returns the number of rows given in a column
    
    
    """
    def count_rows(self, column) -> pd.DataFrame:

        if column in self.data.columns:
            print("The column does not exist")
            return []

        return self.data[column].value_counts()
    
    
        
    
    
    
    
    
        
    
    
    
        
        
        

