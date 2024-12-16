import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

class MLModelPipeline:
    def __init__(self, model=None, train_size=0.8, random_state=11):
        """
        Initialize the pipeline with optional parameters.
        
        Parameters:
        - model: Machine learning model (default is XGBClassifier).
        - train_size: Proportion of the dataset to include in the train split (default: 0.8).
        - random_state: Random seed for reproducibility (default: 11).
        """
        self.model = model or XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.train_size = train_size
        self.random_state = random_state

    def load_data(self, conversion_data_path, nonconversion_data_path):
        """
        Load the datasets from CSV files and ensure column compatibility.
        
        Parameters:
        - conversion_data_path: Path to the conversion data CSV file.
        - nonconversion_data_path: Path to the non-conversion data CSV file.
        """
        # Load the raw data
        self.conversion_data = pd.read_csv(conversion_data_path)
        self.nonconversion_data = pd.read_csv(nonconversion_data_path)

        # Clean column names
        self.conversion_data.columns = self.conversion_data.columns.str.replace(
            r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', regex=True
        )
        self.nonconversion_data.columns = self.nonconversion_data.columns.str.replace(
            r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', regex=True
        )

        # Ensure column names match
        if not (self.conversion_data.columns == self.nonconversion_data.columns).all():
            raise ValueError("Column names do not match between conversion and non-conversion data.")

    def preprocess_data(self):
        """
        Preprocess the data by:
        - Dropping missing values
        - Creating a target column
        - Splitting datasets into train and test sets
        - Balancing the test set ratio
        """
        # Drop rows with missing values
        for dataset in [self.conversion_data, self.nonconversion_data]:
            dataset.dropna(inplace=True)
    
        # Add target column
        self.conversion_data['CLASS'] = 1
        self.nonconversion_data['CLASS'] = 0
    
        # Shuffle data
        self.conversion_data = self.conversion_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.nonconversion_data = self.nonconversion_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
        # Determine train-test split size for conversion data
        n = int(self.train_size * len(self.conversion_data))
        self.conversion_train = self.conversion_data[:n]
        self.conversion_test = self.conversion_data[n:]
    
        # Use the same train size for non-conversion train data
        self.nonconversion_train = self.nonconversion_data[:n]
    
        # Dynamically balance the test set with a 3:1 non-conversion to conversion ratio
        m = len(self.conversion_test)
        self.nonconversion_test = self.nonconversion_data[n: n + 3 * m]
    
        # Combine conversion and non-conversion data for train and test sets
        self.train_data = pd.concat([self.conversion_train, self.nonconversion_train]).reset_index(drop=True)
        self.test_data = pd.concat([self.conversion_test, self.nonconversion_test]).reset_index(drop=True)
    
        # Drop the SITE column if it exists
        for dataset in [self.train_data, self.test_data]:
            if 'SITE' in dataset.columns:
                dataset.drop(columns=['SITE'], inplace=True)
    
    def encode_data(self):
        """
        Encode categorical features using Ordinal Encoding.
        """
        # Identify categorical columns
        columns_to_encode = self.train_data.select_dtypes(include=['object']).columns
        
        # Initialize encoder with handling for unknown categories
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Fit and transform training data
        self.train_data[columns_to_encode] = ordinal_encoder.fit_transform(self.train_data[columns_to_encode])
        
        # Transform test data
        self.test_data[columns_to_encode] = ordinal_encoder.transform(self.test_data[columns_to_encode])
    
    def train_model(self):
        """
        Train the machine learning model on the training dataset.
        """
        self.model = XGBClassifier(
            colsample_bytree=0.8,
            learning_rate=0.2,
            max_depth=7,
            min_child_weight=1,
            n_estimators=500,
            subsample=1.0,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.model.fit(self.train_data.drop(columns=['CLASS']), self.train_data['CLASS'])
    
    def evaluate_model(self):
        """
        Evaluate the model on the test dataset and return metrics.
        """
        y_pred = self.model.predict(self.test_data.drop(columns=['CLASS']))
        y_pred_proba = self.model.predict_proba(self.test_data.drop(columns=['CLASS']))[:, 1]
        return {
            'Accuracy':  float(round(accuracy_score(self.test_data['CLASS'], y_pred), 2)),
            'Precision': float(round(precision_score(self.test_data['CLASS'], y_pred), 2)),
            'Recall':    float(round(recall_score(self.test_data['CLASS'], y_pred), 2)),
            'F1-Score':  float(round(f1_score(self.test_data['CLASS'], y_pred), 2)),
            'ROC-AUC':   float(round(roc_auc_score(self.test_data['CLASS'], y_pred_proba), 2))
        }
    
    def run_pipeline(self, conversion_data_path, nonconversion_data_path):
        """
        Run the entire pipeline: data loading, preprocessing, encoding, training, and evaluation.
        
        Parameters:
        - conversion_data_path: Path to the conversion data CSV file.
        - nonconversion_data_path: Path to the non-conversion data CSV file.
        
        Returns:
        - metrics (dict): Evaluation metrics for the model.
        """
        print("Loading data...")
        self.load_data(conversion_data_path, nonconversion_data_path)
        print("Preprocessing data...")
        self.preprocess_data()
        print("Encoding data...")
        self.encode_data()
        print("Training model...")
        self.train_model()
        print("Evaluating model...")
        metrics = self.evaluate_model()
        print("Pipeline completed successfully.")
        return metrics