import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_processor import DataProcessor

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.processor = DataProcessor()
        
    def train_model(self, data):
        """Train the machine learning model"""
        try:
            # Encode categorical features
            data_encoded = self.processor.encode_categorical_features(data)
            
            # Prepare features and target
            feature_columns = [
                'Bedrooms_clean_encoded',
                'Bathrooms_clean',
                'Furnishing_clean_encoded',
                'Tennants_clean_encoded',
                'Area_clean',
                'Locality_clean_encoded'
            ]
            
            # Check if all feature columns exist
            missing_cols = [col for col in feature_columns if col not in data_encoded.columns]
            if missing_cols:
                print(f"Missing columns: {missing_cols}")
                return None, None, None
            
            X = data_encoded[feature_columns]
            y = data_encoded['Price_clean']
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                print("Insufficient data for training")
                return None, None, None
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            cv_score = cv_scores.mean()
            
            # Feature importance
            feature_importance = self.model.feature_importances_
            
            # Feature names for display
            feature_names = [
                'Bedrooms',
                'Bathrooms', 
                'Furnishing',
                'Tenant Type',
                'Area',
                'Locality'
            ]
            
            metrics = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'cv_score': cv_score,
                'feature_importance': feature_importance
            }
            
            return self.model, metrics, feature_names
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None, None, None
    
    def predict(self, input_data):
        """Make a prediction using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict([input_data])[0]
