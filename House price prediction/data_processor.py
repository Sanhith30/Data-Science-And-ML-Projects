import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_and_clean_data(self, file_path):
        """Load and clean the house data"""
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Drop the unnamed index column if it exists
            if df.columns[0].startswith('Unnamed'):
                df = df.drop(df.columns[0], axis=1)
            
            # Clean the data
            df_cleaned = self._clean_data(df)
            
            return df_cleaned
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _clean_data(self, df):
        """Clean and preprocess the data"""
        df_clean = df.copy()
        
        # Clean price column
        df_clean['Price_clean'] = self._clean_price_column(df_clean['Price'])
        
        # Clean area column
        df_clean['Area_clean'] = self._clean_area_column(df_clean['Area'])
        
        # Clean bedrooms column
        df_clean['Bedrooms_clean'] = self._clean_bedrooms_column(df_clean['Bedrooms'])
        
        # Clean bathrooms column
        df_clean['Bathrooms_clean'] = self._clean_bathrooms_column(df_clean['Bathrooms'])
        
        # Clean locality column
        df_clean['Locality_clean'] = self._clean_locality_column(df_clean['Locality'])
        
        # Clean furnishing column
        df_clean['Furnishing_clean'] = self._clean_furnishing_column(df_clean['Furnishing'])
        
        # Clean tenant type column
        df_clean['Tennants_clean'] = self._clean_tenant_column(df_clean['Tennants'])
        
        # Remove rows with missing critical data
        df_clean = df_clean.dropna(subset=['Price_clean', 'Area_clean', 'Bedrooms_clean'])
        
        # Remove outliers
        df_clean = self._remove_outliers(df_clean)
        
        return df_clean
    
    def _clean_price_column(self, price_col):
        """Clean the price column"""
        def clean_price(price_str):
            if pd.isna(price_str):
                return np.nan
            
            # Convert to string and remove quotes
            price_str = str(price_str).strip().strip('"')
            
            # Remove commas and extract numbers
            price_str = re.sub(r'[,"]', '', price_str)
            
            # Extract numeric value
            numbers = re.findall(r'\d+', price_str)
            if numbers:
                return float(numbers[0])
            return np.nan
        
        return price_col.apply(clean_price)
    
    def _clean_area_column(self, area_col):
        """Clean the area column"""
        def clean_area(area_str):
            if pd.isna(area_str):
                return np.nan
            
            area_str = str(area_str).lower()
            
            # Skip invalid entries
            if 'read more' in area_str or 'facing' in area_str:
                return np.nan
            
            # Extract numeric value
            numbers = re.findall(r'\d+', area_str)
            if numbers:
                area_val = float(numbers[0])
                # Reasonable area bounds
                if 200 <= area_val <= 10000:
                    return area_val
            return np.nan
        
        return area_col.apply(clean_area)
    
    def _clean_bedrooms_column(self, bedrooms_col):
        """Clean the bedrooms column"""
        def clean_bedrooms(bedroom_str):
            if pd.isna(bedroom_str):
                return 'Unknown'
            
            bedroom_str = str(bedroom_str).upper()
            
            if 'STUDIO' in bedroom_str:
                return 'Studio'
            elif '1 BHK' in bedroom_str:
                return '1 BHK'
            elif '2 BHK' in bedroom_str:
                return '2 BHK'
            elif '3 BHK' in bedroom_str:
                return '3 BHK'
            elif '4 BHK' in bedroom_str:
                return '4 BHK'
            else:
                return 'Unknown'
        
        return bedrooms_col.apply(clean_bedrooms)
    
    def _clean_bathrooms_column(self, bathrooms_col):
        """Clean the bathrooms column"""
        def clean_bathrooms(bathroom_str):
            if pd.isna(bathroom_str):
                return 1
            
            bathroom_str = str(bathroom_str)
            
            # Extract number
            numbers = re.findall(r'\d+', bathroom_str)
            if numbers:
                num_bathrooms = int(numbers[0])
                # Reasonable bounds
                if 0 <= num_bathrooms <= 10:
                    return max(1, num_bathrooms)  # At least 1 bathroom
            return 1  # Default to 1
        
        return bathrooms_col.apply(clean_bathrooms)
    
    def _clean_locality_column(self, locality_col):
        """Clean the locality column"""
        def clean_locality(locality_str):
            if pd.isna(locality_str):
                return 'Unknown'
            
            locality_str = str(locality_str).strip()
            
            # Remove extra spaces and standardize
            locality_str = re.sub(r'\s+', ' ', locality_str)
            
            # Extract main locality name (remove additional details)
            # Remove text in parentheses and after commas
            locality_str = re.split(r'[,()]', locality_str)[0].strip()
            
            # Remove common prefixes/suffixes
            locality_str = re.sub(r'^(near|close to|behind)\s+', '', locality_str, flags=re.IGNORECASE)
            
            return locality_str.title() if locality_str else 'Unknown'
        
        return locality_col.apply(clean_locality)
    
    def _clean_furnishing_column(self, furnishing_col):
        """Clean the furnishing column"""
        def clean_furnishing(furnishing_str):
            if pd.isna(furnishing_str):
                return 'Unknown'
            
            furnishing_str = str(furnishing_str).lower()
            
            if 'furnished' in furnishing_str and 'semi' not in furnishing_str and 'un' not in furnishing_str:
                return 'Furnished'
            elif 'semi' in furnishing_str:
                return 'Semi-Furnished'
            elif 'unfurnished' in furnishing_str or 'un-furnished' in furnishing_str:
                return 'Unfurnished'
            else:
                return 'Unknown'
        
        return furnishing_col.apply(clean_furnishing)
    
    def _clean_tenant_column(self, tenant_col):
        """Clean the tenant type column"""
        def clean_tenant(tenant_str):
            if pd.isna(tenant_str):
                return 'Unknown'
            
            tenant_str = str(tenant_str).lower()
            
            if 'bachelor' in tenant_str and 'family' in tenant_str:
                return 'Bachelors/Family'
            elif 'family' in tenant_str:
                return 'Family'
            elif 'bachelor' in tenant_str:
                return 'Bachelors'
            else:
                return 'Unknown'
        
        return tenant_col.apply(clean_tenant)
    
    def _remove_outliers(self, df):
        """Remove price and area outliers"""
        # Remove price outliers (using IQR method)
        Q1_price = df['Price_clean'].quantile(0.25)
        Q3_price = df['Price_clean'].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        
        lower_bound_price = Q1_price - 1.5 * IQR_price
        upper_bound_price = Q3_price + 1.5 * IQR_price
        
        df = df[(df['Price_clean'] >= lower_bound_price) & (df['Price_clean'] <= upper_bound_price)]
        
        # Remove area outliers
        Q1_area = df['Area_clean'].quantile(0.25)
        Q3_area = df['Area_clean'].quantile(0.75)
        IQR_area = Q3_area - Q1_area
        
        lower_bound_area = Q1_area - 1.5 * IQR_area
        upper_bound_area = Q3_area + 1.5 * IQR_area
        
        df = df[(df['Area_clean'] >= lower_bound_area) & (df['Area_clean'] <= upper_bound_area)]
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features for machine learning"""
        df_encoded = df.copy()
        
        categorical_columns = ['Bedrooms_clean', 'Furnishing_clean', 'Tennants_clean', 'Locality_clean']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def prepare_prediction_input(self, bedrooms, bathrooms, furnishing, tenant_type, area, locality, training_data):
        """Prepare input data for prediction"""
        # Create input array in the same order as training features
        input_data = []
        
        # Encode bedrooms
        if 'Bedrooms_clean' in self.label_encoders:
            try:
                bedrooms_encoded = self.label_encoders['Bedrooms_clean'].transform([bedrooms])[0]
            except ValueError:
                # If new category, use the most common one
                bedrooms_encoded = 0
        else:
            bedrooms_encoded = 0
        
        input_data.append(bedrooms_encoded)
        input_data.append(bathrooms)
        
        # Encode furnishing
        if 'Furnishing_clean' in self.label_encoders:
            try:
                furnishing_encoded = self.label_encoders['Furnishing_clean'].transform([furnishing])[0]
            except ValueError:
                furnishing_encoded = 0
        else:
            furnishing_encoded = 0
        
        input_data.append(furnishing_encoded)
        
        # Encode tenant type
        if 'Tennants_clean' in self.label_encoders:
            try:
                tenant_encoded = self.label_encoders['Tennants_clean'].transform([tenant_type])[0]
            except ValueError:
                tenant_encoded = 0
        else:
            tenant_encoded = 0
        
        input_data.append(tenant_encoded)
        input_data.append(area)
        
        # Encode locality
        if 'Locality_clean' in self.label_encoders:
            try:
                locality_encoded = self.label_encoders['Locality_clean'].transform([locality])[0]
            except ValueError:
                # If new locality, use the most common one from training data
                most_common_locality = training_data['Locality_clean'].mode()[0]
                locality_encoded = self.label_encoders['Locality_clean'].transform([most_common_locality])[0]
        else:
            locality_encoded = 0
        
        input_data.append(locality_encoded)
        
        return input_data
