# Hyderabad House Price Predictor

A machine learning-powered web application that predicts house rental prices in Hyderabad, India. Built with Streamlit and Random Forest regression, this app provides accurate price estimates based on property characteristics.

## Features

- **Smart Price Prediction**: Uses Random Forest ML model for accurate estimates
- **Interactive Interface**: Easy-to-use web interface for property details input
- **Data Visualizations**: Market insights and trends analysis
- **Model Performance Metrics**: Detailed accuracy and feature importance charts
- **Help & Guidance**: Built-in tutorial for users

## Screenshots

![App Interface](screenshot.png) *(Add a screenshot of your app)*

## Installation

### Requirements
- Python 3.11+
- Required packages (see requirements below)

### Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/hyderabad-house-price-predictor.git
cd hyderabad-house-price-predictor
```

2. **Install dependencies:**
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

### Using Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install streamlit pandas numpy scikit-learn plotly joblib
streamlit run app.py
```

## Usage

1. **Select Property Details**: Choose bedrooms, bathrooms, and furnishing type
2. **Enter Location**: Select locality from the dropdown menu
3. **Specify Area**: Input property area in square feet
4. **Get Prediction**: Click "Predict Price" for instant estimate

## Input Parameters

- **Bedrooms**: 1 BHK, 2 BHK, 3 BHK, 4 BHK, Studio
- **Bathrooms**: Number of bathrooms (1-5)
- **Locality**: 100+ Hyderabad locations
- **Area**: Property size in square feet
- **Furnishing**: Furnished, Semi-Furnished, Unfurnished
- **Tenant Type**: Bachelors, Family, or Both

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: 6 key property characteristics
- **Dataset**: 1,100+ Hyderabad property listings
- **Accuracy**: R² score of ~0.85
- **Validation**: 5-fold cross-validation

## Project Structure

```
├── app.py                    # Main Streamlit application
├── data_processor.py         # Data cleaning and preprocessing
├── model_trainer.py          # Machine learning model training
├── utils.py                  # Helper functions
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── attached_assets/
│   └── Hyderabad_House_Data_*.csv  # Dataset
├── pyproject.toml           # Python dependencies
└── README.md               # This file
```

## Data Processing

The application includes comprehensive data cleaning:
- Price normalization and outlier removal
- Area standardization
- Locality name cleaning and encoding
- Feature encoding for machine learning

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].

## Acknowledgments

- Data sourced from Hyderabad rental market listings
- Built with Streamlit, scikit-learn, and Plotly
- Designed for real estate professionals and property managers