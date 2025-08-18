import numpy as np

def format_currency(amount):
    """Format currency in Indian Rupees"""
    if amount >= 100000:
        return f"₹{amount/100000:.1f}L"
    elif amount >= 1000:
        return f"₹{amount/1000:.0f}K"
    else:
        return f"₹{amount:,.0f}"

def create_prediction_explanation(bedrooms, locality, area, furnishing, predicted_price):
    """Create explanation for the prediction"""
    explanation = f"""
    **How we calculated this estimate:**
    
     **Property Type:** {bedrooms} apartment with {area:,} sq ft area
     **Location:** {locality} - a key factor in determining rental prices
     **Furnishing:** {furnishing} status affects the rental value
    
    **Price Factors:**
    - Location contributes significantly to the final price
    - Area size directly impacts the rental value
    - Property type and furnishing status add to the premium
    
    **Market Context:**
    Based on similar properties in {locality}, the estimated price of ₹{predicted_price:,.0f} 
    falls within the expected range for this type of property.
    
     **Tip:** Prices may vary based on exact location, building amenities, 
    floor level, and current market conditions.
    """
    
    return explanation

def validate_input_data(bedrooms, bathrooms, area, locality):
    """Validate user input data"""
    errors = []
    
    if not bedrooms:
        errors.append("Please select number of bedrooms")
    
    if bathrooms < 1 or bathrooms > 10:
        errors.append("Number of bathrooms should be between 1 and 10")
    
    if area < 200 or area > 10000:
        errors.append("Area should be between 200 and 10,000 sq ft")
    
    if not locality or locality == "Unknown":
        errors.append("Please select a valid locality")
    
    return errors

def get_locality_insights(data, locality):
    """Get insights about a specific locality"""
    locality_data = data[data['Locality_clean'] == locality]
    
    if len(locality_data) == 0:
        return "No data available for this locality"
    
    avg_price = locality_data['Price_clean'].mean()
    min_price = locality_data['Price_clean'].min()
    max_price = locality_data['Price_clean'].max()
    property_count = len(locality_data)
    
    insights = f"""
    **{locality} Market Insights:**
    - Average Rent: {format_currency(avg_price)}
    - Price Range: {format_currency(min_price)} - {format_currency(max_price)}
    - Properties Available: {property_count}
    """
    
    return insights
