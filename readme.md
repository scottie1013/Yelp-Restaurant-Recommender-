

# Yelp Rating Predictor

Website Link: https://9qj22pj8ec9art4bykrysv.streamlit.app/ 

## Overview
The Yelp Rating Predictor is an interactive web application that predicts how a specific user would rate a restaurant based on their past rating behavior and the characteristics of the restaurant. The application also provides personalized restaurant recommendations for users based on predicted ratings.

## Features
- **Search by ID**: Enter a user ID and business ID to predict how that user would rate that restaurant
- **Discover Restaurants**: Browse through users and get personalized restaurant recommendations
- **Interactive UI**: View restaurant details, user profiles, and predicted ratings in a user-friendly interface
- **Sample Reviews**: See sample reviews for each restaurant to get a feel for the dining experience

## Data
This application uses data from the [Yelp Dataset Challenge](https://www.yelp.com/dataset), which includes:
- **User data**: Information about users, including their review history, average ratings, and engagement metrics
- **Business data**: Details about restaurants, including location, categories, and overall ratings
- **Reviews**: User reviews of businesses
- **Tips**: Short comments left by users about businesses

### Big Data Processing
The application leverages **Apache Spark** and **PySpark** to efficiently process and analyze the large-scale Yelp dataset:
- **Distributed Computing**: Utilizes Spark's distributed computing capabilities to handle millions of reviews and user interactions
- **Parallel Processing**: Implements parallel data processing to extract features and train models on large datasets
- **Efficient Filtering**: Uses Spark SQL and DataFrame operations to filter and transform the raw data
- **Scalable Architecture**: Built on a scalable architecture that can handle the full Yelp dataset (several GB of data)

The data has been preprocessed and filtered to focus on restaurants and relevant features for prediction.

## Technical Architecture

### recommender.py
This module contains the core recommendation engine and data processing functionality:

- **Data Processing with PySpark**:
  - `process_user_features()`: Extracts and normalizes user features like review count, average rating, and engagement metrics
  - `process_business_features()`: Extracts and normalizes business features like category, location, and overall rating
  - `process_tips()`: Analyzes tips data to extract engagement patterns

- **Feature Engineering**:
  - `create_feature_vector()`: Combines user, business, and interaction features into a single vector for prediction
  - `get_global_stats()`: Calculates global statistics used for feature normalization

- **Model Handling**:
  - `load_model_and_scaler()`: Loads the trained machine learning model and feature scaler
  - `train_model()`: Functionality to train or retrain the model using Spark ML

### app.py
This is the Streamlit web application that provides the user interface:

- **UI Components**:
  - Tab-based navigation between "Search by ID" and "Discover Restaurants"
  - Interactive cards displaying user and business information
  - Visual representation of ratings using star icons

- **Key Functions**:
  - `make_prediction()`: Processes user and business data to generate a rating prediction
  - `get_top_recommendations_for_user()`: Generates personalized restaurant recommendations
  - `get_sample_reviews_for_business()`: Creates realistic sample reviews for restaurants
  - `get_valid_example_pairs()`: Provides example user-business pairs for easy testing

- **Data Visualization**:
  - Displays user profiles with avatars and statistics
  - Shows restaurant information with images and details
  - Presents ratings and reviews in an intuitive format

## How to Use

### Search by ID
1. Enter a valid user ID in the "User ID" field
2. Enter a valid business ID in the "Business ID" field
3. Click "Predict Rating"
4. View the predicted rating, business information, and sample reviews

### Discover Restaurants
1. Browse through the available users
2. Click "Select [User Name]" to choose a user
3. View personalized restaurant recommendations for that user
4. Click "View Details" on any restaurant to see more information and sample reviews

## Installation and Setup
1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have Apache Spark installed and configured
4. Ensure you have the data files in the correct directory structure
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Requirements
- Python 3.7+
- Apache Spark 3.0+
- PySpark
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (for visualizations)
- Pillow (for image processing)

## Model Details
The recommendation system uses a machine learning model trained on historical user-restaurant interactions using Spark ML. The model takes into account:

- User characteristics (review history, rating patterns)
- Restaurant attributes (location, cuisine type, price range)
- Interaction features (tips, review text sentiment)

The model predicts a rating on a scale of 1-5 stars that represents how much a specific user would enjoy a specific restaurant.

### Big Data Advantages
- **Scalability**: The Spark-based architecture allows the system to scale to millions of users and businesses
- **Performance**: Distributed processing enables fast predictions even with complex feature engineering
- **Flexibility**: The system can easily incorporate additional data sources and features

## Future Improvements
- Integration with real-time Yelp data
- More sophisticated recommendation algorithms using Spark MLlib
- User authentication to save preferences
- Mobile-friendly responsive design
- Additional filters for restaurant discovery
- Deployment on a cloud-based Spark cluster for improved performance

## Contributors
- [Your Name/Team]

## License
This project uses data from the Yelp Dataset Challenge, which is subject to Yelp's terms of use.

---

*Note: This application is for educational purposes only and is not affiliated with Yelp Inc.*
