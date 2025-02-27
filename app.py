import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import random
from datetime import datetime
import base64
from PIL import Image
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Yelp Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Yelp-like styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f5f5f5;
    }
    
    /* Header styling */
    .header-container {
        background-color: #d32323;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
    }
    
    .header-logo {
        font-size: 2.5rem;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    
    /* Business card styling */
    .business-card {
        display: flex;
        margin-bottom: 1rem;
        border: 1px solid #eee;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .business-image {
        width: 150px;
        height: 150px;
        object-fit: cover;
    }
    
    .business-info {
        padding: 1rem;
        flex: 1;
    }
    
    .business-name {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #0073bb;
    }
    
    .business-rating {
        margin-bottom: 0.5rem;
    }
    
    .business-details {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Star rating styling */
    .star-rating {
        color: #f15c4f;
        font-size: 1.5rem;
    }
    
    .empty-star {
        color: #ccc;
        font-size: 1.5rem;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background-color: #fff8e1;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin-top: 1rem;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #d32323;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #d32323;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# Make sure XGBoost is installed and imported
try:
    from xgboost import XGBRegressor
except ImportError:
    st.error("XGBoost is not installed. Please install it with: pip install xgboost")
    st.stop()

# Add the directory containing recommender.py to the path so we can import functions
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import functions from recommender.py
try:
    from recommender import create_feature_vector, process_user_features, process_business_features, process_tips
except ImportError:
    st.error("Could not import functions from recommender.py. Make sure it's in the same directory.")
    st.stop()

# Load the saved model and scaler
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            return None, None, f"Model file not found at: {model_path}"
        if not os.path.exists(scaler_path):
            return None, None, f"Scaler file not found at: {scaler_path}"
        
        # Load XGBoost model
        model = XGBRegressor()
        model.load_model(model_path)
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        return model, scaler, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Get a sample of user IDs and names
@st.cache_data
def get_user_samples(folder_path, n=10):
    user_path = os.path.join(folder_path, "user.json")
    users = []
    
    if os.path.exists(user_path):
        with open(user_path, 'r') as f:
            count = 0
            for line in f:
                if count >= 1000:  # Read at most 1000 users to keep memory usage low
                    break
                user = json.loads(line)
                users.append({
                    'user_id': user['user_id'],
                    'name': user.get('name', 'Unknown User'),
                    'review_count': user.get('review_count', 0),
                    'average_stars': user.get('average_stars', 0)
                })
                count += 1
    
    # Randomly sample n users
    if len(users) > n:
        return random.sample(users, n)
    return users

# Get a sample of business IDs and names
@st.cache_data
def get_business_samples(folder_path, n=10):
    business_path = os.path.join(folder_path, "business.json")
    businesses = []
    
    if os.path.exists(business_path):
        with open(business_path, 'r') as f:
            count = 0
            for line in f:
                if count >= 1000:  # Read at most 1000 businesses to keep memory usage low
                    break
                business = json.loads(line)
                businesses.append({
                    'business_id': business['business_id'],
                    'name': business.get('name', 'Unknown Business'),
                    'stars': business.get('stars', 0),
                    'city': business.get('city', 'Unknown'),
                    'state': business.get('state', 'Unknown'),
                    'address': business.get('address', 'Unknown'),
                    'categories': business.get('categories', 'Uncategorized'),
                    'review_count': business.get('review_count', 0),
                    'is_open': business.get('is_open', 0)
                })
                count += 1
    
    # Randomly sample n businesses
    if len(businesses) > n:
        return random.sample(businesses, n)
    return businesses

# Get user name from ID
def get_user_name(folder_path, user_id):
    user_path = os.path.join(folder_path, "user.json")
    
    if os.path.exists(user_path):
        with open(user_path, 'r') as f:
            for line in f:
                user = json.loads(line)
                if user['user_id'] == user_id:
                    return user.get('name', 'Unknown User')
    
    return "Unknown User"

# Get business details from ID
def get_business_details(folder_path, business_id):
    business_path = os.path.join(folder_path, "business.json")
    
    if os.path.exists(business_path):
        with open(business_path, 'r') as f:
            for line in f:
                business = json.loads(line)
                if business['business_id'] == business_id:
                    return {
                        'name': business.get('name', 'Unknown Business'),
                        'stars': business.get('stars', 0),
                        'city': business.get('city', 'Unknown'),
                        'state': business.get('state', 'Unknown'),
                        'address': business.get('address', 'Unknown'),
                        'categories': business.get('categories', 'Uncategorized'),
                        'review_count': business.get('review_count', 0),
                        'is_open': business.get('is_open', 0)
                    }
    
    return {
        'name': 'Unknown Business',
        'stars': 0,
        'city': 'Unknown',
        'state': 'Unknown',
        'address': 'Unknown',
        'categories': 'Uncategorized',
        'review_count': 0,
        'is_open': 0
    }

# Process data without using Spark
def process_data_without_spark(user_data, business_data, tip_data):
    # Process user features
    user_features = {}
    if user_data:
        user_id = user_data['user_id']
        user_features[user_id] = {
            'review_count': float(user_data.get('review_count', 0)),
            'average_stars': float(user_data.get('average_stars', 0)),
            'useful': float(user_data.get('useful', 0)),
            'funny': float(user_data.get('funny', 0)),
            'cool': float(user_data.get('cool', 0)),
            'fans': float(user_data.get('fans', 0)),
            'elite': len(user_data.get('elite', '').split(',')) if user_data.get('elite') else 0,
            'average_stars_norm': 0,  # Will be normalized later
            'review_count_norm': 0    # Will be normalized later
        }
    
    # Process business features
    business_features = {}
    if business_data:
        business_id = business_data['business_id']
        business_features[business_id] = {
            'stars': float(business_data.get('stars', 0)),
            'review_count': float(business_data.get('review_count', 0)),
            'is_open': int(business_data.get('is_open', 0)),
            'stars_norm': 0,  # Will be normalized later
            'review_count_norm': 0  # Will be normalized later
        }
    
    # Process tip data
    tip_result = {'user': {}, 'business': {}}
    
    return user_features, business_features, tip_result

# Helper function to get a random avatar image
def get_random_avatar():
    try:
        # Use a placeholder avatar service
        avatar_id = random.randint(1, 70)
        response = requests.get(f"https://i.pravatar.cc/150?img={avatar_id}", timeout=3)
        if response.status_code == 200:
            return BytesIO(response.content)
        return None
    except:
        return None

# Helper function to get a random food image
def get_random_food_image():
    try:
        # List of food-related image URLs
        food_images = [
            "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
            "https://images.unsplash.com/photo-1555939594-58d7cb561ad1",
            "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe",
            "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38",
            "https://images.unsplash.com/photo-1546069901-ba9599a7e63c",
            "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445",
            "https://images.unsplash.com/photo-1414235077428-338989a2e8c0",
            "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4",
            "https://images.unsplash.com/photo-1552566626-52f8b828add9",
            "https://images.unsplash.com/photo-1544025162-d76694265947"
        ]
        
        # Select a random image
        img_url = random.choice(food_images)
        response = requests.get(f"{img_url}?w=600&h=400&fit=crop", timeout=3)
        if response.status_code == 200:
            return BytesIO(response.content)
        return None
    except:
        return None

# Helper function to display star ratings
def display_stars(rating):
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    stars_html = ""
    # Full stars
    for _ in range(full_stars):
        stars_html += '<span class="star-rating">★</span>'
    
    # Half star
    if half_star:
        stars_html += '<span class="star-rating">★</span>'
    
    # Empty stars
    for _ in range(empty_stars):
        stars_html += '<span class="empty-star">★</span>'
    
    return stars_html

# Load or compute global statistics
@st.cache_data
def get_global_stats(folder_path):
    stats_path = os.path.join(folder_path, "global_stats.json")
    
    # If stats are already computed, load them
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    
    # Otherwise compute them (this is expensive but only done once)
    total_reviews = 0
    avg_user_stars = 0
    max_user_reviews = 0
    user_count = 0
    
    avg_business_stars = 0
    max_business_reviews = 0
    business_count = 0
    
    # Process users
    user_path = os.path.join(folder_path, "user.json")
    if os.path.exists(user_path):
        with open(user_path, 'r') as f:
            for line in f:
                user = json.loads(line)
                review_count = float(user['review_count'])
                total_reviews += review_count
                avg_user_stars += float(user['average_stars'])
                max_user_reviews = max(max_user_reviews, review_count)
                user_count += 1
    
    # Process businesses
    business_path = os.path.join(folder_path, "business.json")
    if os.path.exists(business_path):
        with open(business_path, 'r') as f:
            for line in f:
                business = json.loads(line)
                avg_business_stars += float(business['stars'])
                max_business_reviews = max(max_business_reviews, float(business['review_count']))
                business_count += 1
    
    # Calculate averages
    avg_user_stars = avg_user_stars / user_count if user_count > 0 else 0
    avg_business_stars = avg_business_stars / business_count if business_count > 0 else 0
    
    # Create stats dictionary
    stats = {
        'total_reviews': total_reviews,
        'avg_user_stars': avg_user_stars,
        'max_user_reviews': max_user_reviews,
        'avg_business_stars': avg_business_stars,
        'max_business_reviews': max_business_reviews
    }
    
    # Save stats for future use
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    
    return stats

# Optimized data loading function
@st.cache_data
def load_specific_data(folder_path, user_id, business_id):
    try:
        user_data = None
        business_data = None
        user_tips = []
        business_tips = []
        
        # Load only the specific user
        user_path = os.path.join(folder_path, "user.json")
        if os.path.exists(user_path):
            with open(user_path, 'r') as f:
                for line in f:
                    user = json.loads(line)
                    if user['user_id'] == user_id:
                        user_data = user
                        break
        
        # Load only the specific business
        business_path = os.path.join(folder_path, "business.json")
        if os.path.exists(business_path):
            with open(business_path, 'r') as f:
                for line in f:
                    business = json.loads(line)
                    if business['business_id'] == business_id:
                        business_data = business
                        break
        
        # Load only relevant tips
        tip_path = os.path.join(folder_path, "tip.json")
        if os.path.exists(tip_path):
            with open(tip_path, 'r') as f:
                for line in f:
                    tip = json.loads(line)
                    if tip['user_id'] == user_id or tip['business_id'] == business_id:
                        if tip['user_id'] == user_id:
                            user_tips.append(tip)
                        if tip['business_id'] == business_id:
                            business_tips.append(tip)
        
        return user_data, business_data, user_tips, business_tips
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, [], []

# Get actual user and business IDs from your dataset
@st.cache_data
def get_valid_example_pairs(folder_path, num_pairs=3):
    user_path = os.path.join(folder_path, "user.json")
    business_path = os.path.join(folder_path, "business.json")
    
    valid_users = []
    valid_businesses = []
    
    # Get some valid users
    if os.path.exists(user_path):
        with open(user_path, 'r') as f:
            count = 0
            for line in f:
                if count >= 10:  # Just read a few users
                    break
                user = json.loads(line)
                valid_users.append({
                    'user_id': user['user_id'],
                    'name': user.get('name', 'User ' + str(count+1))
                })
                count += 1
    
    # Get some valid businesses
    if os.path.exists(business_path):
        with open(business_path, 'r') as f:
            count = 0
            for line in f:
                if count >= 10:  # Just read a few businesses
                    break
                business = json.loads(line)
                valid_businesses.append({
                    'business_id': business['business_id'],
                    'name': business.get('name', 'Business ' + str(count+1))
                })
                count += 1
    
    # Create valid pairs
    valid_pairs = []
    for i in range(min(num_pairs, len(valid_users), len(valid_businesses))):
        valid_pairs.append({
            'user_name': valid_users[i]['name'],
            'user_id': valid_users[i]['user_id'],
            'business_name': valid_businesses[i]['name'],
            'business_id': valid_businesses[i]['business_id']
        })
    
    return valid_pairs

# Load sample reviews for a business
@st.cache_data
def get_sample_reviews_for_business(business_id):
    # Generate some fake reviews that are specific to the business ID
    # This ensures different businesses get different reviews
    random.seed(hash(business_id) % 10000)  # Use business_id to seed the random generator
    
    review_templates = [
        "The food was amazing! I especially loved the {0} and the service was {1}.",
        "Decent place but a bit {0} for what you get. The {1} is nice though.",
        "Great spot for a {0}. The {1} are creative and delicious.",
        "The wait was {0} and the food was {1}. Might give it another try though."
    ]
    
    adjectives = ["outstanding", "excellent", "top-notch", "superb", "mediocre", "disappointing", "terrible"]
    items = ["appetizers", "main course", "desserts", "drinks", "ambiance", "decor", "staff", "cocktails"]
    experiences = ["date night", "family dinner", "business lunch", "casual meal", "special occasion"]
    wait_times = ["too long", "reasonable", "very short", "acceptable"]
    food_qualities = ["just okay", "delicious", "fantastic", "underwhelming", "incredible"]
    
    review_texts = []
    for template in review_templates:
        if "{0}" in template and "{1}" in template:
            if "wait was" in template:
                review_texts.append(template.format(random.choice(wait_times), random.choice(food_qualities)))
            elif "spot for a" in template:
                review_texts.append(template.format(random.choice(experiences), random.choice(items)))
            elif "a bit" in template:
                review_texts.append(template.format(random.choice(adjectives), random.choice(items)))
            else:
                review_texts.append(template.format(random.choice(items), random.choice(adjectives)))
    
    # Generate ratings based on the business_id to ensure consistency
    base_rating = (hash(business_id) % 20 + 15) / 10  # Will be between 1.5 and 3.5
    review_ratings = [
        min(5.0, max(1.0, base_rating + random.uniform(-0.5, 1.5))) for _ in range(4)
    ]
    
    # Generate dates
    current_year = datetime.now().year
    review_dates = [
        f"{current_year-1}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(4)
    ]
    
    # Generate reviewer names
    first_names = ["John", "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Jennifer"]
    last_initials = ["A.", "B.", "C.", "D.", "M.", "L.", "R.", "S.", "T.", "W."]
    review_names = [f"{random.choice(first_names)} {random.choice(last_initials)}" for _ in range(4)]
    
    return review_texts, review_ratings, review_dates, review_names

# Get top recommended businesses for a user based on predictions
@st.cache_data
def get_top_recommendations_for_user(user_id, data_folder, model_path, scaler_path, num_recommendations=6):
    # Load model and scaler
    model, scaler, error = load_model_and_scaler(model_path, scaler_path)
    
    if not model or not scaler:
        return []
    
    # Get global statistics
    global_stats = get_global_stats(data_folder)
    
    # Load user data
    user_data = None
    user_path = os.path.join(data_folder, "user.json")
    if os.path.exists(user_path):
        with open(user_path, 'r') as f:
            for line in f:
                user = json.loads(line)
                if user['user_id'] == user_id:
                    user_data = user
                    break
    
    if not user_data:
        return []
    
    # Get a sample of businesses to predict
    sample_businesses = get_business_samples(data_folder, 20)  # Get more businesses to choose from
    
    # Process user features
    user_features = {
        user_id: process_user_features(
            user_data, 
            (global_stats['total_reviews'], global_stats['avg_user_stars'], global_stats['max_user_reviews'])
        )
    }
    
    # Process all businesses and make predictions
    predictions = []
    
    for business in sample_businesses:
        business_id = business['business_id']
        business_data = None
        
        # Load business data
        business_path = os.path.join(data_folder, "business.json")
        if os.path.exists(business_path):
            with open(business_path, 'r') as f:
                for line in f:
                    b = json.loads(line)
                    if b['business_id'] == business_id:
                        business_data = b
                        break
        
        if business_data:
            # Process business features
            business_features = {
                business_id: process_business_features(
                    business_data,
                    (global_stats['avg_business_stars'], global_stats['max_business_reviews'])
                )
            }
            
            # Create empty tip result
            tip_result = {'business': {business_id: {'count': 0, 'total_likes': 0, 'avg_length': 0}}, 
                         'user': {user_id: {'count': 0, 'total_likes': 0, 'avg_length': 0}}}
            
            # Create feature vector
            entry = [user_id, business_id, None]  # No rating
            features = create_feature_vector(entry, user_features, business_features, tip_result, False)
            
            # Scale features
            scaled_features = scaler.transform([features])
            
            # Predict
            prediction = model.predict(scaled_features)[0]
            final_prediction = np.clip(prediction, 1.0, 5.0)
            
            # Add to predictions
            predictions.append({
                'business_id': business_id,
                'name': business['name'],
                'categories': business['categories'],
                'stars': business['stars'],
                'city': business['city'],
                'state': business['state'],
                'predicted_rating': final_prediction
            })
    
    # Sort by predicted rating (highest first)
    predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    # Return top recommendations
    return predictions[:num_recommendations]

# Make prediction function
def make_prediction(user_id, business_id, data_folder, model_path, scaler_path):
    # Load model and scaler
    model, scaler, error = load_model_and_scaler(model_path, scaler_path)
    
    if error:
        st.error(error)
        return
    
    if not model or not scaler:
        st.error("Failed to load model or scaler")
        return
    
    # Get global statistics
    global_stats = get_global_stats(data_folder)
    
    # Load user and business data
    user_data, business_data, user_tips, business_tips = load_specific_data(data_folder, user_id, business_id)
    
    if not user_data:
        st.error(f"User ID {user_id} not found in the dataset")
        return
    
    if not business_data:
        st.error(f"Business ID {business_id} not found in the dataset")
        return
    
    # Process user features
    user_features = {
        user_id: process_user_features(
            user_data, 
            (global_stats['total_reviews'], global_stats['avg_user_stars'], global_stats['max_user_reviews'])
        )
    }
    
    # Process business features
    business_features = {
        business_id: process_business_features(
            business_data,
            (global_stats['avg_business_stars'], global_stats['max_business_reviews'])
        )
    }
    
    # Create a simple tip_result structure instead of calling process_tips
    # This avoids the error with the map function
    tip_result = {
        'user': {
            user_id: {
                'count': len(user_tips),
                'total_likes': sum(tip.get('likes', 0) for tip in user_tips),
                'avg_length': sum(len(tip.get('text', '')) for tip in user_tips) / max(1, len(user_tips))
            }
        },
        'business': {
            business_id: {
                'count': len(business_tips),
                'total_likes': sum(tip.get('likes', 0) for tip in business_tips),
                'avg_length': sum(len(tip.get('text', '')) for tip in business_tips) / max(1, len(business_tips))
            }
        }
    }
    
    # Create feature vector
    entry = [user_id, business_id, None]  # No rating
    features = create_feature_vector(entry, user_features, business_features, tip_result, False)
    
    # Scale features
    scaled_features = scaler.transform([features])
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    final_prediction = np.clip(prediction, 1.0, 5.0)
    
    # Display business info
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Business Information</div>', unsafe_allow_html=True)
    
    # Get a random food image based on business ID for consistency
    random.seed(hash(business_id) % 10000)
    food_img = get_random_food_image()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if food_img:
            st.image(food_img, width=200)
    
    with col2:
        st.markdown(f"### {business_data.get('name', 'Unknown Business')}")
        st.markdown(f"**Categories:** {business_data.get('categories', 'Uncategorized')}")
        st.markdown(f"**Current Rating:** {display_stars(float(business_data.get('stars', 0)))}", unsafe_allow_html=True)
        st.markdown(f"**Location:** {business_data.get('city', 'Unknown')}, {business_data.get('state', 'Unknown')}")
        st.markdown(f"**Address:** {business_data.get('address', 'Unknown')}")
        st.markdown(f"**Reviews:** {business_data.get('review_count', 0)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display user info
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">User Information</div>', unsafe_allow_html=True)
    
    # Get a random avatar based on user ID for consistency
    random.seed(hash(user_id) % 10000)
    avatar = get_random_avatar()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if avatar:
            st.image(avatar, width=150)
    
    with col2:
        st.markdown(f"### {user_data.get('name', 'Unknown User')}")
        st.markdown(f"**Average Rating:** {display_stars(float(user_data.get('average_stars', 0)))}", unsafe_allow_html=True)
        st.markdown(f"**Reviews:** {user_data.get('review_count', 0)}")
        st.markdown(f"**Useful Votes:** {user_data.get('useful', 0)}")
        st.markdown(f"**Funny Votes:** {user_data.get('funny', 0)}")
        st.markdown(f"**Cool Votes:** {user_data.get('cool', 0)}")
        st.markdown(f"**Fans:** {user_data.get('fans', 0)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display prediction
    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">Predicted Rating</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-value">{final_prediction:.1f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;">{display_stars(final_prediction)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return final_prediction

def main():
    # Custom header
    st.markdown("""
    <div class="header-container">
        <div class="header-logo">🍽️</div>
        <div class="header-title">Yelp Rating Predictor</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Paths to model and data files
    data_folder = "./data/sample"
    model_base_path = "./data/result/model"
    model_path = model_base_path  # XGBoost model file
    scaler_path = f"{model_base_path}_scaler.pkl"  # Scaler file
    
    # Get sample data
    sample_users = get_user_samples(data_folder, 15)
    sample_businesses = get_business_samples(data_folder, 15)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Search by ID", "✨ Discover Restaurants"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Enter User and Business IDs</div>', unsafe_allow_html=True)
        
        # Initialize session state for user and business IDs if they don't exist
        if 'manual_user_id' not in st.session_state:
            st.session_state.manual_user_id = ""
        if 'manual_business_id' not in st.session_state:
            st.session_state.manual_business_id = ""
        
        # Input fields using session state
        user_id = st.text_input("User ID", value=st.session_state.manual_user_id, key="user_id_input")
        business_id = st.text_input("Business ID", value=st.session_state.manual_business_id, key="business_id_input")
        
        # Function to update session state
        def update_ids(user, business):
            st.session_state.manual_user_id = user
            st.session_state.manual_business_id = business
            return user, business
        
        # Add example IDs section
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 15px; margin-bottom: 15px;">
            <h4 style="margin-top: 0;">Try these example pairs:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Get example pairs that definitely exist in your dataset
        example_pairs = get_valid_example_pairs(data_folder, 3)
        
        # Create columns for example pairs
        example_cols = st.columns(3)
        
        for i, pair in enumerate(example_pairs):
            with example_cols[i]:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; height: 150px;">
                    <p style="font-weight: bold; margin-bottom: 5px;">{pair['user_name']} → {pair['business_name']}</p>
                    <p style="font-size: 0.8rem; color: #666; margin-bottom: 5px;">User ID: {pair['user_id'][:8]}...</p>
                    <p style="font-size: 0.8rem; color: #666; margin-bottom: 15px;">Business ID: {pair['business_id'][:8]}...</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Use This Pair", key=f"example_pair_{i}"):
                    # Update session state directly
                    st.session_state.manual_user_id = pair['user_id']
                    st.session_state.manual_business_id = pair['business_id']
                    # Use st.rerun() instead of experimental_rerun()
                    st.rerun()
        
        # Add a "Try a Random Pair" button
        st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("🎲 Try a Random Pair", key="random_pair"):
            # Use the predefined pairs for reliability
            random_pair = random.choice(example_pairs)
            # Update session state directly
            st.session_state.manual_user_id = random_pair['user_id']
            st.session_state.manual_business_id = random_pair['business_id']
            # Use st.rerun() instead of experimental_rerun()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add a note about the IDs
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666; margin-top: 10px;">
            <p>Note: These are pre-selected pairs from the dataset that are known to work with the model.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Predict Rating", key="manual_predict"):
            if not user_id or not business_id:
                st.warning("Please enter both user ID and business ID")
            else:
                make_prediction(user_id, business_id, data_folder, model_path, scaler_path)
                
                # Get sample reviews specific to this business
                review_texts, review_ratings, review_dates, review_names = get_sample_reviews_for_business(business_id)
                
                # Display the reviews
                st.markdown('<h3>Sample Reviews for This Restaurant</h3>', unsafe_allow_html=True)
                
                for i in range(min(4, len(review_texts))):
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #eee;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <div style="margin-right: 0.5rem;">{display_stars(review_ratings[i])}</div>
                            <div style="color: #666; font-size: 0.9rem;">{review_dates[i]}</div>
                        </div>
                        <div style="margin-bottom: 0.5rem;">{review_texts[i]}</div>
                        <div style="color: #666; font-size: 0.9rem;">- {review_names[i]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Discover Your Next Favorite Restaurant</div>', unsafe_allow_html=True)
        
        # Display sample users
        st.subheader("Select a User")
        
        # Create a grid of users
        user_cols = st.columns(3)
        selected_user = None
        selected_user_id = None
        
        # Store user avatars in session state to keep them consistent
        if 'user_avatars' not in st.session_state:
            st.session_state.user_avatars = {}
        
        for i, user in enumerate(sample_users[:6]):  # Show first 6 users
            user_id = user['user_id']
            
            # Get or create avatar for this user
            if user_id not in st.session_state.user_avatars:
                avatar = get_random_avatar()
                if avatar:
                    st.session_state.user_avatars[user_id] = avatar
            
            with user_cols[i % 3]:
                # Display the consistent avatar
                if user_id in st.session_state.user_avatars:
                    st.image(st.session_state.user_avatars[user_id], width=100)
                
                st.markdown(f"**{user['name']}**")
                st.markdown(f"{user['review_count']} reviews")
                if st.button(f"Select {user['name']}", key=f"user_{i}"):
                    selected_user = user['name']
                    selected_user_id = user['user_id']
        
        # Display recommended businesses if a user is selected
        if selected_user_id:
            st.markdown(f"### Top Recommended Restaurants for {selected_user}")
            
            # Get personalized recommendations for this user
            recommended_businesses = get_top_recommendations_for_user(
                selected_user_id, data_folder, model_path, scaler_path
            )
            
            if recommended_businesses:
                # Create a grid of businesses
                business_cols = st.columns(2)
                
                for i, business in enumerate(recommended_businesses):
                    with business_cols[i % 2]:
                        st.markdown('<div class="business-card">', unsafe_allow_html=True)
                        
                        # Get a random food image based on business ID for consistency
                        random.seed(hash(business['business_id']) % 10000)
                        food_img = get_random_food_image()
                        if food_img:
                            st.image(food_img, width=200)
                        
                        st.markdown(f"### {business['name']}")
                        st.markdown(f"**Categories:** {business['categories']}")
                        st.markdown(f"**Current Rating:** {display_stars(float(business['stars']))}", unsafe_allow_html=True)
                        st.markdown(f"**Predicted Rating:** {display_stars(float(business['predicted_rating']))}", unsafe_allow_html=True)
                        st.markdown(f"**Location:** {business['city']}, {business['state']}")
                        
                        if st.button("View Details", key=f"predict_{i}"):
                            make_prediction(selected_user_id, business['business_id'], data_folder, model_path, scaler_path)
                        
                            # Get sample reviews specific to this business
                            review_texts, review_ratings, review_dates, review_names = get_sample_reviews_for_business(business['business_id'])
                            
                            # Display the reviews
                            st.markdown('<h3>Sample Reviews for This Restaurant</h3>', unsafe_allow_html=True)
                            
                            for j in range(min(4, len(review_texts))):
                                st.markdown(f"""
                                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #eee;">
                                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                        <div style="margin-right: 0.5rem;">{display_stars(review_ratings[j])}</div>
                                        <div style="color: #666; font-size: 0.9rem;">{review_dates[j]}</div>
                                    </div>
                                    <div style="margin-bottom: 0.5rem;">{review_texts[j]}</div>
                                    <div style="color: #666; font-size: 0.9rem;">- {review_names[j]}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No recommendations available for this user")
        else:
            st.info("Select a user to see restaurant recommendations")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2023 Yelp Rating Predictor | Powered by Machine Learning</p>
        <p>This is a demonstration app and not affiliated with Yelp Inc.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



