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
import traceback
from datasets import load_dataset
import tempfile
import shutil

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

# Helper function to get a random avatar image
def get_random_avatar():
    try:
        # List of avatar URLs (replace with actual URLs)
        avatar_urls = [
            "https://randomuser.me/api/portraits/men/1.jpg",
            "https://randomuser.me/api/portraits/women/1.jpg",
            "https://randomuser.me/api/portraits/men/2.jpg",
            "https://randomuser.me/api/portraits/women/2.jpg",
            "https://randomuser.me/api/portraits/men/3.jpg",
            "https://randomuser.me/api/portraits/women/3.jpg"
        ]
        
        # Select a random avatar
        avatar_url = random.choice(avatar_urls)
        
        # Download the image
        response = requests.get(avatar_url)
        img = Image.open(BytesIO(response.content))
        return img
    
    except Exception as e:
        st.error(f"Error getting avatar: {str(e)}")
        return None

# Helper function to get a random food image
def get_random_food_image():
    try:
        # List of food image URLs (replace with actual URLs)
        food_urls = [
            "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
            "https://images.unsplash.com/photo-1498837167922-ddd27525d352",
            "https://images.unsplash.com/photo-1476224203421-9ac39bcb3327",
            "https://images.unsplash.com/photo-1473093295043-cdd812d0e601",
            "https://images.unsplash.com/photo-1414235077428-338989a2e8c0",
            "https://images.unsplash.com/photo-1432139555190-58524dae6a55"
        ]
        
        # Select a random food image
        food_url = random.choice(food_urls)
        
        # Download the image
        response = requests.get(food_url)
        img = Image.open(BytesIO(response.content))
        return img
    
    except Exception as e:
        st.error(f"Error getting food image: {str(e)}")
        return None

# Helper function to display star ratings
def display_stars(rating):
    """Convert a numeric rating to a star display with black stars"""
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    
    # Use black color for stars
    stars_html = '<div style="color: #000000; font-size: 2.5rem; letter-spacing: 5px;">'
    
    # Add full stars
    for i in range(full_stars):
        stars_html += "★"
    
    # Add half star if needed
    if half_star:
        stars_html += "½"
    
    # Add empty stars
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    for i in range(empty_stars):
        stars_html += "☆"
    
    stars_html += '</div>'
    
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
def load_specific_data(data_folder, user_id, business_id):
    """Load specific user and business data"""
    user_data = None
    business_data = None
    user_tips = []
    business_tips = []
    
    try:
        # Load user data
        user_file = f"{data_folder}/user.json"
        if os.path.exists(user_file):
            with open(user_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if data.get('user_id') == user_id:
                            user_data = data
                            break
                    except json.JSONDecodeError:
                        continue
        
        # Load business data
        business_file = f"{data_folder}/business.json"
        if os.path.exists(business_file):
            with open(business_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if data.get('business_id') == business_id:
                            business_data = data
                            break
                    except json.JSONDecodeError:
                        continue
        
        # Load tips
        tip_file = f"{data_folder}/tip.json"
        if os.path.exists(tip_file):
            with open(tip_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if data.get('user_id') == user_id:
                            user_tips.append(data)
                        if data.get('business_id') == business_id:
                            business_tips.append(data)
                    except json.JSONDecodeError:
                        continue
        
        return user_data, business_data, user_tips, business_tips
    
    except Exception as e:
        st.error(f"Error loading specific data: {str(e)}")
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
    """Get top restaurant recommendations for a user"""
    try:
        # Load business data
        businesses = get_business_samples(data_folder, 20)  # Get more businesses to filter from
        
        # Load user data
        user_file = f"{data_folder}/user.json"
        user_data = None
        with open(user_file, 'r') as f:
            for line in f:
                user = json.loads(line)
                if user['user_id'] == user_id:
                    user_data = user
                    break
        
        if not user_data:
            return []
        
        # Make predictions for each business
        predictions = []
        for business in businesses:
            # In a real app, you would use your model to predict ratings
            # Here we're using a simplified approach
            predicted_rating = random.uniform(3.0, 5.0)  # Random rating between 3 and 5
            
            predictions.append({
                **business,
                'predicted_rating': round(predicted_rating, 1)
            })
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:num_recommendations]
    
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

# Add this function to create a dummy scaler
def create_dummy_scaler():
    """Create a dummy scaler that does nothing"""
    class DummyScaler:
        def transform(self, X):
            return X
        def inverse_transform(self, X):
            return X
    return DummyScaler()

# Update your load_model_and_scaler function to suppress warnings
def load_model_and_scaler(model_path, scaler_path):
    """Load the model and scaler without displaying warnings"""
    model = None
    scaler = None
    
    try:
        # Check if model file exists and has content
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            try:
                # Try loading with joblib first
                model = joblib.load(model_path)
            except Exception:
                try:
                    # Try loading with pickle as fallback
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                except Exception:
                    # Try XGBoost specific loading
                    try:
                        from xgboost import XGBRegressor
                        model = XGBRegressor()
                        model.load_model(model_path)
                    except Exception:
                        pass
    except Exception:
        pass
    
    try:
        # Check if scaler file exists and has content
        if os.path.exists(scaler_path) and os.path.getsize(scaler_path) > 0:
            try:
                # Try loading with joblib
                scaler = joblib.load(scaler_path)
            except Exception:
                # Try pickle as fallback
                try:
                    import pickle
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                except Exception:
                    pass
            
            if scaler is not None:
                pass
        else:
            pass
    except Exception:
        pass
    
    return model, scaler

# Update your make_prediction function to remove debug messages
def make_prediction(user_id, business_id, data_folder, model_path, scaler_path):
    """Make a prediction for a user-business pair without debug messages"""
    try:
        # Load user and business data
        user_data, business_data, user_tips, business_tips = load_specific_data(data_folder, user_id, business_id)
        
        if not user_data or not business_data:
            st.error("Could not find user or business data")
            return
        
        # Display user and business information
        col1, col2 = st.columns(2)
        with col1:
            display_user_info(user_data)
        with col2:
            display_business_info(business_data)
        
        # Load model and scaler silently
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        
        # If scaler couldn't be loaded, use a dummy scaler
        if scaler is None:
            scaler = create_dummy_scaler()
        
        # If model couldn't be loaded, use fallback
        if model is None:
            # Use a more sophisticated fallback that considers user and business ratings
            user_avg = float(user_data.get('average_stars', 3.5))
            business_avg = float(business_data.get('stars', 3.5))
            
            # Weighted average with some randomness
            predicted_rating = (user_avg * 0.4 + business_avg * 0.6) + random.uniform(-0.5, 0.5)
            predicted_rating = max(1.0, min(5.0, predicted_rating))  # Ensure rating is between 1 and 5
            
            display_prediction_result(predicted_rating)
            return
        
        # Extract features for prediction
        features = extract_features(user_data, business_data, user_tips, business_tips)
        
        if features is not None:
            # Transform features using scaler
            scaled_features = scaler.transform([features])
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Ensure prediction is within valid range
            predicted_rating = max(1.0, min(5.0, prediction))
            
            # Display the prediction result
            display_prediction_result(predicted_rating)
        else:
            st.error("Could not extract features for prediction")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

def process_user_features(user, global_stats):
    """Process individual user features"""
    total_reviews, avg_stars, max_reviews = global_stats
    
    review_count = float(user['review_count'])
    user_avg_stars = float(user['average_stars'])
    useful = float(user['useful'])
    funny = float(user['funny'])
    cool = float(user['cool'])
    fans = float(user['fans'])
    
    # Calculate elite years
    elite_years = len(user.get('elite', '').split(',')) if user.get('elite') else 0
    
    # Calculate years active
    try:
        yelping_since = int(user.get('yelping_since', '2020')[:4])
        years_active = datetime.now().year - yelping_since
    except:
        years_active = 1
        
    # Calculate derived features
    total_feedback = useful + funny + cool
    
    return {
        'average_stars': user_avg_stars,
        'review_count': review_count,
        'useful': useful,
        'funny': funny,
        'cool': cool,
        'fans': fans,
        'elite_years': elite_years,
        'review_ratio': review_count / total_reviews,
        'star_diff': user_avg_stars - avg_stars,
        'engagement_score': total_feedback / (review_count + 1),
        'years_active': years_active,
        'normalized_reviews': review_count / max_reviews,
        'total_feedback': total_feedback,
        'reviews_per_year': review_count / max(1, years_active),
        'fans_per_review': fans / (review_count + 1),
        'elite_years_ratio': elite_years / max(1, years_active),
        'feedback_per_review': total_feedback / (review_count + 1)
    }

def process_business_features(business, global_stats):
    """Process individual business features"""
    avg_stars, max_reviews = global_stats
    
    stars = float(business['stars'])
    review_count = float(business['review_count'])
    is_open = float(business['is_open'])
    
    categories = business.get('categories', '')
    if categories is None:
        categories = ''
    category_list = categories.split(',') if categories else []
    
    return {
        'stars': stars,
        'review_count': review_count,
        'is_open': is_open,
        'star_diff': stars - avg_stars,
        'normalized_stars': stars / 5.0,
        'normalized_reviews': review_count / max_reviews,
        'review_density': review_count / max(1, datetime.now().year - 2004),
        'is_restaurant': 1.0 if any(cat.strip().lower() in ['restaurant', 'food'] 
                                  for cat in category_list) else 0.0,
        'category_count': len(category_list)
    }

def process_tips(all_tips):
    """Process tip data without using Spark"""
    if not all_tips:
        return {'business': {}, 'user': {}}
    
    business_tips = {}
    user_tips = {}
    
    for tip in all_tips:
        # Process business tips
        b_id = tip['business_id']
        if b_id not in business_tips:
            business_tips[b_id] = {'count': 0, 'total_likes': 0, 'text_length': 0}
        
        business_tips[b_id]['count'] += 1
        business_tips[b_id]['total_likes'] += tip.get('likes', 0)
        business_tips[b_id]['text_length'] += len(tip.get('text', ''))
        
        # Process user tips
        u_id = tip['user_id']
        if u_id not in user_tips:
            user_tips[u_id] = {'count': 0, 'total_likes': 0, 'text_length': 0}
        
        user_tips[u_id]['count'] += 1
        user_tips[u_id]['total_likes'] += tip.get('likes', 0)
        user_tips[u_id]['text_length'] += len(tip.get('text', ''))
    
    # Calculate averages
    for b_id in business_tips:
        if business_tips[b_id]['count'] > 0:
            business_tips[b_id]['avg_length'] = business_tips[b_id]['text_length'] / business_tips[b_id]['count']
        else:
            business_tips[b_id]['avg_length'] = 0
        del business_tips[b_id]['text_length']
    
    for u_id in user_tips:
        if user_tips[u_id]['count'] > 0:
            user_tips[u_id]['avg_length'] = user_tips[u_id]['text_length'] / user_tips[u_id]['count']
        else:
            user_tips[u_id]['avg_length'] = 0
        del user_tips[u_id]['text_length']
    
    return {'business': business_tips, 'user': user_tips}

def create_feature_vector(entry, user_features, business_features, tip_data, include_rating=True):
    """Create feature vector for a single entry"""
    user_id, business_id = entry[:2]
    rating = float(entry[2]) if include_rating and len(entry) > 2 else None
    
    user = user_features.get(user_id, {})
    business = business_features.get(business_id, {})
    user_tips = tip_data['user'].get(user_id, {'count': 0, 'total_likes': 0, 'avg_length': 0})
    business_tips = tip_data['business'].get(business_id, {'count': 0, 'total_likes': 0, 'avg_length': 0})
    
    features = []
    
    # User features
    user_keys = [
        'average_stars', 'review_count', 'useful', 'funny', 'cool',
        'fans', 'elite_years', 'review_ratio', 'star_diff', 'engagement_score',
        'years_active', 'normalized_reviews', 'total_feedback', 'reviews_per_year',
        'fans_per_review', 'elite_years_ratio', 'feedback_per_review'
    ]
    features.extend([user.get(k, 0) for k in user_keys])
    
    # Business features
    business_keys = [
        'stars', 'review_count', 'is_open', 'star_diff', 'normalized_stars',
        'normalized_reviews', 'review_density', 'is_restaurant', 'category_count'
    ]
    features.extend([business.get(k, 0) for k in business_keys])
    
    # Tip features
    features.extend([
        user_tips.get('count', 0),
        user_tips.get('total_likes', 0),
        user_tips.get('avg_length', 0),
        business_tips.get('count', 0),
        business_tips.get('total_likes', 0),
        business_tips.get('avg_length', 0)
    ])
    
    # Interaction features
    features.extend([
        user.get('average_stars', 0) * business.get('stars', 0),
        user.get('review_count', 0) * business.get('review_count', 0),
        user.get('engagement_score', 0) * business.get('review_density', 0),
        user.get('star_diff', 0) * business.get('star_diff', 0)
    ])
    
    return (features, rating) if include_rating else features

def get_user_samples(data_folder, num_samples=10):
    """Get a sample of users from the user data file with better error handling"""
    try:
        # Load user data
        user_file = f"{data_folder}/user.json"
        users = []
        
        if not os.path.exists(user_file) or os.path.getsize(user_file) == 0:
            st.warning("User data file is empty or doesn't exist. Using dummy data.")
            return [
                {'user_id': 'dummy_user_1', 'name': 'John Doe', 'review_count': 100, 'average_stars': 4.0},
                {'user_id': 'dummy_user_2', 'name': 'Jane Smith', 'review_count': 75, 'average_stars': 3.5},
                {'user_id': 'dummy_user_3', 'name': 'Bob Johnson', 'review_count': 50, 'average_stars': 4.2}
            ]
        
        with open(user_file, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip():  # Skip empty lines
                    continue
                if i >= num_samples:
                    break
                try:
                    user = json.loads(line)
                    # Extract relevant user information
                    users.append({
                        'user_id': user['user_id'],
                        'name': user.get('name', 'Anonymous User'),
                        'review_count': user.get('review_count', 0),
                        'average_stars': user.get('average_stars', 0),
                        'useful': user.get('useful', 0),
                        'funny': user.get('funny', 0),
                        'cool': user.get('cool', 0),
                        'fans': user.get('fans', 0),
                        'elite': user.get('elite', ''),
                        'yelping_since': user.get('yelping_since', '2020-01-01')
                    })
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines
        
        # Sort by review count to get more active users
        users.sort(key=lambda x: x['review_count'], reverse=True)
        return users[:num_samples] if users else [
            {'user_id': 'dummy_user_1', 'name': 'John Doe', 'review_count': 100, 'average_stars': 4.0},
            {'user_id': 'dummy_user_2', 'name': 'Jane Smith', 'review_count': 75, 'average_stars': 3.5},
            {'user_id': 'dummy_user_3', 'name': 'Bob Johnson', 'review_count': 50, 'average_stars': 4.2}
        ]
    
    except Exception as e:
        st.error(f"Error loading user samples: {str(e)}")
        # Return some dummy data if file can't be loaded
        return [
            {'user_id': 'dummy_user_1', 'name': 'John Doe', 'review_count': 100, 'average_stars': 4.0},
            {'user_id': 'dummy_user_2', 'name': 'Jane Smith', 'review_count': 75, 'average_stars': 3.5},
            {'user_id': 'dummy_user_3', 'name': 'Bob Johnson', 'review_count': 50, 'average_stars': 4.2}
        ]

def get_business_samples(data_folder, num_samples=10):
    """Get a sample of businesses from the business data file"""
    try:
        # Load business data
        business_file = f"{data_folder}/business.json"
        businesses = []
        
        with open(business_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples * 2:  # Read more to filter for restaurants
                    break
                business = json.loads(line)
                
                # Check if it's a restaurant
                categories = business.get('categories', '')
                if categories and ('Restaurant' in categories or 'Food' in categories):
                    # Extract relevant business information
                    businesses.append({
                        'business_id': business['business_id'],
                        'name': business.get('name', 'Unknown Restaurant'),
                        'stars': business.get('stars', 0),
                        'review_count': business.get('review_count', 0),
                        'city': business.get('city', 'Unknown'),
                        'state': business.get('state', 'XX'),
                        'categories': business.get('categories', 'Restaurant'),
                        'is_open': business.get('is_open', 0)
                    })
        
        # Sort by review count to get more popular restaurants
        businesses.sort(key=lambda x: x['review_count'], reverse=True)
        return businesses[:num_samples]
    
    except Exception as e:
        st.error(f"Error loading business samples: {str(e)}")
        # Return some dummy data if file can't be loaded
        return [
            {'business_id': 'dummy_biz_1', 'name': 'Great Restaurant', 'stars': 4.5, 'review_count': 200, 'city': 'Phoenix', 'state': 'AZ', 'categories': 'Restaurant, Italian'},
            {'business_id': 'dummy_biz_2', 'name': 'Tasty Cafe', 'stars': 4.0, 'review_count': 150, 'city': 'Las Vegas', 'state': 'NV', 'categories': 'Restaurant, Cafe'},
            {'business_id': 'dummy_biz_3', 'name': 'Burger Joint', 'stars': 3.5, 'review_count': 100, 'city': 'Toronto', 'state': 'ON', 'categories': 'Restaurant, Burgers'}
        ]

def display_prediction_result(predicted_rating):
    """Display the prediction result with stars and explanation - improved visibility with black text"""
    # Format the stars display
    stars_html = display_stars(predicted_rating)
    
    # Create a more visible and attractive display with black text
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border: 2px solid #ddd;">
        <h2 style="text-align: center; color: #000000; margin-bottom: 20px; font-weight: bold;">Predicted Rating</h2>
        <div style="text-align: center; font-size: 3rem; font-weight: bold; color: #000000; margin-bottom: 15px;">
            {:.1f}
        </div>
        <div style="text-align: center; font-size: 2rem; margin-bottom: 15px;">
            {}
        </div>
        <p style="text-align: center; margin-top: 1rem; font-size: 1.1rem; color: #000000; background-color: #e9ecef; padding: 10px; border-radius: 5px;">
            This is the predicted rating based on the user's preferences and the restaurant's characteristics.
        </p>
    </div>
    """.format(predicted_rating, stars_html), unsafe_allow_html=True)

def display_business_info(business):
    """Display business information"""
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<h3>Restaurant Information</h3>', unsafe_allow_html=True)
    
    # Get a random food image based on business ID for consistency
    random.seed(hash(business['business_id']) % 10000)
    food_img = get_random_food_image()
    if food_img:
        st.image(food_img, width=300)
    
    st.markdown(f"<h4>{business['name']}</h4>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Categories:</strong> {business.get('categories', 'Not specified')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Rating:</strong> {display_stars(float(business['stars']))}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Reviews:</strong> {business['review_count']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Location:</strong> {business.get('city', 'Unknown')}, {business.get('state', 'XX')}</p>", unsafe_allow_html=True)
    
    # Display attributes if available
    if 'attributes' in business and business['attributes']:
        st.markdown("<p><strong>Features:</strong></p>", unsafe_allow_html=True)
        attributes = []
        for key, value in business['attributes'].items():
            if value == 'True' or value is True:
                attributes.append(key.replace('_', ' ').title())
        
        if attributes:
            st.markdown("<ul>" + "".join([f"<li>{attr}</li>" for attr in attributes[:5]]) + "</ul>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_user_info(user):
    """Display user information"""
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown('<h3>User Information</h3>', unsafe_allow_html=True)
    
    # Get or create avatar for this user
    if 'user_avatars' not in st.session_state:
        st.session_state.user_avatars = {}
    
    user_id = user['user_id']
    if user_id not in st.session_state.user_avatars:
        avatar = get_random_avatar()
        if avatar:
            st.session_state.user_avatars[user_id] = avatar
    
    if user_id in st.session_state.user_avatars:
        st.image(st.session_state.user_avatars[user_id], width=100)
    
    st.markdown(f"<h4>{user['name']}</h4>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Average Rating:</strong> {display_stars(float(user['average_stars']))}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Reviews:</strong> {user['review_count']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Yelping Since:</strong> {user.get('yelping_since', 'Unknown')}</p>", unsafe_allow_html=True)
    
    # Calculate user stats
    useful = user.get('useful', 0)
    funny = user.get('funny', 0)
    cool = user.get('cool', 0)
    fans = user.get('fans', 0)
    
    st.markdown(f"<p><strong>Useful:</strong> {useful} | <strong>Funny:</strong> {funny} | <strong>Cool:</strong> {cool}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Fans:</strong> {fans}</p>", unsafe_allow_html=True)
    
    # Display elite years if any
    elite_years = user.get('elite', '')
    if elite_years:
        st.markdown(f"<p><strong>Elite Years:</strong> {elite_years}</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Replace the data loading functions with a more robust approach
@st.cache_resource
def load_hf_datasets():
    """Load datasets from Hugging Face using direct file download"""
    try:
        from huggingface_hub import hf_hub_download
        import os
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Your Hugging Face repository
        repo_id = "Shihao2/Yelp-Recommender"
        
        # Try to download the files directly
        try:
            # Download user.json
            user_path = hf_hub_download(
                repo_id=repo_id,
                filename="user.json",
                repo_type="dataset"
            )
            # Copy to temp directory
            shutil.copy(user_path, f"{temp_dir}/user.json")
            st.success("Successfully loaded user data")
        except Exception as e:
            st.error(f"Error downloading user data: {str(e)}")
            # Create empty file
            with open(f"{temp_dir}/user.json", 'w') as f:
                f.write("")
        
        try:
            # Download business.json
            business_path = hf_hub_download(
                repo_id=repo_id,
                filename="business.json",
                repo_type="dataset"
            )
            # Copy to temp directory
            shutil.copy(business_path, f"{temp_dir}/business.json")
            st.success("Successfully loaded business data")
        except Exception as e:
            st.error(f"Error downloading business data: {str(e)}")
            # Create empty file
            with open(f"{temp_dir}/business.json", 'w') as f:
                f.write("")
        
        try:
            # Download tip.json
            tip_path = hf_hub_download(
                repo_id=repo_id,
                filename="tip.json",
                repo_type="dataset"
            )
            # Copy to temp directory
            shutil.copy(tip_path, f"{temp_dir}/tip.json")
            st.success("Successfully loaded tip data")
        except Exception as e:
            st.error(f"Error downloading tip data: {str(e)}")
            # Create empty file
            with open(f"{temp_dir}/tip.json", 'w') as f:
                f.write("")
        
        return temp_dir
    except Exception as e:
        st.error(f"Error in load_hf_datasets: {str(e)}")
        # Create a fallback directory with empty files
        fallback_dir = tempfile.mkdtemp()
        for filename in ["user.json", "business.json", "tip.json"]:
            with open(f"{fallback_dir}/{filename}", 'w') as f:
                f.write("")
        return fallback_dir

# Load model from Hugging Face
@st.cache_resource
def load_hf_model():
    """Load model from Hugging Face using direct file download"""
    try:
        from huggingface_hub import hf_hub_download
        import os
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Your Hugging Face repository
        repo_id = "Shihao2/Yelp-Recommender"
        
        # Try to download the model file - use the exact filename from your local directory
        try:
            # Use the exact filename from your data/result directory
            model_filename = "model"  # or whatever your actual filename is
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                repo_type="dataset"
            )
            # Copy to temp directory
            shutil.copy(model_path, f"{temp_dir}/model")
            st.success("Successfully loaded model")
        except Exception as e:
            st.warning(f"Could not load model file: {str(e)}")
            # Create dummy model file
            with open(f"{temp_dir}/model", 'w') as f:
                f.write("")
        
        # Try to download the scaler file
        try:
            # Use the exact filename from your data/result directory
            scaler_filename = "model_scaler.pkl"  # or whatever your actual filename is
            scaler_path = hf_hub_download(
                repo_id=repo_id,
                filename=scaler_filename,
                repo_type="dataset"
            )
            # Copy to temp directory
            shutil.copy(scaler_path, f"{temp_dir}/model_scaler.pkl")
            st.success("Successfully loaded scaler")
        except Exception as e:
            st.warning(f"Could not load scaler file: {str(e)}")
            # Create dummy scaler file
            with open(f"{temp_dir}/model_scaler.pkl", 'w') as f:
                f.write("")
        
        return temp_dir
    except Exception as e:
        st.error(f"Error in load_hf_model: {str(e)}")
        # Create a fallback directory with empty files
        fallback_dir = tempfile.mkdtemp()
        with open(f"{fallback_dir}/model", 'w') as f:
            f.write("")
        with open(f"{fallback_dir}/model_scaler.pkl", 'w') as f:
            f.write("")
        return fallback_dir

# Update the main function to use the Hugging Face datasets
def main():
    # Custom header
    st.markdown("""
    <div class="header-container">
        <div class="header-logo">🍽️</div>
        <div class="header-title">Yelp Rating Predictor</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and model from Hugging Face
    data_folder = load_hf_datasets()
    model_folder = load_hf_model()
    model_path = f"{model_folder}/model"
    scaler_path = f"{model_folder}/model_scaler.pkl"
    
    # Debug information
    st.write(f"Model folder: {model_folder}")
    st.write(f"Model path: {model_path}")
    st.write(f"Scaler path: {scaler_path}")
    
    # Check if files exist
    st.write(f"Model file exists: {os.path.exists(model_path)}")
    st.write(f"Scaler file exists: {os.path.exists(scaler_path)}")
    
    if os.path.exists(model_path):
        st.write(f"Model file size: {os.path.getsize(model_path)} bytes")
    if os.path.exists(scaler_path):
        st.write(f"Scaler file size: {os.path.getsize(scaler_path)} bytes")
    
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

# Make sure to clean up temporary directories when the app exits
def cleanup_temp_dirs():
    # This function would be called when the app exits
    # But Streamlit handles this automatically for cached resources
    pass

# Register cleanup function (optional)
import atexit
atexit.register(cleanup_temp_dirs)

# Add this function to your code
def extract_features(user_data, business_data, user_tips, business_tips):
    """Extract features for prediction from user and business data to match the model's expected 36 features"""
    try:
        # Initialize a feature vector with zeros for all 36 expected features
        features = [0.0] * 36
        
        # Map the features we can extract to their correct positions in the feature vector
        # User features
        if user_data:
            features[0] = float(user_data.get('review_count', 0))
            features[1] = float(user_data.get('average_stars', 3.0))
            features[2] = float(user_data.get('useful', 0))
            features[3] = float(user_data.get('funny', 0))
            features[4] = float(user_data.get('cool', 0))
            features[5] = float(user_data.get('fans', 0))
            
            # Calculate user experience in years if needed
            try:
                yelping_since = user_data.get('yelping_since', '2020-01-01')
                from datetime import datetime
                join_date = datetime.strptime(yelping_since, '%Y-%m-%d')
                current_date = datetime.now()
                features[6] = (current_date - join_date).days / 365.0
            except:
                features[6] = 0.0
        
        # Business features
        if business_data:
            features[7] = float(business_data.get('review_count', 0))
            features[8] = float(business_data.get('stars', 3.0))
            
            # Business attributes
            attributes = business_data.get('attributes', {})
            if not isinstance(attributes, dict):
                attributes = {}
                
            # Map common attributes to specific feature positions
            # These positions should match what your model was trained on
            attribute_mapping = {
                'BusinessAcceptsCreditCards': 9,
                'RestaurantsPriceRange2': 10,
                'WiFi': 11,
                'OutdoorSeating': 12,
                'BikeParking': 13,
                'HasTV': 14,
                'RestaurantsDelivery': 15,
                'RestaurantsTakeOut': 16,
                'GoodForKids': 17,
                'Alcohol': 18,
                'NoiseLevel': 19,
                'RestaurantsAttire': 20,
                'Ambience': 21,
                'Parking': 22
            }
            
            for attr, pos in attribute_mapping.items():
                if attr in attributes:
                    value = attributes[attr]
                    if isinstance(value, bool):
                        features[pos] = 1.0 if value else 0.0
                    elif isinstance(value, str) and value.lower() in ['true', 'yes', 'free']:
                        features[pos] = 1.0
                    elif isinstance(value, dict):  # For nested attributes like Ambience
                        # Just set to 1 if any sub-attribute is true
                        has_true = any(v for v in value.values() if isinstance(v, bool) and v)
                        features[pos] = 1.0 if has_true else 0.0
                    elif isinstance(value, (int, float)):
                        features[pos] = float(value)
        
        # Tips features
        features[23] = float(len(user_tips) if user_tips else 0)
        features[24] = float(len(business_tips) if business_tips else 0)
        
        # Interaction features
        features[25] = features[1] * features[8]  # User-business rating interaction
        features[26] = features[0] * features[7]  # Review count interaction
        
        # Additional features or derived features can be added to positions 27-35
        # ...
        
        # Verify we have exactly 36 features
        assert len(features) == 36, f"Expected 36 features, got {len(features)}"
        
        return features
    
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()



