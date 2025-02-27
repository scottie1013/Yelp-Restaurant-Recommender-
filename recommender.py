import os
import findspark

# Set SPARK_HOME to the PySpark installation directory
spark_home = os.path.join('/opt/anaconda3/envs/pyspark_env/lib/python3.11/site-packages/pyspark')
os.environ['SPARK_HOME'] = spark_home
findspark.init()

import csv
import json
import sys
import time
import numpy as np
import os

# Import pyspark modules
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib
import argparse

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

def process_tips(tip_rdd):
    """Process tip data"""
    if not tip_rdd:
        return {'business': {}, 'user': {}}
        
    business_tips = (tip_rdd
        .map(lambda x: (
            x['business_id'],
            (1, x.get('likes', 0), len(x.get('text', '')))
        ))
        .reduceByKey(lambda a, b: (
            a[0] + b[0], a[1] + b[1], a[2] + b[2]
        ))
        .mapValues(lambda v: {
            'count': v[0],
            'total_likes': v[1],
            'avg_length': v[2] / v[0] if v[0] > 0 else 0
        })
        .collectAsMap())
        
    user_tips = (tip_rdd
        .map(lambda x: (
            x['user_id'],
            (1, x.get('likes', 0), len(x.get('text', '')))
        ))
        .reduceByKey(lambda a, b: (
            a[0] + b[0], a[1] + b[1], a[2] + b[2]
        ))
        .mapValues(lambda v: {
            'count': v[0],
            'total_likes': v[1],
            'avg_length': v[2] / v[0] if v[0] > 0 else 0
        })
        .collectAsMap())
        
    return {'business': business_tips, 'user': user_tips}

def create_feature_vector(entry, user_features, business_features, tip_data, include_rating=True):
    """Create feature vector for a single entry"""
    user_id, business_id = entry[:2]
    rating = float(entry[2]) if include_rating else None
    
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
        user_tips['count'],
        user_tips['total_likes'],
        user_tips['avg_length'],
        business_tips['count'],
        business_tips['total_likes'],
        business_tips['avg_length']
    ])
    
    # Interaction features
    features.extend([
        user.get('average_stars', 0) * business.get('stars', 0),
        user.get('review_count', 0) * business.get('review_count', 0),
        user.get('engagement_score', 0) * business.get('review_density', 0),
        user.get('star_diff', 0) * business.get('star_diff', 0)
    ])
    
    return (features, rating) if include_rating else features

def load_and_process_data(sc, folder_path):
    """Load and process all data sources"""
    # Read data
    train_rdd = sc.textFile(f"{folder_path}/yelp_train.csv")
    header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda x: x != header).map(lambda x: x.split(','))
    
    user_rdd = sc.textFile(f"{folder_path}/user.json").map(json.loads)
    business_rdd = sc.textFile(f"{folder_path}/business.json").map(json.loads)
    
    try:
        tip_rdd = sc.textFile(f"{folder_path}/tip.json").map(json.loads)
    except:
        tip_rdd = None
    
    # Calculate global statistics
    user_stats = user_rdd.map(lambda x: (
        float(x['review_count']),
        float(x['average_stars'])
    )).cache()
    
    total_reviews = user_stats.map(lambda x: x[0]).sum()
    avg_user_stars = user_stats.map(lambda x: x[1]).mean()
    max_user_reviews = user_stats.map(lambda x: x[0]).max()
    
    business_stats = business_rdd.map(lambda x: (
        float(x['stars']),
        float(x['review_count'])
    )).cache()
    
    avg_business_stars = business_stats.map(lambda x: x[0]).mean()
    max_business_reviews = business_stats.map(lambda x: x[1]).max()
    
    # Broadcast global statistics
    user_globals = sc.broadcast((total_reviews, avg_user_stars, max_user_reviews))
    business_globals = sc.broadcast((avg_business_stars, max_business_reviews))
    
    # Process features
    user_features = user_rdd.map(
        lambda x: (x['user_id'], process_user_features(x, user_globals.value))
    ).collectAsMap()
    
    business_features = business_rdd.map(
        lambda x: (x['business_id'], process_business_features(x, business_globals.value))
    ).collectAsMap()
    
    tip_data = process_tips(tip_rdd)
    
    return train_rdd, user_features, business_features, tip_data

def train_and_predict(train_rdd, test_rdd, user_features, business_features, tip_data, model_path=None):
    """Train model and generate predictions"""
    # Prepare training data
    train_features = train_rdd.map(
        lambda x: create_feature_vector(x, user_features, business_features, tip_data, True)
    ).collect()
    
    X_train = np.array([f[0] for f in train_features], dtype=np.float32)
    y_train = np.array([f[1] for f in train_features], dtype=np.float32)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Train model
    model = XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.004,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds = 50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Save model if path is provided
    if model_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        joblib.dump(scaler, f"{model_path}_scaler.pkl")
        st.write(f"Model saved to {model_path}")
    
    # Generate predictions
    test_features = np.array([
        create_feature_vector(x, user_features, business_features, tip_data, False)
        for x in test_rdd.collect()
    ], dtype=np.float32)
    
    test_features = scaler.transform(test_features)
    predictions = model.predict(test_features)
    
    return np.clip(predictions, 1.0, 5.0)

def predict_single_rating(user_id, business_id, user_features, business_features, tip_data, model_path, scaler_path):
    """Predict rating for a single user-business pair using pre-trained model"""
    # Load model and scaler
    model = XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Create feature vector for this pair
    entry = [user_id, business_id, None]  # No rating
    features = create_feature_vector(entry, user_features, business_features, tip_data, False)
    
    # Scale features
    scaled_features = scaler.transform([features])
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    
    return np.clip(prediction, 1.0, 5.0)

def run_recommender(folder_path, test_path, output_path, model_path=None):
    """Main execution function"""
    # Create a SparkSession
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("YelpRecommender") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    try:
        start_time = time.time()
        st.write("Starting recommendation system...")
        
        # Load and process data
        st.write("Processing data...")
        train_rdd, user_features, business_features, tip_data = load_and_process_data(sc, folder_path)
        
        # Load test data
        test_rdd = sc.textFile(test_path)
        header = test_rdd.first()
        test_rdd = test_rdd.filter(lambda x: x != header).map(lambda x: x.split(','))
        
        # Train model and generate predictions
        st.write("Training model and generating predictions...")
        predictions = train_and_predict(train_rdd, test_rdd, user_features, business_features, tip_data, model_path)
        
        # Save results
        st.write("Saving predictions...")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'business_id', 'prediction'])
            for (user_id, business_id, _), pred in zip(test_rdd.collect(), predictions):
                writer.writerow([user_id, business_id, pred])
        
        st.write(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
    finally:
        spark.stop()  # Stop the SparkSession instead of the SparkContext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yelp Recommendation System')
    parser.add_argument('folder_path', help='Path to the folder containing data files')
    parser.add_argument('test_path', help='Path to the test CSV file')
    parser.add_argument('output_path', help='Path to save the output predictions')
    parser.add_argument('--save_model', help='Path to save the trained model', default=None)
    
    args = parser.parse_args()
    
    run_recommender(args.folder_path, args.test_path, args.output_path, args.save_model)  