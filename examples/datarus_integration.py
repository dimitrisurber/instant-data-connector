#!/usr/bin/env python3
"""
ML Integration Demo - PostgreSQL FDW-based ML Pipeline

This example demonstrates comprehensive integration with the ML platform,
showcasing how the new FDW-based architecture enables:
- Efficient feature extraction and preparation
- Real-time ML pipeline integration
- Advanced analytics and model training workflows
- Production-ready ML data pipelines
"""

import asyncio
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import InstantDataConnector
from instant_connector import SecureCredentialManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLMLPipeline:
    """
    Simulated ML pipeline integration class.
    
    In a real implementation, this would connect to the actual ML platform.
    This demo shows the integration patterns and data flows.
    """
    
    def __init__(self, connector: InstantDataConnector):
        self.connector = connector
        self.models = {}
        self.features = {}
        
    async def extract_user_features(self, cutoff_date: str = None) -> pd.DataFrame:
        """Extract comprehensive user features for ML models."""
        if not cutoff_date:
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Complex feature extraction query
        features_sql = """
        SELECT 
            u.user_id,
            u.username,
            u.registration_date,
            u.last_login,
            u.is_active,
            
            -- User tenure features
            EXTRACT(days FROM CURRENT_DATE - u.registration_date) as days_since_registration,
            CASE 
                WHEN u.last_login IS NOT NULL THEN EXTRACT(days FROM CURRENT_DATE - u.last_login::date)
                ELSE -1 
            END as days_since_last_login,
            
            -- Order behavior features
            COUNT(o.order_id) as total_orders,
            SUM(o.total_amount) as total_spent,
            AVG(o.total_amount) as avg_order_value,
            STDDEV(o.total_amount) as order_value_std,
            MAX(o.total_amount) as max_order_value,
            MIN(o.total_amount) as min_order_value,
            
            -- Temporal features
            COUNT(CASE WHEN o.order_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as orders_last_30_days,
            COUNT(CASE WHEN o.order_date >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as orders_last_90_days,
            
            -- Status distribution
            COUNT(CASE WHEN o.status = 'completed' THEN 1 END) as completed_orders,
            COUNT(CASE WHEN o.status = 'cancelled' THEN 1 END) as cancelled_orders,
            
            -- Recency features  
            MAX(o.order_date) as last_order_date,
            CASE 
                WHEN MAX(o.order_date) IS NOT NULL THEN 
                    EXTRACT(days FROM CURRENT_DATE - MAX(o.order_date)::date)
                ELSE -1 
            END as days_since_last_order,
            
            -- Frequency features
            CASE 
                WHEN COUNT(o.order_id) > 0 AND u.registration_date IS NOT NULL THEN
                    COUNT(o.order_id)::float / GREATEST(EXTRACT(days FROM CURRENT_DATE - u.registration_date), 1)
                ELSE 0 
            END as order_frequency_per_day
            
        FROM users u
        LEFT JOIN orders o ON u.user_id = o.customer_id
        WHERE u.registration_date >= $1
        GROUP BY u.user_id, u.username, u.registration_date, u.last_login, u.is_active
        ORDER BY u.user_id
        """
        
        features_df = await self.connector.execute_query(
            features_sql,
            params=[cutoff_date],
            cache_key=f'user_features_{cutoff_date}',
            cache_ttl=3600
        )
        
        return features_df
    
    async def extract_product_features(self) -> pd.DataFrame:
        """Extract product features for recommendation models."""
        product_sql = """
        SELECT 
            p.product_id,
            p.sku,
            p.name,
            p.category,
            p.price,
            p.cost,
            p.price - p.cost as margin,
            (p.price - p.cost) / p.price as margin_ratio,
            p.weight,
            p.stock_quantity,
            p.is_active,
            
            -- Sales performance
            COUNT(oi.order_item_id) as times_ordered,
            SUM(oi.quantity) as total_quantity_sold,
            SUM(oi.total_price) as total_revenue,
            AVG(oi.quantity) as avg_quantity_per_order,
            AVG(oi.unit_price) as avg_selling_price,
            
            -- Product popularity
            COUNT(DISTINCT oi.order_id) as unique_orders,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            
            -- Recent performance
            COUNT(CASE WHEN o.order_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as orders_last_30_days,
            SUM(CASE WHEN o.order_date >= CURRENT_DATE - INTERVAL '30 days' THEN oi.total_price ELSE 0 END) as revenue_last_30_days
            
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        LEFT JOIN orders o ON oi.order_id = o.order_id
        GROUP BY p.product_id, p.sku, p.name, p.category, p.price, p.cost, p.weight, p.stock_quantity, p.is_active
        ORDER BY total_revenue DESC NULLS LAST
        """
        
        return await self.connector.execute_query(
            product_sql,
            cache_key='product_features',
            cache_ttl=1800
        )
    
    async def build_user_product_matrix(self) -> pd.DataFrame:
        """Build user-product interaction matrix for collaborative filtering."""
        matrix_sql = """
        SELECT 
            o.customer_id as user_id,
            oi.product_id,
            SUM(oi.quantity) as total_quantity,
            SUM(oi.total_price) as total_spent,
            COUNT(oi.order_item_id) as interaction_count,
            MAX(o.order_date) as last_interaction_date,
            AVG(oi.unit_price) as avg_price_paid
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.status IN ('completed', 'shipped')
        GROUP BY o.customer_id, oi.product_id
        ORDER BY total_spent DESC
        """
        
        return await self.connector.execute_query(
            matrix_sql,
            cache_key='user_product_matrix',
            cache_ttl=3600
        )
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        df = df.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(0)
        
        # Create derived features
        if 'total_spent' in df.columns and 'total_orders' in df.columns:
            df['spending_efficiency'] = df['total_spent'] / (df['total_orders'] + 1)
            
        if 'days_since_registration' in df.columns and 'total_orders' in df.columns:
            df['engagement_score'] = df['total_orders'] / (df['days_since_registration'] / 30 + 1)
            
        if 'completed_orders' in df.columns and 'total_orders' in df.columns:
            df['completion_rate'] = df['completed_orders'] / (df['total_orders'] + 1)
            
        # Create categorical features
        if 'days_since_last_login' in df.columns:
            df['login_recency_category'] = pd.cut(
                df['days_since_last_login'], 
                bins=[-1, 0, 7, 30, 90, float('inf')], 
                labels=['never', 'recent', 'week', 'month', 'inactive']
            )
            
        return df
    
    def simulate_model_training(self, features_df: pd.DataFrame, model_type: str) -> Dict:
        """Simulate ML model training with ML patterns."""
        print(f"🤖 Simulating {model_type} model training...")
        
        # Simulate training metrics
        training_metrics = {
            'model_type': model_type,
            'training_samples': len(features_df),
            'feature_count': len(features_df.columns),
            'training_time': np.random.uniform(5, 15),
            'accuracy': np.random.uniform(0.82, 0.95),
            'precision': np.random.uniform(0.80, 0.92),
            'recall': np.random.uniform(0.78, 0.90),
            'f1_score': np.random.uniform(0.79, 0.91),
            'auc_roc': np.random.uniform(0.85, 0.96)
        }
        
        self.models[model_type] = {
            'metrics': training_metrics,
            'features': list(features_df.columns),
            'trained_at': datetime.now().isoformat()
        }
        
        return training_metrics
    
    def simulate_predictions(self, model_type: str, data: pd.DataFrame) -> pd.DataFrame:
        """Simulate model predictions."""
        predictions = data.copy()
        
        if model_type == 'churn_prediction':
            predictions['churn_probability'] = np.random.beta(2, 8, len(data))
            predictions['churn_risk'] = pd.cut(
                predictions['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['low', 'medium', 'high']
            )
            
        elif model_type == 'customer_ltv':
            predictions['predicted_ltv'] = np.random.lognormal(
                mean=np.log(500), sigma=0.8, size=len(data)
            ).round(2)
            predictions['ltv_percentile'] = pd.qcut(
                predictions['predicted_ltv'],
                q=5,
                labels=['bottom_20', 'low_20', 'mid_20', 'high_20', 'top_20']
            )
            
        elif model_type == 'product_recommendation':
            # Simulate top-N product recommendations
            all_products = list(range(1, 11))  # Product IDs 1-10
            predictions['recommended_products'] = [
                np.random.choice(all_products, size=3, replace=False).tolist()
                for _ in range(len(data))
            ]
            predictions['recommendation_scores'] = [
                np.random.uniform(0.6, 0.95, 3).round(3).tolist()
                for _ in range(len(data))
            ]
        
        return predictions


async def example_1_feature_extraction_pipeline():
    """Example 1: Comprehensive feature extraction for ML."""
    print("\n" + "="*70)
    print("🔍 Example 1: Feature Extraction for ML Pipeline")
    print("="*70)
    
    connector = InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        },
        enable_caching=True
    )
    
    try:
        await connector.setup_fdw_infrastructure()
        
        # Initialize ML pipeline
        ml_pipeline = MLMLPipeline(connector)
        
        # Extract user features
        print("📊 Extracting user behavioral features...")
        user_features = await ml_pipeline.extract_user_features('2024-01-01')
        
        if not user_features.empty:
            print(f"✅ Extracted features for {len(user_features)} users")
            print("Sample features:")
            print(user_features[['user_id', 'days_since_registration', 'total_orders', 
                               'total_spent', 'avg_order_value', 'order_frequency_per_day']].head().to_string(index=False))
            
            # Feature engineering
            print("\n🔧 Applying feature engineering...")
            engineered_features = ml_pipeline.engineer_features(user_features)
            
            print(f"✅ Engineered {len(engineered_features.columns)} features")
            if 'engagement_score' in engineered_features.columns:
                print("New derived features:")
                print(engineered_features[['user_id', 'engagement_score', 'spending_efficiency', 
                                         'completion_rate']].head().to_string(index=False))
        
        # Extract product features
        print("\n🛍️  Extracting product performance features...")
        product_features = await ml_pipeline.extract_product_features()
        
        if not product_features.empty:
            print(f"✅ Extracted features for {len(product_features)} products")
            print("Top performing products:")
            print(product_features[['name', 'category', 'times_ordered', 'total_revenue', 
                                  'unique_customers']].head().to_string(index=False))
        
        # Build interaction matrix
        print("\n🤝 Building user-product interaction matrix...")
        interaction_matrix = await ml_pipeline.build_user_product_matrix()
        
        if not interaction_matrix.empty:
            print(f"✅ Built interaction matrix with {len(interaction_matrix)} interactions")
            print("Sample interactions:")
            print(interaction_matrix[['user_id', 'product_id', 'total_quantity', 
                                    'total_spent', 'interaction_count']].head().to_string(index=False))
        
    finally:
        await connector.close()


async def example_2_churn_prediction_model():
    """Example 2: Churn prediction model training and inference."""
    print("\n" + "="*70)
    print("🎯 Example 2: Churn Prediction Model with ML")
    print("="*70)
    
    connector = InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    )
    
    try:
        await connector.setup_fdw_infrastructure()
        ml_pipeline = MLMLPipeline(connector)
        
        # Extract features for churn prediction
        print("📊 Preparing churn prediction dataset...")
        user_features = await ml_pipeline.extract_user_features('2024-01-01')
        
        if not user_features.empty:
            # Feature engineering for churn
            churn_features = ml_pipeline.engineer_features(user_features)
            
            # Select relevant features for churn prediction
            churn_model_features = churn_features[[
                'user_id', 'days_since_registration', 'days_since_last_login',
                'total_orders', 'total_spent', 'avg_order_value', 
                'orders_last_30_days', 'orders_last_90_days',
                'completion_rate', 'engagement_score', 'is_active'
            ]].copy()
            
            print(f"✅ Prepared {len(churn_model_features)} samples for training")
            print("Feature summary:")
            print(churn_model_features.describe().round(2))
            
            # Simulate model training
            print("\n🤖 Training churn prediction model...")
            training_metrics = ml_pipeline.simulate_model_training(
                churn_model_features, 'churn_prediction'
            )
            
            print("✅ Model training completed!")
            print(f"   Training samples: {training_metrics['training_samples']}")
            print(f"   Feature count: {training_metrics['feature_count']}")
            print(f"   Training time: {training_metrics['training_time']:.1f} seconds")
            print(f"   Model accuracy: {training_metrics['accuracy']:.3f}")
            print(f"   AUC-ROC: {training_metrics['auc_roc']:.3f}")
            
            # Generate predictions
            print("\n🔮 Generating churn predictions...")
            predictions = ml_pipeline.simulate_predictions('churn_prediction', churn_model_features)
            
            # Show high-risk customers
            high_risk_customers = predictions[predictions['churn_risk'] == 'high'].sort_values(
                'churn_probability', ascending=False
            )
            
            if not high_risk_customers.empty:
                print(f"⚠️  Found {len(high_risk_customers)} high-risk customers:")
                print(high_risk_customers[['user_id', 'churn_probability', 'total_orders', 
                                         'days_since_last_login', 'orders_last_30_days']].head().to_string(index=False))
                
                # Actionable insights
                print("\n💡 Actionable insights for retention campaigns:")
                avg_risk_score = predictions['churn_probability'].mean()
                high_risk_count = len(predictions[predictions['churn_risk'] == 'high'])
                
                print(f"   Average churn risk: {avg_risk_score:.3f}")
                print(f"   High-risk customers: {high_risk_count} ({high_risk_count/len(predictions)*100:.1f}%)")
                print("   Recommended actions:")
                print("   - Send personalized re-engagement emails to high-risk customers")
                print("   - Offer targeted discounts to inactive users")
                print("   - Implement win-back campaigns for customers with no recent orders")
        
    finally:
        await connector.close()


async def example_3_customer_ltv_prediction():
    """Example 3: Customer Lifetime Value prediction model."""
    print("\n" + "="*70)
    print("💰 Example 3: Customer Lifetime Value Prediction")
    print("="*70)
    
    connector = InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    )
    
    try:
        await connector.setup_fdw_infrastructure()
        ml_pipeline = MLMLPipeline(connector)
        
        # Extract features for LTV prediction
        print("📊 Preparing customer LTV dataset...")
        user_features = await ml_pipeline.extract_user_features('2024-01-01')
        
        if not user_features.empty:
            ltv_features = ml_pipeline.engineer_features(user_features)
            
            # Filter to customers with purchase history
            active_customers = ltv_features[ltv_features['total_orders'] > 0].copy()
            
            print(f"✅ Prepared {len(active_customers)} active customers for LTV modeling")
            
            # Show current LTV distribution
            print("\n📈 Current customer value distribution:")
            ltv_stats = active_customers['total_spent'].describe()
            print(f"   Average total spent: ${ltv_stats['mean']:.2f}")
            print(f"   Median total spent: ${ltv_stats['50%']:.2f}")
            print(f"   Top 10% threshold: ${ltv_stats['90%']:.2f}")
            
            # Train LTV model
            print("\n🤖 Training customer LTV prediction model...")
            training_metrics = ml_pipeline.simulate_model_training(
                active_customers, 'customer_ltv'
            )
            
            print("✅ LTV model training completed!")
            print(f"   Model accuracy: {training_metrics['accuracy']:.3f}")
            
            # Generate LTV predictions
            print("\n🔮 Generating LTV predictions...")
            ltv_predictions = ml_pipeline.simulate_predictions('customer_ltv', active_customers)
            
            # Show high-value customer segments
            high_ltv_customers = ltv_predictions[
                ltv_predictions['ltv_percentile'] == 'top_20'
            ].sort_values('predicted_ltv', ascending=False)
            
            if not high_ltv_customers.empty:
                print(f"💎 Top 20% predicted LTV customers ({len(high_ltv_customers)} customers):")
                print(high_ltv_customers[['user_id', 'predicted_ltv', 'total_spent', 
                                        'total_orders', 'avg_order_value']].head().to_string(index=False))
                
                # Business insights
                print("\n💡 Customer segmentation insights:")
                avg_predicted_ltv = ltv_predictions['predicted_ltv'].mean()
                top_20_avg = high_ltv_customers['predicted_ltv'].mean()
                
                print(f"   Average predicted LTV: ${avg_predicted_ltv:.2f}")
                print(f"   Top 20% average LTV: ${top_20_avg:.2f}")
                print(f"   Value multiplier: {top_20_avg/avg_predicted_ltv:.1f}x")
                print("   Recommended strategies:")
                print("   - VIP treatment and exclusive offers for top 20% customers")
                print("   - Personalized product recommendations for high-LTV segments")
                print("   - Premium customer service tier for valuable customers")
        
    finally:
        await connector.close()


async def example_4_product_recommendation_engine():
    """Example 4: Product recommendation engine using collaborative filtering."""
    print("\n" + "="*70)
    print("🛒 Example 4: Product Recommendation Engine")
    print("="*70)
    
    connector = InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    )
    
    try:
        await connector.setup_fdw_infrastructure()
        ml_pipeline = MLMLPipeline(connector)
        
        # Build recommendation dataset
        print("🤝 Building collaborative filtering dataset...")
        interaction_matrix = await ml_pipeline.build_user_product_matrix()
        product_features = await ml_pipeline.extract_product_features()
        
        if not interaction_matrix.empty and not product_features.empty:
            print(f"✅ Built recommendation dataset:")
            print(f"   User-product interactions: {len(interaction_matrix)}")
            print(f"   Unique users: {interaction_matrix['user_id'].nunique()}")
            print(f"   Unique products: {interaction_matrix['product_id'].nunique()}")
            
            # Show popular products
            popular_products = interaction_matrix.groupby('product_id').agg({
                'user_id': 'nunique',
                'total_quantity': 'sum',
                'total_spent': 'sum'
            }).rename(columns={'user_id': 'unique_buyers'}).sort_values('total_spent', ascending=False)
            
            # Join with product details
            popular_with_details = popular_products.merge(
                product_features[['product_id', 'name', 'category']], 
                on='product_id'
            )
            
            print("\n🔥 Most popular products:")
            print(popular_with_details[['name', 'category', 'unique_buyers', 
                                      'total_quantity', 'total_spent']].head().to_string(index=False))
            
            # Train recommendation model
            print("\n🤖 Training collaborative filtering model...")
            training_metrics = ml_pipeline.simulate_model_training(
                interaction_matrix, 'product_recommendation'
            )
            
            print("✅ Recommendation model training completed!")
            
            # Generate recommendations for sample users
            print("\n🎯 Generating personalized recommendations...")
            sample_users = interaction_matrix['user_id'].unique()[:10]
            user_sample = pd.DataFrame({'user_id': sample_users})
            
            recommendations = ml_pipeline.simulate_predictions('product_recommendation', user_sample)
            
            print(f"📋 Sample recommendations for {len(recommendations)} users:")
            for _, row in recommendations.head().iterrows():
                user_id = row['user_id']
                rec_products = row['recommended_products']
                rec_scores = row['recommendation_scores']
                
                print(f"   User {user_id}:")
                for prod_id, score in zip(rec_products, rec_scores):
                    product_name = product_features[
                        product_features['product_id'] == prod_id
                    ]['name'].iloc[0] if len(product_features[product_features['product_id'] == prod_id]) > 0 else f"Product {prod_id}"
                    print(f"     - {product_name} (score: {score:.3f})")
            
            # Cross-sell analysis
            print("\n🔄 Cross-sell opportunity analysis...")
            cross_sell_sql = """
            SELECT 
                p1.name as product_1,
                p2.name as product_2,
                COUNT(*) as co_purchase_count,
                AVG(o1.total_amount + o2.total_amount) as avg_combined_value
            FROM order_items oi1
            JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
            JOIN products p1 ON oi1.product_id = p1.product_id
            JOIN products p2 ON oi2.product_id = p2.product_id
            JOIN orders o1 ON oi1.order_id = o1.order_id
            JOIN orders o2 ON oi2.order_id = o2.order_id
            WHERE o1.status = 'completed' AND o2.status = 'completed'
            GROUP BY p1.product_id, p1.name, p2.product_id, p2.name
            HAVING COUNT(*) >= 2
            ORDER BY co_purchase_count DESC, avg_combined_value DESC
            LIMIT 10
            """
            
            cross_sell_opportunities = await connector.execute_query(cross_sell_sql)
            
            if not cross_sell_opportunities.empty:
                print("🎯 Top cross-sell opportunities:")
                print(cross_sell_opportunities.to_string(index=False))
        
    finally:
        await connector.close()


async def example_5_real_time_ml_serving():
    """Example 5: Real-time ML model serving and monitoring."""
    print("\n" + "="*70)
    print("⚡ Example 5: Real-time ML Model Serving")
    print("="*70)
    
    connector = InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    )
    
    try:
        await connector.setup_fdw_infrastructure()
        ml_pipeline = MLMLPipeline(connector)
        
        # Simulate real-time feature serving
        print("🚀 Simulating real-time ML feature serving...")
        
        # Fast user lookup for real-time features
        realtime_feature_sql = """
        SELECT 
            u.user_id,
            u.is_active,
            EXTRACT(days FROM CURRENT_DATE - u.registration_date) as days_since_registration,
            COALESCE(recent_stats.order_count_30d, 0) as recent_orders,
            COALESCE(recent_stats.total_spent_30d, 0) as recent_spending,
            COALESCE(lifetime_stats.total_orders, 0) as lifetime_orders,
            COALESCE(lifetime_stats.total_spent, 0) as lifetime_spending
        FROM users u
        LEFT JOIN (
            SELECT 
                customer_id,
                COUNT(*) as order_count_30d,
                SUM(total_amount) as total_spent_30d
            FROM orders 
            WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY customer_id
        ) recent_stats ON u.user_id = recent_stats.customer_id
        LEFT JOIN (
            SELECT 
                customer_id,
                COUNT(*) as total_orders,
                SUM(total_amount) as total_spent
            FROM orders
            GROUP BY customer_id
        ) lifetime_stats ON u.user_id = lifetime_stats.customer_id
        WHERE u.user_id = $1
        """
        
        # Test real-time serving for sample users
        sample_user_ids = [1, 2, 3, 5, 7]
        
        print("⚡ Real-time feature serving performance:")
        total_time = 0
        
        for user_id in sample_user_ids:
            start_time = time.time()
            
            user_features = await connector.execute_query(
                realtime_feature_sql,
                params=[user_id],
                cache_key=f'realtime_features_{user_id}',
                cache_ttl=60  # 1-minute cache for real-time
            )
            
            query_time = time.time() - start_time
            total_time += query_time
            
            if not user_features.empty:
                features = user_features.iloc[0]
                churn_risk = "High" if features['recent_orders'] == 0 and features['days_since_registration'] > 30 else "Low"
                
                print(f"   User {user_id}: {query_time:.3f}s - Churn risk: {churn_risk}")
        
        avg_time = total_time / len(sample_user_ids)
        print(f"✅ Average feature serving time: {avg_time:.3f}s (suitable for real-time)")
        
        # Model serving simulation
        print("\n🤖 Real-time model inference simulation...")
        
        # Batch inference for efficiency
        batch_features = await connector.execute_query(
            """
            SELECT user_id, is_active, 
                   EXTRACT(days FROM CURRENT_DATE - registration_date) as days_since_registration,
                   CASE WHEN last_login IS NOT NULL THEN 
                        EXTRACT(days FROM CURRENT_DATE - last_login::date) 
                        ELSE -1 END as days_since_last_login
            FROM users 
            WHERE user_id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            """,
            cache_key='batch_inference_features',
            cache_ttl=300
        )
        
        if not batch_features.empty:
            # Simulate batch predictions
            start_time = time.time()
            predictions = ml_pipeline.simulate_predictions('churn_prediction', batch_features)
            inference_time = time.time() - start_time
            
            print(f"✅ Batch inference completed: {len(predictions)} predictions in {inference_time:.3f}s")
            print(f"   Average per-prediction: {inference_time/len(predictions):.4f}s")
            
            # Show predictions
            print("\n📊 Real-time prediction results:")
            print(predictions[['user_id', 'churn_probability', 'churn_risk']].to_string(index=False))
        
        # Model monitoring simulation
        print("\n📈 Model performance monitoring:")
        
        # Simulate monitoring metrics
        monitoring_metrics = {
            'model_version': '1.2.3',
            'predictions_served_today': 15420,
            'average_latency_ms': avg_time * 1000,
            'cache_hit_rate': 0.87,
            'error_rate': 0.002,
            'feature_drift_score': 0.12,
            'model_accuracy_today': 0.892,
            'last_retrain_date': '2024-08-01'
        }
        
        print("🔍 Model health dashboard:")
        for metric, value in monitoring_metrics.items():
            if isinstance(value, float):
                if metric.endswith('_rate') or metric.endswith('_score') or metric.endswith('_accuracy'):
                    print(f"   {metric}: {value:.3f}")
                elif metric.endswith('_ms'):
                    print(f"   {metric}: {value:.1f}ms")
                else:
                    print(f"   {metric}: {value}")
            else:
                print(f"   {metric}: {value}")
        
        # Alert simulation
        if monitoring_metrics['feature_drift_score'] > 0.1:
            print("\n⚠️  Model monitoring alert:")
            print("   Feature drift detected - consider model retraining")
            print("   Recommended action: Schedule model refresh within 7 days")
        
    finally:
        await connector.close()


def setup_environment():
    """Setup environment variables for demo."""
    defaults = {
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'postgres',
        'POSTGRES_USER': 'postgres',
        'POSTGRES_PASSWORD': ''
    }
    
    for key, default_value in defaults.items():
        if key not in os.environ:
            os.environ[key] = default_value


async def main():
    """Run all ML integration examples."""
    print("🤖 ML Platform Integration Examples")
    print("=" * 70)
    print("Demonstrates comprehensive ML pipeline integration with")
    print("PostgreSQL FDW-based instant data connector architecture")
    print()
    
    setup_environment()
    
    examples = [
        ("Feature Extraction Pipeline", example_1_feature_extraction_pipeline),
        ("Churn Prediction Model", example_2_churn_prediction_model),
        ("Customer LTV Prediction", example_3_customer_ltv_prediction),
        ("Product Recommendation Engine", example_4_product_recommendation_engine),
        ("Real-time ML Serving", example_5_real_time_ml_serving)
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except KeyboardInterrupt:
            print(f"\n❌ Example '{name}' interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Example '{name}' failed: {e}")
            logger.exception(f"Detailed error for {name}")
    
    print("\n" + "="*70)
    print("🎉 ML Integration Examples Completed!")
    print()
    print("💡 Key ML integration patterns demonstrated:")
    print("  🔍 Efficient feature extraction with push-down optimization")
    print("  🤖 End-to-end ML pipeline workflows")
    print("  🎯 Real-time model serving with low latency")
    print("  📊 Comprehensive model monitoring and alerting")
    print("  🚀 Production-ready scalable architecture")
    print()
    print("🌟 Next steps for ML integration:")
    print("  1. Connect to actual ML platform APIs")
    print("  2. Implement automated feature pipelines")
    print("  3. Set up model training automation")
    print("  4. Deploy real-time inference endpoints")
    print("  5. Configure production monitoring dashboards")
    print("="*70)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ML integration demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo error details")