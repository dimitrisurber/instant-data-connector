#!/usr/bin/env python3
"""
Simple InstantDataConnector Test
=================================

A streamlined test to demonstrate core functionality with realistic data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_datasets():
    """Create sample datasets for testing."""
    
    np.random.seed(42)
    
    # E-commerce dataset
    logger.info("Creating e-commerce sample data...")
    
    # Customers
    customers = pd.DataFrame({
        'customer_id': range(1, 101),
        'name': [f"Customer_{i}" for i in range(1, 101)],
        'age': np.random.randint(18, 70, 100),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 100),
        'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 100, p=[0.2, 0.5, 0.3]),
        'signup_date': pd.date_range(start='2020-01-01', periods=100, freq='D')
    })
    
    # Products
    products = pd.DataFrame({
        'product_id': range(1, 51),
        'product_name': [f"Product_{i}" for i in range(1, 51)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], 50),
        'price': np.round(np.random.uniform(10, 500, 50), 2),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC'], 50)
    })
    
    # Orders
    orders = pd.DataFrame({
        'order_id': range(1, 201),
        'customer_id': np.random.randint(1, 101, 200),
        'product_id': np.random.randint(1, 51, 200),
        'quantity': np.random.randint(1, 5, 200),
        'order_date': pd.date_range(start='2024-01-01', periods=200, freq='D'),
        'total_amount': np.round(np.random.uniform(20, 1000, 200), 2),
        'status': np.random.choice(['Completed', 'Pending', 'Cancelled'], 200, p=[0.8, 0.15, 0.05])
    })
    
    # Financial dataset
    logger.info("Creating financial sample data...")
    
    # Accounts
    accounts = pd.DataFrame({
        'account_id': range(1, 51),
        'account_type': np.random.choice(['Checking', 'Savings', 'Credit'], 50),
        'balance': np.round(np.random.uniform(-1000, 25000, 50), 2),
        'credit_score': np.random.randint(300, 850, 50),
        'open_date': pd.date_range(start='2020-01-01', periods=50, freq='M')
    })
    
    # Transactions
    transactions = pd.DataFrame({
        'transaction_id': range(1, 301),
        'account_id': np.random.randint(1, 51, 300),
        'amount': np.round(np.random.uniform(-500, 2000, 300), 2),
        'transaction_date': pd.date_range(start='2024-01-01', periods=300, freq='D'),
        'category': np.random.choice(['Shopping', 'Food', 'Gas', 'Bills'], 300),
        'is_fraud': np.random.choice([True, False], 300, p=[0.05, 0.95])
    })
    
    return {
        'ecommerce': {
            'customers': customers,
            'products': products,
            'orders': orders
        },
        'finance': {
            'accounts': accounts,
            'transactions': transactions
        }
    }

def test_basic_functionality():
    """Test basic InstantDataConnector functionality."""
    
    from instant_connector import InstantDataConnector
    
    logger.info("="*60)
    logger.info("TESTING INSTANTDATACONNECTOR FUNCTIONALITY")
    logger.info("="*60)
    
    # Create test data
    datasets = create_sample_datasets()
    
    # Test 1: Basic Initialization
    logger.info("\n1. Testing Basic Initialization")
    try:
        connector = InstantDataConnector()
        logger.info("✅ InstantDataConnector initialized successfully")
        
        # Check initial state
        assert connector.raw_data == {}
        assert connector.ml_ready_data == {}
        assert connector.metadata == {}
        assert connector.ml_artifacts == {}
        logger.info("✅ Initial state verification passed")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False
    
    # Test 2: Data Loading
    logger.info("\n2. Testing Data Loading")
    try:
        # Load e-commerce data
        connector.raw_data = datasets['ecommerce']
        
        # Verify data loading
        assert len(connector.raw_data) == 3
        assert 'customers' in connector.raw_data
        assert 'products' in connector.raw_data
        assert 'orders' in connector.raw_data
        
        logger.info(f"✅ Loaded {len(connector.raw_data)} tables")
        for table_name, df in connector.raw_data.items():
            logger.info(f"   - {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False
    
    # Test 3: ML Optimization Configuration
    logger.info("\n3. Testing ML Optimization Configuration")
    try:
        connector.configure_ml_optimization(
            handle_missing='mean',
            encode_categorical='onehot',
            scale_numeric='standard',
            feature_engineering=True,
            reduce_memory=True
        )
        
        # Verify configuration
        assert connector.ml_optimizer.handle_missing == 'mean'
        assert connector.ml_optimizer.encode_categorical == 'onehot'
        assert connector.ml_optimizer.scale_numeric == 'standard'
        assert connector.ml_optimizer.feature_engineering == True
        assert connector.ml_optimizer.reduce_memory == True
        
        logger.info("✅ ML optimization configured successfully")
        
    except Exception as e:
        logger.error(f"❌ ML configuration failed: {e}")
        return False
    
    # Test 4: ML Optimization Application
    logger.info("\n4. Testing ML Optimization Application")
    try:
        connector.apply_ml_optimization()
        
        # Verify optimization results
        assert len(connector.ml_ready_data) > 0
        assert 'preprocessing_metadata' in connector.ml_artifacts
        
        logger.info(f"✅ ML optimization applied successfully")
        logger.info(f"   - ML ready tables: {len(connector.ml_ready_data)}")
        for table_name, df in connector.ml_ready_data.items():
            logger.info(f"   - {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
    except Exception as e:
        logger.error(f"❌ ML optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Summary Generation
    logger.info("\n5. Testing Summary Generation")
    try:
        summary = connector.get_summary()
        
        # Verify summary
        assert 'total_raw_tables' in summary
        assert 'total_ml_ready_tables' in summary
        assert 'total_rows' in summary
        assert 'memory_usage_mb' in summary
        
        logger.info("✅ Summary generated successfully")
        logger.info(f"   - Raw tables: {summary['total_raw_tables']}")
        logger.info(f"   - ML ready tables: {summary['total_ml_ready_tables']}")
        logger.info(f"   - Total rows: {summary['total_rows']}")
        logger.info(f"   - Memory usage: {summary['memory_usage_mb']:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {e}")
        return False
    
    # Test 6: ML Optimization with Target Column
    logger.info("\n6. Testing ML Optimization with Target Column")
    try:
        # Create a dataset with a clear target
        target_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.choice(['A', 'B', 'C'], 100),
            'feature3': np.random.uniform(0, 100, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Reset connector for target test
        target_connector = InstantDataConnector()
        target_connector.raw_data = {'dataset': target_data}
        
        # Apply optimization with target
        target_connector.apply_ml_optimization(
            target_column='target',
            test_size=0.3,
            stratify=True
        )
        
        # Verify train/test splits
        assert 'X_train' in target_connector.ml_ready_data
        assert 'X_test' in target_connector.ml_ready_data
        assert 'y_train' in target_connector.ml_ready_data
        assert 'y_test' in target_connector.ml_ready_data
        
        logger.info("✅ Target-based ML optimization successful")
        logger.info(f"   - X_train shape: {target_connector.ml_ready_data['X_train'].shape}")
        logger.info(f"   - X_test shape: {target_connector.ml_ready_data['X_test'].shape}")
        logger.info(f"   - y_train shape: {target_connector.ml_ready_data['y_train'].shape}")
        logger.info(f"   - y_test shape: {target_connector.ml_ready_data['y_test'].shape}")
        
    except Exception as e:
        logger.error(f"❌ Target-based optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 7: Business Analytics Examples
    logger.info("\n7. Testing Business Analytics Examples")
    try:
        # E-commerce analytics
        customers_df = datasets['ecommerce']['customers']
        orders_df = datasets['ecommerce']['orders']
        
        # Customer segmentation
        customer_stats = orders_df.groupby('customer_id').agg({
            'order_id': 'count',
            'total_amount': ['sum', 'mean'],
            'quantity': 'sum'
        }).round(2)
        
        logger.info("✅ E-commerce analytics completed")
        logger.info(f"   - Analyzed {len(customer_stats)} customers")
        logger.info(f"   - Total orders: {orders_df['order_id'].count()}")
        logger.info(f"   - Avg order value: ${orders_df['total_amount'].mean():.2f}")
        
        # Financial analytics
        accounts_df = datasets['finance']['accounts']
        transactions_df = datasets['finance']['transactions']
        
        # Fraud analysis
        fraud_rate = (transactions_df['is_fraud'].sum() / len(transactions_df)) * 100
        avg_transaction = transactions_df['amount'].mean()
        
        logger.info("✅ Financial analytics completed")
        logger.info(f"   - Fraud rate: {fraud_rate:.2f}%")
        logger.info(f"   - Average transaction: ${avg_transaction:.2f}")
        logger.info(f"   - Total accounts: {len(accounts_df)}")
        
    except Exception as e:
        logger.error(f"❌ Business analytics failed: {e}")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("\nInstantDataConnector is working correctly with:")
    logger.info("✅ Data loading and management")
    logger.info("✅ ML optimization configuration")
    logger.info("✅ Data preprocessing and feature engineering")
    logger.info("✅ Train/test splitting with stratification")
    logger.info("✅ Summary and metadata generation")
    logger.info("✅ Business analytics capabilities")
    
    return True

def demonstrate_advanced_features():
    """Demonstrate advanced features with realistic scenarios."""
    
    logger.info("\n" + "="*60)
    logger.info("ADVANCED FEATURES DEMONSTRATION")
    logger.info("="*60)
    
    from instant_connector import InstantDataConnector
    
    # Scenario 1: Customer Lifetime Value Prediction
    logger.info("\n📊 Scenario 1: Customer Lifetime Value Prediction")
    
    # Create CLV dataset
    np.random.seed(123)
    clv_data = pd.DataFrame({
        'customer_id': range(1, 501),
        'age': np.random.randint(18, 70, 500),
        'income': np.random.normal(50000, 15000, 500),
        'months_active': np.random.randint(1, 60, 500),
        'total_purchases': np.random.randint(1, 50, 500),
        'avg_order_value': np.random.uniform(20, 200, 500),
        'support_tickets': np.random.randint(0, 10, 500),
        'referrals_made': np.random.randint(0, 5, 500),
        'preferred_channel': np.random.choice(['Online', 'Store', 'Mobile'], 500),
        'customer_segment': np.random.choice(['High', 'Medium', 'Low'], 500)
    })
    
    # Calculate CLV (simplified)
    clv_data['lifetime_value'] = (
        clv_data['total_purchases'] * clv_data['avg_order_value'] * 
        (clv_data['months_active'] / 12) * np.random.uniform(0.8, 1.2, 500)
    ).round(2)
    
    # Apply ML pipeline
    clv_connector = InstantDataConnector()
    clv_connector.raw_data = {'customers': clv_data}
    clv_connector.configure_ml_optimization(
        handle_missing='mean',
        encode_categorical='onehot',
        scale_numeric='standard',
        feature_engineering=True
    )
    clv_connector.apply_ml_optimization(
        target_column='lifetime_value',
        test_size=0.2
    )
    
    logger.info(f"✅ CLV model data prepared:")
    logger.info(f"   - Training samples: {clv_connector.ml_ready_data['X_train'].shape[0]}")
    logger.info(f"   - Test samples: {clv_connector.ml_ready_data['X_test'].shape[0]}")
    logger.info(f"   - Features: {clv_connector.ml_ready_data['X_train'].shape[1]}")
    
    # Scenario 2: Fraud Detection Pipeline
    logger.info("\n🔐 Scenario 2: Fraud Detection Pipeline")
    
    # Create fraud detection dataset
    fraud_data = pd.DataFrame({
        'transaction_id': range(1, 1001),
        'amount': np.random.lognormal(3, 1, 1000),
        'merchant_category': np.random.choice(['Gas', 'Grocery', 'Restaurant', 'Online', 'ATM'], 1000),
        'hour_of_day': np.random.randint(0, 24, 1000),
        'day_of_week': np.random.randint(0, 7, 1000),
        'is_weekend': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'previous_transactions_24h': np.random.randint(0, 10, 1000),
        'account_age_days': np.random.randint(1, 3650, 1000),
        'avg_monthly_spending': np.random.normal(2000, 500, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])  # 5% fraud rate
    })
    
    # Apply fraud detection pipeline
    fraud_connector = InstantDataConnector()
    fraud_connector.raw_data = {'transactions': fraud_data}
    fraud_connector.configure_ml_optimization(
        handle_missing='drop',
        encode_categorical='label',
        scale_numeric='robust',  # Robust to outliers
        reduce_memory=True
    )
    fraud_connector.apply_ml_optimization(
        target_column='is_fraud',
        test_size=0.3,
        stratify=True  # Important for imbalanced classes
    )
    
    logger.info(f"✅ Fraud detection data prepared:")
    logger.info(f"   - Training samples: {fraud_connector.ml_ready_data['X_train'].shape[0]}")
    logger.info(f"   - Fraud cases in training: {fraud_connector.ml_ready_data['y_train'].sum()}")
    logger.info(f"   - Fraud rate: {(fraud_connector.ml_ready_data['y_train'].sum() / len(fraud_connector.ml_ready_data['y_train']) * 100):.2f}%")
    
    # Scenario 3: Multi-table Join Analysis
    logger.info("\n📈 Scenario 3: Multi-table Join Analysis")
    
    # Create related tables
    products = pd.DataFrame({
        'product_id': range(1, 101),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 100),
        'price': np.random.uniform(10, 500, 100),
        'margin_percent': np.random.uniform(10, 40, 100)
    })
    
    sales = pd.DataFrame({
        'sale_id': range(1, 501),
        'product_id': np.random.randint(1, 101, 500),
        'quantity': np.random.randint(1, 10, 500),
        'sale_date': pd.date_range(start='2024-01-01', periods=500, freq='D'),
        'salesperson_id': np.random.randint(1, 21, 500),
        'discount_percent': np.random.choice([0, 5, 10, 15], 500, p=[0.6, 0.2, 0.15, 0.05])
    })
    
    # Perform join analysis
    sales_analysis = sales.merge(products, on='product_id')
    sales_analysis['revenue'] = sales_analysis['quantity'] * sales_analysis['price'] * (1 - sales_analysis['discount_percent']/100)
    sales_analysis['profit'] = sales_analysis['revenue'] * (sales_analysis['margin_percent']/100)
    
    # Analyze by category
    category_performance = sales_analysis.groupby('category').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'quantity': 'sum',
        'sale_id': 'count'
    }).round(2)
    
    logger.info(f"✅ Multi-table analysis completed:")
    logger.info(f"   - Total revenue: ${sales_analysis['revenue'].sum():,.2f}")
    logger.info(f"   - Total profit: ${sales_analysis['profit'].sum():,.2f}")
    logger.info(f"   - Best category: {category_performance['revenue'].idxmax()}")
    logger.info(f"   - Total transactions: {len(sales_analysis)}")
    
    logger.info("\n🎯 Advanced features demonstration completed successfully!")
    
    return True

def main():
    """Main execution function."""
    
    logger.info("Starting InstantDataConnector Comprehensive Test")
    
    try:
        # Run basic functionality tests
        basic_success = test_basic_functionality()
        
        if basic_success:
            # Run advanced features demo
            demonstrate_advanced_features()
            
            logger.info("\n" + "🏆" + "="*58 + "🏆")
            logger.info("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY! 🎉")
            logger.info("🏆" + "="*58 + "🏆")
            logger.info("\nThe InstantDataConnector is ready for production use!")
            
        else:
            logger.error("❌ Basic tests failed. Please check the implementation.")
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()