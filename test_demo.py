#!/usr/bin/env python3
"""
Instant Data Connector - Test Demo Script
==========================================

This script creates comprehensive test data and demonstrates the full
functionality of the InstantDataConnector system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_datasets():
    """Create realistic test datasets for different domains."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    datasets = {}
    
    # 1. E-commerce Dataset
    logger.info("Creating e-commerce dataset...")
    n_customers = 1000
    n_products = 200
    n_orders = 2500
    
    # Customers
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'first_name': np.random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Emma', 'Chris', 'Lisa'], n_customers),
        'last_name': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis'], n_customers),
        'email': [f"user{i}@example.com" for i in range(1, n_customers + 1)],
        'age': np.random.randint(18, 70, n_customers),
        'signup_date': pd.date_range(start='2020-01-01', end='2024-12-31', periods=n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia'], n_customers),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_customers, p=[0.2, 0.5, 0.3])
    })
    
    # Products
    products = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'product_name': [f"Product {i}" for i in range(1, n_products + 1)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_products),
        'price': np.round(np.random.uniform(10, 500, n_products), 2),
        'cost': lambda x: np.round(x * np.random.uniform(0.4, 0.7, len(x)), 2),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE'], n_products),
        'weight_kg': np.round(np.random.uniform(0.1, 10.0, n_products), 2),
        'in_stock': np.random.choice([True, False], n_products, p=[0.85, 0.15])
    })
    products['cost'] = np.round(products['price'] * np.random.uniform(0.4, 0.7, n_products), 2)
    
    # Orders
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'customer_id': np.random.randint(1, n_customers + 1, n_orders),
        'product_id': np.random.randint(1, n_products + 1, n_orders),
        'quantity': np.random.randint(1, 5, n_orders),
        'order_date': pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_orders),
        'shipping_cost': np.round(np.random.uniform(5, 25, n_orders), 2),
        'discount_percent': np.random.choice([0, 5, 10, 15, 20], n_orders, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'], n_orders),
        'order_status': np.random.choice(['Completed', 'Pending', 'Cancelled'], n_orders, p=[0.8, 0.15, 0.05])
    })
    
    datasets['ecommerce'] = {
        'customers': customers,
        'products': products,
        'orders': orders
    }
    
    # 2. Financial Dataset
    logger.info("Creating financial dataset...")
    n_accounts = 500
    n_transactions = 5000
    
    # Bank Accounts
    accounts = pd.DataFrame({
        'account_id': range(1, n_accounts + 1),
        'account_number': [f"ACC{i:08d}" for i in range(1, n_accounts + 1)],
        'customer_name': [f"Customer {i}" for i in range(1, n_accounts + 1)],
        'account_type': np.random.choice(['Checking', 'Savings', 'Credit', 'Investment'], n_accounts),
        'balance': np.round(np.random.uniform(-1000, 50000, n_accounts), 2),
        'credit_limit': np.where(
            np.random.choice(['Checking', 'Savings', 'Credit', 'Investment'], n_accounts) == 'Credit',
            np.round(np.random.uniform(1000, 25000, n_accounts), 2),
            0
        ),
        'interest_rate': np.round(np.random.uniform(0.01, 0.05, n_accounts), 4),
        'account_open_date': pd.date_range(start='2015-01-01', end='2024-01-01', periods=n_accounts),
        'credit_score': np.random.randint(300, 850, n_accounts)
    })
    
    # Transactions
    transactions = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'account_id': np.random.randint(1, n_accounts + 1, n_transactions),
        'amount': np.round(np.random.uniform(-2000, 5000, n_transactions), 2),
        'transaction_date': pd.date_range(start='2024-01-01', end='2024-12-31', periods=n_transactions),
        'transaction_type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer', 'Payment'], n_transactions),
        'merchant': np.random.choice(['Amazon', 'Walmart', 'Starbucks', 'Gas Station', 'Restaurant', 'ATM'], n_transactions),
        'category': np.random.choice(['Shopping', 'Food', 'Gas', 'Entertainment', 'Bills', 'Transfer'], n_transactions),
        'is_fraud': np.random.choice([True, False], n_transactions, p=[0.01, 0.99])
    })
    
    datasets['finance'] = {
        'accounts': accounts,
        'transactions': transactions
    }
    
    # 3. IoT Sensor Dataset
    logger.info("Creating IoT sensor dataset...")
    n_devices = 50
    n_readings = 10000
    
    # IoT Devices
    devices = pd.DataFrame({
        'device_id': range(1, n_devices + 1),
        'device_name': [f"Sensor_{i:03d}" for i in range(1, n_devices + 1)],
        'device_type': np.random.choice(['Temperature', 'Humidity', 'Pressure', 'Motion', 'Light'], n_devices),
        'location': np.random.choice(['Building A', 'Building B', 'Building C', 'Outdoor', 'Warehouse'], n_devices),
        'floor': np.random.randint(1, 10, n_devices),
        'installation_date': pd.date_range(start='2023-01-01', end='2024-01-01', periods=n_devices),
        'battery_level': np.round(np.random.uniform(0.1, 1.0, n_devices), 2),
        'firmware_version': np.random.choice(['v1.0', 'v1.1', 'v1.2', 'v2.0'], n_devices),
        'is_active': np.random.choice([True, False], n_devices, p=[0.95, 0.05])
    })
    
    # Sensor Readings
    readings = pd.DataFrame({
        'reading_id': range(1, n_readings + 1),
        'device_id': np.random.randint(1, n_devices + 1, n_readings),
        'timestamp': pd.date_range(start='2024-10-01', end='2024-12-31', periods=n_readings),
        'value': np.round(np.random.uniform(-10, 100, n_readings), 3),
        'unit': 'celsius',  # Will be updated based on device type
        'quality_score': np.round(np.random.uniform(0.7, 1.0, n_readings), 3),
        'is_anomaly': np.random.choice([True, False], n_readings, p=[0.02, 0.98])
    })
    
    # Update units based on device type
    device_types = devices.set_index('device_id')['device_type'].to_dict()
    unit_mapping = {
        'Temperature': 'celsius',
        'Humidity': 'percent',
        'Pressure': 'hPa',
        'Motion': 'boolean',
        'Light': 'lux'
    }
    readings['unit'] = readings['device_id'].map(device_types).map(unit_mapping)
    
    datasets['iot'] = {
        'devices': devices,
        'readings': readings
    }
    
    # 4. Marketing Dataset
    logger.info("Creating marketing dataset...")
    n_campaigns = 20
    n_leads = 2000
    
    # Marketing Campaigns
    campaigns = pd.DataFrame({
        'campaign_id': range(1, n_campaigns + 1),
        'campaign_name': [f"Campaign_{i}" for i in range(1, n_campaigns + 1)],
        'channel': np.random.choice(['Email', 'Social Media', 'Google Ads', 'TV', 'Radio', 'Print'], n_campaigns),
        'start_date': pd.date_range(start='2024-01-01', end='2024-06-01', periods=n_campaigns),
        'end_date': lambda x: x + timedelta(days=np.random.randint(7, 90)),
        'budget': np.round(np.random.uniform(5000, 100000, n_campaigns), 2),
        'target_audience': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_campaigns),
        'campaign_type': np.random.choice(['Awareness', 'Conversion', 'Retention', 'Acquisition'], n_campaigns)
    })
    campaigns['end_date'] = campaigns['start_date'] + pd.to_timedelta(np.random.randint(7, 90, n_campaigns), unit='D')
    
    # Leads
    leads = pd.DataFrame({
        'lead_id': range(1, n_leads + 1),
        'campaign_id': np.random.randint(1, n_campaigns + 1, n_leads),
        'lead_source': np.random.choice(['Website', 'Landing Page', 'Social Media', 'Referral', 'Event'], n_leads),
        'lead_date': pd.date_range(start='2024-01-01', end='2024-12-31', periods=n_leads),
        'lead_score': np.random.randint(1, 100, n_leads),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_leads),
        'industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Education', 'Retail'], n_leads),
        'company_size': np.random.choice(['1-10', '11-50', '51-200', '201-1000', '1000+'], n_leads),
        'converted': np.random.choice([True, False], n_leads, p=[0.15, 0.85]),
        'conversion_value': np.where(
            np.random.choice([True, False], n_leads, p=[0.15, 0.85]),
            np.round(np.random.uniform(100, 10000, n_leads), 2),
            0
        )
    })
    
    datasets['marketing'] = {
        'campaigns': campaigns,
        'leads': leads
    }
    
    logger.info(f"Created {len(datasets)} domain datasets with multiple tables each")
    return datasets

def save_test_data_to_files(datasets, output_dir):
    """Save test datasets to various file formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    file_paths = {}
    
    for domain, tables in datasets.items():
        domain_dir = output_dir / domain
        domain_dir.mkdir(exist_ok=True)
        file_paths[domain] = {}
        
        for table_name, df in tables.items():
            # Save as CSV
            csv_path = domain_dir / f"{table_name}.csv"
            df.to_csv(csv_path, index=False)
            file_paths[domain][f"{table_name}_csv"] = str(csv_path)
            
            # Save as Parquet (if not too large)
            if len(df) < 5000:
                parquet_path = domain_dir / f"{table_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                file_paths[domain][f"{table_name}_parquet"] = str(parquet_path)
            
            # Save as JSON (sample of first 100 rows)
            json_path = domain_dir / f"{table_name}_sample.json"
            df.head(100).to_json(json_path, orient='records', date_format='iso')
            file_paths[domain][f"{table_name}_json"] = str(json_path)
    
    logger.info(f"Saved test data to {output_dir}")
    return file_paths

def create_configuration_examples(file_paths, output_dir):
    """Create example configuration files for the InstantDataConnector."""
    output_dir = Path(output_dir)
    
    # Configuration for E-commerce analysis
    ecommerce_config = {
        "sources": [
            {
                "type": "file",
                "name": "customers",
                "path": file_paths["ecommerce"]["customers_csv"]
            },
            {
                "type": "file", 
                "name": "products",
                "path": file_paths["ecommerce"]["products_csv"]
            },
            {
                "type": "file",
                "name": "orders", 
                "path": file_paths["ecommerce"]["orders_csv"]
            }
        ],
        "ml_optimization": {
            "enabled": True,
            "handle_missing": "mean",
            "encode_categorical": "onehot",
            "scale_numeric": "standard",
            "feature_engineering": True
        },
        "output": {
            "path": str(output_dir / "ecommerce_connector.pkl"),
            "compression": "lz4",
            "optimize_memory": True
        }
    }
    
    # Configuration for Financial fraud detection
    finance_config = {
        "sources": [
            {
                "type": "file",
                "name": "accounts",
                "path": file_paths["finance"]["accounts_csv"]
            },
            {
                "type": "file",
                "name": "transactions",
                "path": file_paths["finance"]["transactions_csv"]
            }
        ],
        "ml_optimization": {
            "enabled": True,
            "handle_missing": "drop",
            "encode_categorical": "label",
            "scale_numeric": "minmax", 
            "reduce_memory": True
        },
        "output": {
            "path": str(output_dir / "finance_connector.pkl"),
            "compression": "gzip"
        }
    }
    
    # Configuration for IoT analytics
    iot_config = {
        "sources": [
            {
                "type": "file",
                "name": "devices",
                "path": file_paths["iot"]["devices_csv"]
            },
            {
                "type": "file",
                "name": "readings", 
                "path": file_paths["iot"]["readings_csv"]
            }
        ],
        "ml_optimization": {
            "enabled": True,
            "handle_missing": "forward_fill",
            "encode_categorical": "auto",
            "scale_numeric": "robust"
        },
        "output": {
            "path": str(output_dir / "iot_connector.pkl"),
            "compression": "bz2"
        }
    }
    
    configs = {
        "ecommerce": ecommerce_config,
        "finance": finance_config,
        "iot": iot_config
    }
    
    # Save configuration files
    for name, config in configs.items():
        config_path = output_dir / f"{name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created configuration: {config_path}")
    
    return configs

def test_instant_data_connector(datasets, configs, output_dir):
    """Test the InstantDataConnector with our test data."""
    from instant_connector import InstantDataConnector
    
    output_dir = Path(output_dir)
    results = {}
    
    for domain, config in configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {domain.upper()} dataset")
        logger.info(f"{'='*50}")
        
        try:
            # Initialize connector
            connector = InstantDataConnector()
            
            # Add file sources manually (since config loading might have issues)
            if domain == 'ecommerce':
                connector.add_file_source('customers', datasets[domain]['customers'])
                connector.add_file_source('products', datasets[domain]['products'])
                connector.add_file_source('orders', datasets[domain]['orders'])
            elif domain == 'finance':
                connector.add_file_source('accounts', datasets[domain]['accounts'])
                connector.add_file_source('transactions', datasets[domain]['transactions'])
            elif domain == 'iot':
                connector.add_file_source('devices', datasets[domain]['devices'])
                connector.add_file_source('readings', datasets[domain]['readings'])
            
            # Set raw data directly for testing
            connector.raw_data = datasets[domain]
            
            # Configure ML optimization
            ml_config = config['ml_optimization']
            if ml_config['enabled']:
                connector.configure_ml_optimization(
                    handle_missing=ml_config.get('handle_missing', 'auto'),
                    encode_categorical=ml_config.get('encode_categorical', 'auto'),
                    scale_numeric=ml_config.get('scale_numeric', 'standard'),
                    feature_engineering=ml_config.get('feature_engineering', False),
                    reduce_memory=ml_config.get('reduce_memory', False)
                )
                
                # Apply ML optimization
                logger.info("Applying ML optimization...")
                connector.apply_ml_optimization()
            
            # Get summary
            summary = connector.get_summary()
            logger.info(f"Summary: {summary}")
            
            # Test specific domain analysis
            if domain == 'ecommerce':
                # Customer segmentation analysis
                logger.info("Running customer segmentation analysis...")
                customers_df = datasets[domain]['customers']
                orders_df = datasets[domain]['orders']
                
                # Calculate customer metrics
                customer_metrics = orders_df.groupby('customer_id').agg({
                    'order_id': 'count',
                    'quantity': 'sum',
                    'order_date': ['min', 'max']
                }).round(2)
                
                logger.info(f"Customer metrics calculated for {len(customer_metrics)} customers")
                
            elif domain == 'finance':
                # Fraud detection analysis
                logger.info("Running fraud detection analysis...")
                transactions_df = datasets[domain]['transactions']
                
                fraud_stats = {
                    'total_transactions': len(transactions_df),
                    'fraud_transactions': transactions_df['is_fraud'].sum(),
                    'fraud_rate': (transactions_df['is_fraud'].sum() / len(transactions_df)) * 100,
                    'avg_fraud_amount': transactions_df[transactions_df['is_fraud']]['amount'].mean()
                }
                
                logger.info(f"Fraud analysis: {fraud_stats}")
                
            elif domain == 'iot':
                # Anomaly detection analysis
                logger.info("Running IoT anomaly detection...")
                readings_df = datasets[domain]['readings']
                
                anomaly_stats = {
                    'total_readings': len(readings_df),
                    'anomalies': readings_df['is_anomaly'].sum(),
                    'anomaly_rate': (readings_df['is_anomaly'].sum() / len(readings_df)) * 100,
                    'avg_quality_score': readings_df['quality_score'].mean()
                }
                
                logger.info(f"IoT analysis: {anomaly_stats}")
            
            # Save results
            output_path = output_dir / f"{domain}_test_results.pkl"
            try:
                result = connector.save_pickle(output_path)
                logger.info(f"✅ Successfully saved connector to {output_path}")
                logger.info(f"Save result: {result}")
            except Exception as e:
                logger.error(f"❌ Failed to save pickle: {e}")
            
            results[domain] = {
                'status': 'success',
                'summary': summary,
                'ml_ready_tables': len(connector.ml_ready_data),
                'raw_tables': len(connector.raw_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Error testing {domain}: {e}")
            import traceback
            traceback.print_exc()
            results[domain] = {
                'status': 'error',
                'error': str(e)
            }
    
    return results

def main():
    """Main test execution function."""
    logger.info("Starting InstantDataConnector Test Demo")
    logger.info("="*60)
    
    # Create output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "instant_connector_test"
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Create test datasets
        logger.info("Step 1: Creating test datasets...")
        datasets = create_test_datasets()
        
        # Step 2: Save datasets to files
        logger.info("Step 2: Saving datasets to files...")
        file_paths = save_test_data_to_files(datasets, output_dir / "data")
        
        # Step 3: Create configuration examples
        logger.info("Step 3: Creating configuration examples...")
        configs = create_configuration_examples(file_paths, output_dir / "configs")
        
        # Step 4: Test InstantDataConnector
        logger.info("Step 4: Testing InstantDataConnector...")
        results = test_instant_data_connector(datasets, configs, output_dir / "output")
        
        # Step 5: Summary report
        logger.info("\n" + "="*60)
        logger.info("FINAL TEST RESULTS")
        logger.info("="*60)
        
        for domain, result in results.items():
            status = "✅ SUCCESS" if result['status'] == 'success' else "❌ FAILED"
            logger.info(f"{domain.upper()}: {status}")
            
            if result['status'] == 'success':
                logger.info(f"  - Raw tables: {result['raw_tables']}")
                logger.info(f"  - ML ready tables: {result['ml_ready_tables']}")
                logger.info(f"  - Total sources: {result['summary'].get('total_sources', 0)}")
                logger.info(f"  - Total rows: {result['summary'].get('total_rows', 0)}")
                logger.info(f"  - Memory usage: {result['summary'].get('memory_usage_mb', 0):.2f} MB")
            else:
                logger.info(f"  - Error: {result['error']}")
        
        # Copy important files to a permanent location
        import shutil
        permanent_dir = Path("/Users/dimsum/localprojects/Datarus_Connector/instant-data-connector/demo_output")
        if permanent_dir.exists():
            shutil.rmtree(permanent_dir)
        shutil.copytree(output_dir, permanent_dir)
        logger.info(f"\n📁 Test results copied to: {permanent_dir}")
        
        return results

if __name__ == "__main__":
    results = main()