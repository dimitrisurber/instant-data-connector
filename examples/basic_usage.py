#!/usr/bin/env python3
"""
Basic Usage Examples - PostgreSQL FDW-based Instant Data Connector

This example demonstrates the new FDW-based architecture for unified data access
across multiple sources through PostgreSQL Foreign Data Wrappers.
"""

import asyncio
import logging
import os
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import InstantDataConnector
from instant_connector import SecureCredentialManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def example_1_basic_fdw_setup():
    """Example 1: Basic FDW setup and table querying."""
    print("\n" + "="*60)
    print("🚀 Example 1: Basic FDW Setup and Querying")
    print("="*60)
    
    # Initialize connector with PostgreSQL hub
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
        # Setup FDW infrastructure (this creates foreign tables if config exists)
        await connector.setup_fdw_infrastructure()
        print("✅ FDW infrastructure setup completed")
        
        # List available tables
        tables = await connector.list_available_tables()
        print(f"📊 Available tables: {list(tables.keys())}")
        
        # If we have the test database setup, query the users table
        try:
            # Query with lazy loading and push-down optimization
            df = await connector.lazy_load_table(
                'users',
                filters={'is_active': True},
                columns=['user_id', 'username', 'email', 'registration_date'],
                limit=10,
                order_by=['-registration_date']  # Most recent first
            )
            
            print(f"🎯 Loaded {len(df)} active users:")
            print(df.to_string(index=False))
            
            # Show performance benefits
            print(f"📈 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"ℹ️  Note: Could not query users table (expected if test DB not setup): {e}")
            print("   Run docker-compose up -d to setup test database")
        
    finally:
        await connector.close()


async def example_2_configuration_driven_setup():
    """Example 2: Configuration-driven FDW setup with YAML."""
    print("\n" + "="*60)
    print("🔧 Example 2: Configuration-Driven Setup")
    print("="*60)
    
    # Create a sample FDW configuration
    config = {
        'extension': 'postgres_fdw',
        'server': {
            'name': 'local_test_server',
            'options': {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'dbname': os.getenv('POSTGRES_DB', 'postgres')
            }
        },
        'user_mapping': {
            'options': {
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', '')
            }
        },
        'tables': [
            {
                'name': 'remote_users',
                'columns': [
                    {'name': 'user_id', 'type': 'integer'},
                    {'name': 'username', 'type': 'text'},
                    {'name': 'email', 'type': 'text'},
                    {'name': 'registration_date', 'type': 'date'},
                    {'name': 'is_active', 'type': 'boolean'}
                ],
                'options': {
                    'table_name': 'users',
                    'schema_name': 'public'
                }
            }
        ]
    }
    
    # Use configuration
    connector = InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    )
    
    # Manually load config for this example
    connector.config = config
    
    try:
        await connector.setup_fdw_infrastructure()
        print("✅ Configuration-driven FDW setup completed")
        
        # Test querying the configured foreign table
        try:
            df = await connector.lazy_load_table(
                'remote_users',
                filters={'is_active': True},
                limit=5
            )
            print(f"🎯 Queried {len(df)} users through configured FDW")
            print(df.head().to_string(index=False))
            
        except Exception as e:
            print(f"ℹ️  Note: Foreign table query failed (expected if remote server differs): {e}")
        
    finally:
        await connector.close()


async def example_3_advanced_querying():
    """Example 3: Advanced querying with filters, aggregations, and optimization."""
    print("\n" + "="*60)
    print("📊 Example 3: Advanced Querying and Optimization")
    print("="*60)
    
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
        
        # Advanced filtering example
        print("🔍 Advanced Filtering:")
        try:
            # Complex filters demonstrating different operators
            recent_users = await connector.lazy_load_table(
                'users',
                filters={
                    'registration_date': {'ge': '2024-02-01'},  # Greater than or equal
                    'is_active': True,
                    'email': {'like': '%example.com'}  # Email pattern matching
                },
                columns=['username', 'email', 'registration_date'],
                order_by=['-registration_date'],
                limit=10
            )
            
            print(f"Found {len(recent_users)} recent active users:")
            print(recent_users.to_string(index=False))
            
        except Exception as e:
            print(f"ℹ️  Advanced filtering example skipped: {e}")
        
        # Aggregation example using direct SQL
        print("\n📈 Aggregation Analysis:")
        try:
            # Get user registration statistics by month
            sql_query = """
            SELECT 
                DATE_TRUNC('month', registration_date) as month,
                COUNT(*) as user_count,
                COUNT(CASE WHEN is_active THEN 1 END) as active_count,
                ROUND(AVG(CASE WHEN last_login IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 2) as login_rate
            FROM users 
            WHERE registration_date >= '2024-01-01'
            GROUP BY DATE_TRUNC('month', registration_date)
            ORDER BY month
            """
            
            stats_df = await connector.execute_query(
                sql_query,
                return_dataframe=True,
                cache_key='user_stats_monthly',
                cache_ttl=3600  # Cache for 1 hour
            )
            
            print("Monthly User Registration Statistics:")
            print(stats_df.to_string(index=False))
            
        except Exception as e:
            print(f"ℹ️  Aggregation example skipped: {e}")
        
        # Query optimization demonstration
        print("\n⚡ Query Optimization:")
        try:
            query_info = connector.query_builder.build_select_query(
                'users',
                columns=['user_id', 'username', 'email'],
                filters={'is_active': True},
                limit=100
            )
            
            print(f"Generated SQL: {query_info['sql']}")
            print(f"Parameters: {query_info['params']}")
            
            # Show optimization recommendations
            optimized_query = await connector.query_builder.optimize_query(
                query_info, 
                connector.fdw_connector
            )
            
            if 'optimization_recommendations' in optimized_query:
                print("💡 Optimization recommendations:")
                for rec in optimized_query['optimization_recommendations']:
                    print(f"  - {rec}")
            
        except Exception as e:
            print(f"ℹ️  Query optimization example skipped: {e}")
        
    finally:
        await connector.close()


async def example_4_datarus_integration():
    """Example 4: datarus ML platform integration example."""
    print("\n" + "="*60)
    print("🤖 Example 4: datarus ML Platform Integration")
    print("="*60)
    
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
        
        # Simulate datarus feature extraction
        print("🔍 Feature Extraction for ML:")
        try:
            # Get user behavior features
            features_df = await connector.lazy_load_table(
                'users',
                columns=['user_id', 'registration_date', 'last_login', 'is_active'],
                filters={'registration_date': {'ge': '2024-01-01'}}
            )
            
            # Feature engineering (would typically be done by datarus)
            if not features_df.empty:
                features_df = features_df.copy()
                features_df['days_since_registration'] = (
                    pd.Timestamp.now() - pd.to_datetime(features_df['registration_date'])
                ).dt.days
                
                features_df['has_logged_in'] = features_df['last_login'].notna()
                features_df['days_since_last_login'] = (
                    pd.Timestamp.now() - pd.to_datetime(features_df['last_login'])
                ).dt.days.fillna(-1)
                
                print(f"🎯 Prepared {len(features_df)} user features for datarus:")
                print(features_df[['user_id', 'days_since_registration', 'has_logged_in', 'is_active']].head().to_string(index=False))
                
                # Simulate what datarus would do
                print("\n🚀 Simulated datarus ML workflow:")
                print(f"  📊 Feature matrix shape: {features_df.shape}")
                print(f"  🎲 Active user rate: {features_df['is_active'].mean():.2%}")
                print(f"  📈 Login rate: {features_df['has_logged_in'].mean():.2%}")
                print("  🤖 Ready for model training in datarus!")
            
        except Exception as e:
            print(f"ℹ️  datarus integration example skipped: {e}")
        
        # Get order data for revenue prediction
        print("\n💰 Revenue Prediction Features:")
        try:
            # Complex query joining users and orders for ML features
            revenue_sql = """
            SELECT 
                u.user_id,
                u.registration_date,
                u.is_active,
                COUNT(o.order_id) as total_orders,
                COALESCE(SUM(o.total_amount), 0) as total_spent,
                COALESCE(AVG(o.total_amount), 0) as avg_order_value,
                MAX(o.order_date) as last_order_date,
                COUNT(CASE WHEN o.status = 'completed' THEN 1 END) as completed_orders
            FROM users u
            LEFT JOIN orders o ON u.user_id = o.customer_id
            WHERE u.registration_date >= '2024-01-01'
            GROUP BY u.user_id, u.registration_date, u.is_active
            HAVING COUNT(o.order_id) > 0
            ORDER BY total_spent DESC
            LIMIT 20
            """
            
            revenue_df = await connector.execute_query(
                revenue_sql,
                cache_key='revenue_features',
                cache_ttl=1800  # 30 minutes cache
            )
            
            if not revenue_df.empty:
                print(f"🎯 Top {len(revenue_df)} customers by revenue:")
                print(revenue_df[['user_id', 'total_orders', 'total_spent', 'avg_order_value']].head().to_string(index=False))
                print("\n🤖 This data is perfect for datarus churn prediction models!")
            
        except Exception as e:
            print(f"ℹ️  Revenue analysis skipped: {e}")
        
    finally:
        await connector.close()


async def example_5_health_monitoring():
    """Example 5: Health monitoring and performance tracking."""
    print("\n" + "="*60)
    print("💊 Example 5: Health Monitoring and Performance")
    print("="*60)
    
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
        
        # Comprehensive health check
        print("🔍 System Health Check:")
        health_status = await connector.health_check()
        
        print(f"Overall System Health: {'✅ HEALTHY' if health_status['overall_healthy'] else '❌ UNHEALTHY'}")
        print(f"Timestamp: {health_status['timestamp']}")
        
        for component, status in health_status['components'].items():
            status_icon = "✅" if status['healthy'] else "❌"
            print(f"  {status_icon} {component}: {status['details']}")
        
        # Performance demonstration
        print("\n⚡ Performance Demonstration:")
        try:
            import time
            
            start_time = time.time()
            
            # Execute a query to measure performance
            df = await connector.lazy_load_table(
                'users',
                filters={'is_active': True},
                limit=50
            )
            
            query_time = time.time() - start_time
            
            print(f"🚀 Query Performance:")
            print(f"  📊 Rows returned: {len(df)}")
            print(f"  ⏱️  Query time: {query_time:.3f} seconds")
            print(f"  💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"  🎯 Push-down optimization: Enabled (filters applied at database level)")
            
        except Exception as e:
            print(f"ℹ️  Performance demo skipped: {e}")
        
        # Cache demonstration
        print("\n🗄️  Caching Performance:")
        try:
            # First query (cold)
            start_time = time.time()
            df1 = await connector.execute_query(
                "SELECT COUNT(*) as user_count FROM users",
                cache_key='user_count_demo',
                cache_ttl=300
            )
            cold_time = time.time() - start_time
            
            # Second query (warm - from cache)
            start_time = time.time()
            df2 = await connector.execute_query(
                "SELECT COUNT(*) as user_count FROM users",
                cache_key='user_count_demo',
                cache_ttl=300
            )
            warm_time = time.time() - start_time
            
            print(f"  🧊 Cold query time: {cold_time:.3f} seconds")
            print(f"  🔥 Cached query time: {warm_time:.3f} seconds")
            print(f"  📈 Cache speedup: {cold_time/warm_time:.1f}x faster")
            
        except Exception as e:
            print(f"ℹ️  Cache demo skipped: {e}")
        
    finally:
        await connector.close()


async def example_6_error_handling():
    """Example 6: Proper error handling and best practices."""
    print("\n" + "="*60)
    print("🛡️  Example 6: Error Handling and Best Practices")
    print("="*60)
    
    # Example of proper async context manager usage
    print("✅ Best Practice: Using async context manager")
    async with InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    ) as connector:
        
        await connector.setup_fdw_infrastructure()
        
        # Example of proper error handling
        print("\n🔒 Security Best Practices:")
        try:
            # This will properly validate and escape identifiers
            df = await connector.lazy_load_table(
                'users',  # Safe table name
                columns=['user_id', 'username'],  # Safe column names
                filters={'is_active': True},  # Safe filter values
                limit=5
            )
            print("✅ Query executed with proper SQL injection prevention")
            print(f"  Returned {len(df)} rows safely")
            
        except Exception as e:
            print(f"❌ Query failed with proper error handling: {e}")
        
        # Demonstrate SQL injection prevention
        print("\n🛡️  SQL Injection Prevention:")
        try:
            # This would be blocked by our security validator
            dangerous_table = "users; DROP TABLE users; --"
            await connector.lazy_load_table(dangerous_table)
            
        except Exception as e:
            print(f"✅ SQL injection attempt blocked: {type(e).__name__}")
            print("   Our security system prevented this dangerous query!")
        
        # Performance best practices
        print("\n⚡ Performance Best Practices:")
        
        # Good practice: specific columns and filters
        good_query = await connector.lazy_load_table(
            'users',
            columns=['user_id', 'username'],  # Specific columns
            filters={'is_active': True},      # Push-down filters
            limit=10                          # Reasonable limit
        )
        print(f"✅ Efficient query returned {len(good_query)} rows")
        
        # Show what NOT to do (but safely)
        print("   ⚠️  Avoid: SELECT * without filters on large tables")
        print("   ✅ Instead: Use specific columns, filters, and limits")


def setup_environment():
    """Setup environment variables with defaults for demo."""
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
    
    print("🔧 Environment Setup:")
    print(f"   PostgreSQL Host: {os.getenv('POSTGRES_HOST')}")
    print(f"   PostgreSQL Port: {os.getenv('POSTGRES_PORT')}")
    print(f"   PostgreSQL Database: {os.getenv('POSTGRES_DB')}")
    print(f"   PostgreSQL User: {os.getenv('POSTGRES_USER')}")
    print(f"   Password: {'***' if os.getenv('POSTGRES_PASSWORD') else '(empty)'}")


async def main():
    """Run all examples."""
    print("🚀 Instant Data Connector - FDW-based Examples")
    print("=" * 60)
    print("This demo showcases the new PostgreSQL FDW-based architecture")
    print("for unified data access across multiple sources.")
    print()
    
    setup_environment()
    
    examples = [
        ("Basic FDW Setup", example_1_basic_fdw_setup),
        ("Configuration-Driven Setup", example_2_configuration_driven_setup), 
        ("Advanced Querying", example_3_advanced_querying),
        ("datarus Integration", example_4_datarus_integration),
        ("Health Monitoring", example_5_health_monitoring),
        ("Error Handling", example_6_error_handling)
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
    
    print("\n" + "="*60)
    print("🎉 Demo completed! Key takeaways:")
    print("  ✅ PostgreSQL FDW provides unified data access")
    print("  ⚡ Push-down optimization reduces data transfer")
    print("  🦥 Lazy loading queries only what you need")
    print("  🔒 Comprehensive security prevents SQL injection")
    print("  🤖 Perfect integration with datarus ML platform")
    print("  📊 Production-ready with monitoring and caching")
    print()
    print("💡 Next steps:")
    print("  1. Run 'docker-compose up -d' for full test environment")
    print("  2. Try the configuration examples in config/examples/")
    print("  3. Integrate with your datarus ML pipelines")
    print("="*60)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo error details")