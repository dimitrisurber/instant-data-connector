#!/usr/bin/env python3
"""
Advanced PostgreSQL Foreign Data Wrapper (FDW) Examples

This example demonstrates advanced FDW usage including:
- Multi-source data federation
- File-based FDW for CSV/JSON data
- Query optimization and push-down analysis
- Performance monitoring and health checks
- Real-world integration patterns
"""

import asyncio
import logging
import os
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import InstantDataConnector, PostgreSQLFDWConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def example_1_direct_fdw_management():
    """Example 1: Direct FDW management with PostgreSQLFDWConnector."""
    print("\n" + "="*70)
    print("🔧 Example 1: Direct FDW Management")
    print("="*70)
    
    # Direct FDW connector usage
    fdw_connector = PostgreSQLFDWConnector(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'postgres'),
        username=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
        pool_size=5
    )
    
    try:
        await fdw_connector.initialize()
        print("✅ FDW connector initialized")
        
        # Install postgres_fdw extension
        try:
            await fdw_connector.install_extension('postgres_fdw')
            print("✅ postgres_fdw extension installed")
        except Exception as e:
            print(f"ℹ️  Extension install: {e} (may already exist)")
        
        # Create foreign server for local connection (demo)
        server_config = {
            'name': 'demo_local_server',
            'options': {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'dbname': os.getenv('POSTGRES_DB', 'postgres')
            }
        }
        
        try:
            await fdw_connector.create_foreign_server(
                server_config['name'],
                'postgres_fdw',
                server_config['options']
            )
            print(f"✅ Created foreign server: {server_config['name']}")
        except Exception as e:
            print(f"ℹ️  Server creation: {e} (may already exist)")
        
        # Create user mapping
        try:
            await fdw_connector.create_user_mapping(
                server_config['name'],
                'current_user',
                {
                    'user': os.getenv('POSTGRES_USER', 'postgres'),
                    'password': os.getenv('POSTGRES_PASSWORD', '')
                }
            )
            print("✅ Created user mapping")
        except Exception as e:
            print(f"ℹ️  User mapping: {e} (may already exist)")
        
        # Create foreign table for existing users table
        try:
            await fdw_connector.create_foreign_table(
                'fdw_users',
                server_config['name'],
                [
                    {'name': 'user_id', 'type': 'integer'},
                    {'name': 'username', 'type': 'text'},
                    {'name': 'email', 'type': 'text'},
                    {'name': 'registration_date', 'type': 'date'},
                    {'name': 'is_active', 'type': 'boolean'}
                ],
                {'table_name': 'users', 'schema_name': 'public'}
            )
            print("✅ Created foreign table: fdw_users")
        except Exception as e:
            print(f"ℹ️  Foreign table: {e} (may already exist)")
        
        # Test querying the foreign table
        try:
            df = fdw_connector.execute_lazy_query(
                'fdw_users',
                columns=['user_id', 'username', 'email'],
                filters={'is_active': True},
                limit=10
            )
            print(f"🎯 Queried {len(df)} users through FDW:")
            print(df.head().to_string(index=False))
        except Exception as e:
            print(f"ℹ️  Query failed: {e} (expected if no test data)")
        
        # Show managed objects
        managed = fdw_connector.get_managed_objects()
        print(f"\n📊 Managed FDW objects:")
        for obj_type, objects in managed.items():
            print(f"  {obj_type}: {objects}")
        
    finally:
        await fdw_connector.close()


async def example_2_file_fdw_csv_demo():
    """Example 2: File FDW demonstration with CSV data."""
    print("\n" + "="*70)
    print("📁 Example 2: File FDW with CSV Data")
    print("="*70)
    
    # Ensure sample data exists
    sample_data_dir = Path(__file__).parent / 'sample_data'
    sales_csv = sample_data_dir / 'sales_data.csv'
    customer_csv = sample_data_dir / 'customer_data.csv'
    
    if not sales_csv.exists() or not customer_csv.exists():
        print("⚠️  Sample data files not found. Please run the basic_usage.py example first.")
        return
    
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
        
        # Configure file FDW for CSV files
        file_fdw_config = {
            'extension': 'file_fdw',
            'server': {
                'name': 'csv_file_server',
                'options': {}
            },
            'tables': [
                {
                    'name': 'csv_sales_data',
                    'columns': [
                        {'name': 'product_id', 'type': 'integer'},
                        {'name': 'product_name', 'type': 'text'},
                        {'name': 'quantity', 'type': 'integer'},
                        {'name': 'price', 'type': 'decimal(10,2)'},
                        {'name': 'sale_date', 'type': 'date'},
                        {'name': 'customer_id', 'type': 'integer'},
                        {'name': 'region', 'type': 'text'},
                        {'name': 'sales_rep', 'type': 'text'}
                    ],
                    'options': {
                        'filename': str(sales_csv.absolute()),
                        'format': 'csv',
                        'header': 'true'
                    }
                },
                {
                    'name': 'csv_customer_data',
                    'columns': [
                        {'name': 'customer_id', 'type': 'integer'},
                        {'name': 'first_name', 'type': 'text'},
                        {'name': 'last_name', 'type': 'text'},
                        {'name': 'email', 'type': 'text'},
                        {'name': 'registration_date', 'type': 'date'},
                        {'name': 'country', 'type': 'text'},
                        {'name': 'subscription_tier', 'type': 'text'},
                        {'name': 'total_spent', 'type': 'decimal(10,2)'},
                        {'name': 'is_active', 'type': 'boolean'}
                    ],
                    'options': {
                        'filename': str(customer_csv.absolute()),
                        'format': 'csv',
                        'header': 'true'
                    }
                }
            ]
        }
        
        # Setup file FDW
        connector.config = file_fdw_config
        await connector.setup_fdw_infrastructure()
        print("✅ File FDW setup completed")
        
        # Query CSV data through FDW
        try:
            sales_df = await connector.lazy_load_table(
                'csv_sales_data',
                filters={'region': 'North America'},
                columns=['product_name', 'quantity', 'price', 'sale_date'],
                order_by=['-sale_date'],
                limit=10
            )
            print(f"🎯 North American sales ({len(sales_df)} records):")
            print(sales_df.to_string(index=False))
        except Exception as e:
            print(f"ℹ️  CSV sales query failed: {e}")
        
        # Query customer data
        try:
            customer_df = await connector.lazy_load_table(
                'csv_customer_data',
                filters={'subscription_tier': 'Premium', 'is_active': True},
                columns=['first_name', 'last_name', 'country', 'total_spent'],
                order_by=['-total_spent'],
                limit=10
            )
            print(f"\n💎 Premium customers ({len(customer_df)} records):")
            print(customer_df.to_string(index=False))
        except Exception as e:
            print(f"ℹ️  CSV customer query failed: {e}")
        
        # Cross-source analytics via SQL
        try:
            analytics_sql = """
            SELECT 
                c.country,
                COUNT(*) as customer_count,
                SUM(c.total_spent) as total_revenue,
                AVG(c.total_spent) as avg_customer_value
            FROM csv_customer_data c
            WHERE c.is_active = true
            GROUP BY c.country
            ORDER BY total_revenue DESC
            LIMIT 5
            """
            
            analytics_df = await connector.execute_query(
                analytics_sql,
                cache_key='country_analytics',
                cache_ttl=1800
            )
            
            print(f"\n📊 Revenue by Country:")
            print(analytics_df.to_string(index=False))
            
        except Exception as e:
            print(f"ℹ️  Analytics query failed: {e}")
        
    finally:
        await connector.close()


async def example_3_query_optimization_analysis():
    """Example 3: Query optimization and performance analysis."""
    print("\n" + "="*70)
    print("⚡ Example 3: Query Optimization Analysis")
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
        
        # Build complex query for analysis
        print("🔍 Building complex query for optimization analysis:")
        
        query_info = connector.query_builder.build_select_query(
            'users',
            columns=['user_id', 'username', 'email', 'registration_date'],
            filters={
                'registration_date': {'ge': '2024-01-01'},
                'is_active': True,
                'email': {'like': '%@example.com'}
            },
            order_by=['-registration_date'],
            limit=50
        )
        
        print(f"Generated SQL:")
        print(f"  {query_info['sql']}")
        print(f"Parameters: {query_info['params']}")
        
        # Analyze query optimization
        try:
            optimized_query = await connector.query_builder.optimize_query(
                query_info,
                connector.fdw_connector
            )
            
            print(f"\n📈 Query Optimization Analysis:")
            if 'cost_estimate' in optimized_query:
                print(f"  Estimated cost: {optimized_query['cost_estimate']}")
            if 'push_down_eligible' in optimized_query:
                print(f"  Push-down eligible: {optimized_query['push_down_eligible']}")
            if 'optimization_recommendations' in optimized_query:
                print(f"  Recommendations:")
                for rec in optimized_query['optimization_recommendations']:
                    print(f"    - {rec}")
            
        except Exception as e:
            print(f"ℹ️  Optimization analysis skipped: {e}")
        
        # Performance comparison
        print(f"\n⏱️  Performance Testing:")
        
        # Test 1: Query with filters (good performance)
        start_time = time.time()
        try:
            df1 = await connector.lazy_load_table(
                'users',
                filters={'is_active': True},
                columns=['user_id', 'username'],
                limit=20
            )
            time1 = time.time() - start_time
            print(f"  ✅ Filtered query: {time1:.3f}s ({len(df1)} rows)")
        except Exception as e:
            print(f"  ℹ️  Filtered query failed: {e}")
        
        # Test 2: Query with aggregation
        start_time = time.time()
        try:
            agg_sql = """
            SELECT 
                is_active,
                COUNT(*) as user_count,
                MIN(registration_date) as earliest_reg,
                MAX(registration_date) as latest_reg
            FROM users 
            GROUP BY is_active
            """
            
            df2 = await connector.execute_query(agg_sql)
            time2 = time.time() - start_time
            print(f"  ✅ Aggregation query: {time2:.3f}s ({len(df2)} rows)")
            print(f"     Results:")
            print(f"     {df2.to_string(index=False)}")
            
        except Exception as e:
            print(f"  ℹ️  Aggregation query failed: {e}")
        
        # Test 3: Cache performance
        print(f"\n🗄️  Cache Performance Test:")
        cache_key = 'perf_test_cache'
        
        # Cold query
        start_time = time.time()
        try:
            df3 = await connector.execute_query(
                "SELECT COUNT(*) as total_users FROM users",
                cache_key=cache_key,
                cache_ttl=300
            )
            cold_time = time.time() - start_time
            print(f"  🧊 Cold query: {cold_time:.3f}s")
        except Exception as e:
            print(f"  ℹ️  Cold query failed: {e}")
            cold_time = 0
        
        # Warm query (from cache)
        start_time = time.time()
        try:
            df4 = await connector.execute_query(
                "SELECT COUNT(*) as total_users FROM users",
                cache_key=cache_key,
                cache_ttl=300
            )
            warm_time = time.time() - start_time
            speedup = cold_time / warm_time if warm_time > 0 else 0
            print(f"  🔥 Cached query: {warm_time:.3f}s (speedup: {speedup:.1f}x)")
        except Exception as e:
            print(f"  ℹ️  Cached query failed: {e}")
        
    finally:
        await connector.close()


async def example_4_health_monitoring_advanced():
    """Example 4: Advanced health monitoring and diagnostics."""
    print("\n" + "="*70)
    print("💊 Example 4: Advanced Health Monitoring")
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
        
        # Comprehensive health check
        print("🔍 Comprehensive System Health Check:")
        health_status = await connector.health_check()
        
        # Display health status with details
        overall_status = "✅ HEALTHY" if health_status['overall_healthy'] else "❌ UNHEALTHY"
        print(f"Overall Status: {overall_status}")
        print(f"Timestamp: {health_status['timestamp']}")
        print(f"Components:")
        
        for component, status in health_status['components'].items():
            icon = "✅" if status['healthy'] else "❌"
            print(f"  {icon} {component.replace('_', ' ').title()}")
            print(f"     {status['details']}")
        
        # Connection pool diagnostics
        if connector.fdw_connector and hasattr(connector.fdw_connector, '_async_pool'):
            pool = connector.fdw_connector._async_pool
            if pool:
                print(f"\n🔗 Connection Pool Status:")
                print(f"  Pool size: {pool.get_size()}")
                print(f"  Available connections: {pool.get_idle_size()}")
                print(f"  Max connections: {pool.get_max_size()}")
        
        # Database diagnostics
        try:
            diagnostics_sql = """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes
            FROM pg_stat_user_tables 
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT 5
            """
            
            diag_df = await connector.execute_query(diagnostics_sql)
            if not diag_df.empty:
                print(f"\n📊 Database Table Statistics:")
                print(diag_df.to_string(index=False))
            
        except Exception as e:
            print(f"ℹ️  Database diagnostics skipped: {e}")
        
        # Performance metrics over time
        print(f"\n📈 Performance Metrics Collection:")
        metrics = []
        
        for i in range(3):
            start_time = time.time()
            try:
                test_df = await connector.execute_query(
                    "SELECT COUNT(*) FROM users WHERE is_active = true",
                    cache_key=f'health_test_{i}',
                    cache_ttl=60
                )
                query_time = time.time() - start_time
                metrics.append({
                    'iteration': i + 1,
                    'query_time': query_time,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                print(f"  Test {i+1}: {query_time:.3f}s")
            except Exception as e:
                print(f"  Test {i+1}: Failed - {e}")
            
            if i < 2:  # Don't sleep after last iteration
                await asyncio.sleep(1)
        
        # Calculate performance statistics
        if metrics:
            times = [m['query_time'] for m in metrics]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Performance Summary:")
            print(f"  Average query time: {avg_time:.3f}s")
            print(f"  Min query time: {min_time:.3f}s")
            print(f"  Max query time: {max_time:.3f}s")
            print(f"  Performance variance: {max_time - min_time:.3f}s")
        
        # Test error handling
        print(f"\n🛡️  Error Handling Test:")
        try:
            # This should fail gracefully
            await connector.lazy_load_table('nonexistent_table')
        except Exception as e:
            print(f"  ✅ Error handled gracefully: {type(e).__name__}")
            print(f"     Error message: {str(e)[:100]}...")
        
    finally:
        await connector.close()


async def example_5_real_world_integration():
    """Example 5: Real-world integration patterns and best practices."""
    print("\n" + "="*70)
    print("🌍 Example 5: Real-World Integration Patterns")
    print("="*70)
    
    # Demonstrate proper resource management with context manager
    async with InstantDataConnector(
        postgres_config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'postgres'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        },
        enable_caching=True
    ) as connector:
        
        await connector.setup_fdw_infrastructure()
        
        # Pattern 1: Data Pipeline Integration
        print("🔄 Pattern 1: Data Pipeline Integration")
        
        async def extract_user_features(date_filter: str) -> pd.DataFrame:
            """Extract user features for ML pipeline."""
            return await connector.lazy_load_table(
                'users',
                columns=['user_id', 'registration_date', 'is_active', 'last_login'],
                filters={'registration_date': {'ge': date_filter}},
                order_by=['user_id']
            )
        
        async def extract_order_metrics(date_filter: str) -> pd.DataFrame:
            """Extract order metrics for analytics."""
            sql = """
            SELECT 
                DATE_TRUNC('day', order_date) as order_day,
                status,
                COUNT(*) as order_count,
                SUM(total_amount) as daily_revenue,
                AVG(total_amount) as avg_order_value
            FROM orders 
            WHERE order_date >= $1
            GROUP BY DATE_TRUNC('day', order_date), status
            ORDER BY order_day DESC, status
            """
            return await connector.execute_query(
                sql, 
                params=[date_filter],
                cache_key=f'daily_metrics_{date_filter}',
                cache_ttl=3600
            )
        
        try:
            # Simulate daily data extraction
            cutoff_date = '2024-07-01'
            
            user_features = await extract_user_features(cutoff_date)
            order_metrics = await extract_order_metrics(cutoff_date)
            
            print(f"  ✅ Extracted {len(user_features)} user features")
            print(f"  ✅ Extracted {len(order_metrics)} order metrics")
            
            if not user_features.empty:
                print(f"     User feature sample:")
                print(f"     {user_features.head(3).to_string(index=False)}")
            
            if not order_metrics.empty:
                print(f"     Order metrics sample:")
                print(f"     {order_metrics.head(3).to_string(index=False)}")
                
        except Exception as e:
            print(f"  ℹ️  Data extraction skipped: {e}")
        
        # Pattern 2: Batch Processing with Error Handling
        print(f"\n⚙️  Pattern 2: Batch Processing with Error Handling")
        
        batch_queries = [
            ("Active Users", "SELECT COUNT(*) as count FROM users WHERE is_active = true"),
            ("Recent Orders", "SELECT COUNT(*) as count FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '7 days'"),
            ("Revenue Today", "SELECT COALESCE(SUM(total_amount), 0) as revenue FROM orders WHERE DATE(order_date) = CURRENT_DATE")
        ]
        
        results = {}
        for name, query in batch_queries:
            try:
                start_time = time.time()
                df = await connector.execute_query(
                    query,
                    cache_key=f'batch_{name.lower().replace(" ", "_")}',
                    cache_ttl=300
                )
                query_time = time.time() - start_time
                
                if not df.empty:
                    value = df.iloc[0, 0]
                    results[name] = {'value': value, 'time': query_time}
                    print(f"  ✅ {name}: {value} ({query_time:.3f}s)")
                else:
                    results[name] = {'value': 0, 'time': query_time}
                    print(f"  ⚠️  {name}: No data ({query_time:.3f}s)")
                    
            except Exception as e:
                results[name] = {'error': str(e), 'time': 0}
                print(f"  ❌ {name}: Error - {e}")
        
        # Pattern 3: Configuration-Driven Multi-Source
        print(f"\n🔧 Pattern 3: Configuration-Driven Multi-Source Setup")
        
        # Example configuration for multiple data sources
        multi_source_config = {
            'sources': {
                'production_db': {
                    'extension': 'postgres_fdw',
                    'server': {
                        'name': 'prod_server',
                        'options': {
                            'host': 'prod-db.example.com',
                            'port': '5432',
                            'dbname': 'production'
                        }
                    },
                    'tables': ['users', 'orders', 'products']
                },
                'analytics_db': {
                    'extension': 'postgres_fdw', 
                    'server': {
                        'name': 'analytics_server',
                        'options': {
                            'host': 'analytics-db.example.com',
                            'port': '5432',
                            'dbname': 'analytics'
                        }
                    },
                    'tables': ['user_events', 'conversion_metrics']
                }
            }
        }
        
        print(f"  📋 Multi-source configuration example:")
        print(f"     Sources: {list(multi_source_config['sources'].keys())}")
        print(f"     Total tables: {sum(len(src['tables']) for src in multi_source_config['sources'].values())}")
        print(f"  💡 This pattern enables unified queries across multiple databases")
        
        # Pattern 4: Monitoring and Alerting Integration
        print(f"\n📊 Pattern 4: Monitoring Integration")
        
        # Simulate monitoring metrics collection
        monitoring_metrics = {
            'query_performance': results,
            'system_health': health_status if 'health_status' in locals() else {'status': 'unknown'},
            'cache_effectiveness': 'High' if len([r for r in results.values() if 'time' in r and r['time'] < 0.1]) > 0 else 'Medium',
            'error_rate': len([r for r in results.values() if 'error' in r]) / len(results) * 100
        }
        
        print(f"  📈 Monitoring metrics collected:")
        for metric, value in monitoring_metrics.items():
            if isinstance(value, dict):
                print(f"     {metric}: {len(value)} items")
            else:
                print(f"     {metric}: {value}")
        
        print(f"\n💡 Integration Patterns Summary:")
        print(f"  ✅ Context managers ensure proper resource cleanup")
        print(f"  ✅ Async patterns enable concurrent data extraction")
        print(f"  ✅ Error handling prevents pipeline failures")
        print(f"  ✅ Caching reduces database load")
        print(f"  ✅ Configuration-driven setup enables flexibility")
        print(f"  ✅ Monitoring integration provides observability")


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
    """Run all FDW examples."""
    print("🚀 Advanced PostgreSQL FDW Examples")
    print("=" * 70)
    print("Demonstrates advanced FDW patterns for production use")
    print()
    
    setup_environment()
    
    examples = [
        ("Direct FDW Management", example_1_direct_fdw_management),
        ("File FDW with CSV", example_2_file_fdw_csv_demo),
        ("Query Optimization", example_3_query_optimization_analysis),
        ("Health Monitoring", example_4_health_monitoring_advanced),
        ("Real-World Integration", example_5_real_world_integration)
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
    print("🎉 Advanced FDW Examples Completed!")
    print()
    print("💡 Key learnings:")
    print("  🔧 Direct FDW management for fine-grained control")
    print("  📁 File FDW enables CSV/JSON data integration") 
    print("  ⚡ Query optimization improves performance")
    print("  💊 Health monitoring ensures system reliability")
    print("  🌍 Real-world patterns for production deployment")
    print()
    print("🚀 Next steps:")
    print("  1. Adapt these patterns to your data sources")
    print("  2. Implement monitoring in your environment")
    print("  3. Set up automated health checks")
    print("  4. Configure multi-source federation")
    print("="*70)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Advanced FDW demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo error details")