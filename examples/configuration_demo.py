#!/usr/bin/env python3
"""
Configuration-Driven Demo - PostgreSQL FDW-based Data Connector

This example demonstrates comprehensive configuration-driven setup using YAML
configuration files to define multiple data sources and foreign data wrappers.
Shows how to build production-ready multi-source data integration pipelines.
"""

import asyncio
import logging
import os
import pandas as pd
import yaml
from pathlib import Path
import sys
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import InstantDataConnector
from instant_connector import SecureCredentialManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages YAML configuration files and environment variable substitution."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.configs = {}
        
    def load_config(self, config_name: str) -> Dict:
        """Load and parse a YAML configuration file."""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Substitute environment variables
        processed_config = self._substitute_env_vars(raw_config)
        self.configs[config_name] = processed_config
        
        return processed_config
    
    def _substitute_env_vars(self, obj):
        """Recursively substitute environment variables in configuration."""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Handle ${VAR} and ${VAR:default} patterns
            import re
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.getenv(var_name, default_value)
            
            return re.sub(pattern, replace_var, obj)
        else:
            return obj
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration files."""
        return [f.stem for f in self.config_dir.glob("*.yaml")]
    
    def get_config_summary(self, config_name: str) -> Dict:
        """Get a summary of a configuration."""
        if config_name not in self.configs:
            self.load_config(config_name)
        
        config = self.configs[config_name]
        sources = config.get('sources', {})
        
        summary = {
            'name': config.get('metadata', {}).get('name', config_name),
            'description': config.get('metadata', {}).get('description', ''),
            'source_count': len(sources),
            'source_types': list(set(src.get('type', 'unknown') for src in sources.values())),
            'total_tables': sum(len(src.get('tables', [])) for src in sources.values()),
            'enabled_sources': sum(1 for src in sources.values() if src.get('enabled', True))
        }
        
        return summary


async def example_1_postgres_fdw_configuration():
    """Example 1: PostgreSQL FDW configuration from YAML."""
    print("\n" + "="*70)
    print("🔧 Example 1: PostgreSQL FDW Configuration")
    print("="*70)
    
    # Setup environment variables for demo
    demo_env = {
        'ANALYTICS_DB_HOST': 'localhost',
        'ANALYTICS_DB_USER': 'postgres',
        'ANALYTICS_DB_PASSWORD': '',
        'WAREHOUSE_DB_HOST': 'localhost',
        'WAREHOUSE_DB_PORT': '5432',
        'WAREHOUSE_DB_USER': 'postgres',
        'REPLICA_DB_HOST': 'localhost',
        'REPLICA_DB_PASSWORD': ''
    }
    
    for key, value in demo_env.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Load configuration
    config_dir = Path(__file__).parent.parent / 'config' / 'examples'
    config_manager = ConfigurationManager(config_dir)
    
    try:
        postgres_config = config_manager.load_config('postgres_fdw_example')
        
        print("✅ Configuration loaded successfully")
        print(f"📋 Configuration: {postgres_config['metadata']['name']}")
        print(f"📝 Description: {postgres_config['metadata']['description']}")
        
        # Show configuration summary
        summary = config_manager.get_config_summary('postgres_fdw_example')
        print(f"\n📊 Configuration Summary:")
        print(f"   Sources: {summary['source_count']} ({summary['enabled_sources']} enabled)")
        print(f"   Types: {', '.join(summary['source_types'])}")
        print(f"   Total tables: {summary['total_tables']}")
        
        # Initialize connector with base PostgreSQL config
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
            
            # Apply configuration to connector
            connector.config = postgres_config
            print("\n🔄 Applying multi-source configuration...")
            
            # Process each source from configuration
            for source_name, source_config in postgres_config['sources'].items():
                if not source_config.get('enabled', True):
                    print(f"⏭️  Skipping disabled source: {source_name}")
                    continue
                
                print(f"\n📡 Processing source: {source_name}")
                print(f"   Type: {source_config['type']}")
                print(f"   Description: {source_config.get('description', 'N/A')}")
                print(f"   Tables: {len(source_config.get('tables', []))}")
                
                # Show what would be created (simulation)
                if source_config['type'] == 'postgres_fdw':
                    server_opts = source_config.get('server_options', {})
                    host = server_opts.get('host', 'unknown')
                    dbname = server_opts.get('dbname', 'unknown')
                    
                    print(f"   Would create foreign server for: {host}/{dbname}")
                    
                    for table_config in source_config.get('tables', []):
                        table_name = table_config['name']
                        column_count = len(table_config.get('columns', []))
                        remote_table = table_config.get('options', {}).get('table_name', table_name)
                        
                        print(f"     - Table: {table_name} -> {remote_table} ({column_count} columns)")
            
            # Demonstrate cross-source query planning
            print(f"\n🎯 Cross-source Query Examples:")
            print("   The following queries would be possible with this configuration:")
            
            cross_source_queries = [
                {
                    'description': 'User activity analysis across analytics and warehouse',
                    'sql': '''
                    SELECT 
                        up.username,
                        up.email,
                        COUNT(ue.id) as event_count,
                        cs.segment,
                        cs.ltv_score
                    FROM user_profiles up
                    LEFT JOIN user_events ue ON up.id = ue.user_id
                    LEFT JOIN customer_segments cs ON up.id = cs.customer_id
                    WHERE ue.created_at >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY up.username, up.email, cs.segment, cs.ltv_score
                    ORDER BY event_count DESC
                    '''
                },
                {
                    'description': 'Sales performance with real-time replica data',
                    'sql': '''
                    SELECT 
                        DATE_TRUNC('day', o.order_date) as day,
                        COUNT(*) as order_count,
                        SUM(o.total_amount) as daily_revenue,
                        ss.total_sales as warehouse_total
                    FROM orders o
                    LEFT JOIN sales_summary ss ON DATE_TRUNC('day', o.order_date) = ss.date
                    WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY DATE_TRUNC('day', o.order_date), ss.total_sales
                    ORDER BY day DESC
                    '''
                }
            ]
            
            for i, query in enumerate(cross_source_queries, 1):
                print(f"\n   Query {i}: {query['description']}")
                print(f"   Preview: {query['sql'].strip()[:100]}...")
            
        finally:
            await connector.close()
        
    except Exception as e:
        print(f"❌ Configuration demo failed: {e}")
        logger.exception("Configuration error details")


async def example_2_mixed_sources_configuration():
    """Example 2: Mixed sources configuration with multiple FDW types."""
    print("\n" + "="*70)
    print("🌐 Example 2: Mixed Sources Configuration")
    print("="*70)
    
    # Setup comprehensive demo environment
    demo_env = {
        'WAREHOUSE_DB_HOST': 'localhost',
        'WAREHOUSE_DB_USER': 'postgres',
        'WAREHOUSE_DB_PASSWORD': '',
        'MYSQL_HOST': 'localhost',
        'MYSQL_USER': 'root',
        'MYSQL_PASSWORD': '',
        'DATA_FEEDS_DIR': '/tmp/data_feeds',
        'SUPPLIER_DATA_DIR': '/tmp/supplier_data',
        'WEATHER_API_URL': 'https://api.weather.example.com',
        'WEATHER_API_KEY': 'demo_key_12345',
        'EXCHANGE_API_URL': 'https://api.exchange.example.com',
        'EXCHANGE_API_TOKEN': 'demo_token_67890',
        'MAINFRAME_HOST': 'mainframe.internal.com',
        'MAINFRAME_PORT': '8471',
        'MAINFRAME_USER': 'fdw_user',
        'MONGODB_URI': 'mongodb://localhost:27017',
        'MONGODB_USER': 'analytics_user',
        'MONGODB_PASSWORD': 'secure_password',
        'DATA_LAKE_PATH': '/data/lake'
    }
    
    for key, value in demo_env.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Load mixed sources configuration
    config_dir = Path(__file__).parent.parent / 'config' / 'examples'
    config_manager = ConfigurationManager(config_dir)
    
    try:
        mixed_config = config_manager.load_config('mixed_sources_example')
        
        print("✅ Mixed sources configuration loaded")
        summary = config_manager.get_config_summary('mixed_sources_example')
        
        print(f"\n📊 Comprehensive Data Architecture:")
        print(f"   Total sources: {summary['source_count']}")
        print(f"   FDW types: {', '.join(summary['source_types'])}")
        print(f"   Total tables: {summary['total_tables']}")
        print(f"   Enabled sources: {summary['enabled_sources']}")
        
        # Analyze each source type
        sources = mixed_config['sources']
        source_analysis = {}
        
        for source_name, source_config in sources.items():
            source_type = source_config.get('type', 'unknown')
            if source_type not in source_analysis:
                source_analysis[source_type] = {'count': 0, 'tables': 0, 'sources': []}
            
            source_analysis[source_type]['count'] += 1
            source_analysis[source_type]['tables'] += len(source_config.get('tables', []))
            source_analysis[source_type]['sources'].append(source_name)
        
        print(f"\n🔍 Source Type Analysis:")
        for fdw_type, info in source_analysis.items():
            print(f"   {fdw_type}:")
            print(f"     Sources: {info['count']} ({', '.join(info['sources'])})")
            print(f"     Tables: {info['tables']}")
        
        # Show data integration opportunities
        print(f"\n🔄 Data Integration Opportunities:")
        
        integration_scenarios = [
            {
                'name': 'Real-time Inventory Management',
                'description': 'Combine warehouse dimensions with operational MySQL inventory',
                'sources': ['main_warehouse.dim_customers', 'operational_mysql.current_inventory'],
                'benefit': 'Real-time stock levels with customer segmentation'
            },
            {
                'name': 'Weather-Influenced Sales Analysis',
                'description': 'Correlate sales data with weather conditions',
                'sources': ['main_warehouse.fact_sales', 'api_integrations.weather_data'],
                'benefit': 'Understand weather impact on purchasing patterns'
            },
            {
                'name': 'Multi-Channel Customer View',
                'description': 'Unified view across operational, warehouse, and legacy systems',
                'sources': ['main_warehouse.dim_customers', 'operational_mysql.active_orders', 'legacy_system.legacy_customers'],
                'benefit': '360-degree customer insights'
            },
            {
                'name': 'Supplier Performance Analytics',
                'description': 'Track supplier reliability using inventory and file data',
                'sources': ['operational_mysql.current_inventory', 'external_files.supplier_catalog'],
                'benefit': 'Optimize supplier relationships and procurement'
            },
            {
                'name': 'Web Analytics + Sales Correlation',
                'description': 'Connect web behavior with purchase outcomes',
                'sources': ['external_files.web_analytics', 'document_store.user_events', 'main_warehouse.fact_sales'],
                'benefit': 'Improve digital marketing ROI'
            }
        ]
        
        for i, scenario in enumerate(integration_scenarios, 1):
            print(f"\n   Scenario {i}: {scenario['name']}")
            print(f"      Description: {scenario['description']}")
            print(f"      Data sources: {', '.join(scenario['sources'])}")
            print(f"      Business benefit: {scenario['benefit']}")
        
        # Initialize connector for demonstration
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
            
            # Simulate configuration application
            print(f"\n🚀 Configuration Application Simulation:")
            connector.config = mixed_config
            
            total_setup_time = 0
            for source_name, source_config in sources.items():
                if not source_config.get('enabled', True):
                    continue
                
                start_time = time.time()
                
                # Simulate setup time based on source type and complexity
                source_type = source_config.get('type', 'unknown')
                table_count = len(source_config.get('tables', []))
                
                if source_type == 'postgres_fdw':
                    setup_time = 0.5 + (table_count * 0.1)
                elif source_type == 'mysql_fdw':
                    setup_time = 0.8 + (table_count * 0.15)
                elif source_type == 'file_fdw':
                    setup_time = 0.3 + (table_count * 0.05)
                elif source_type == 'multicorn':
                    setup_time = 1.2 + (table_count * 0.2)  # API connections take longer
                else:
                    setup_time = 1.0
                
                # Simulate setup delay
                await asyncio.sleep(min(setup_time, 0.5))  # Cap at 0.5s for demo
                
                actual_time = time.time() - start_time
                total_setup_time += actual_time
                
                status = "✅" if source_type in ['postgres_fdw', 'file_fdw'] else "🔄"
                print(f"   {status} {source_name} ({source_type}): {actual_time:.2f}s - {table_count} tables")
            
            print(f"\n⏱️  Total configuration setup time: {total_setup_time:.2f}s")
            print(f"📈 Performance: {summary['total_tables'] / total_setup_time:.1f} tables/second")
            
            # Show example unified queries
            print(f"\n🎯 Example Unified Queries:")
            
            unified_queries = [
                '''-- Customer Lifetime Value with Real-time Context
                SELECT 
                    dc.customer_name,
                    dc.customer_segment,
                    dc.lifetime_value as historical_ltv,
                    COUNT(ao.order_id) as active_orders,
                    SUM(ao.total_amount) as pending_revenue,
                    lc.credit_limit,
                    lc.current_balance
                FROM dim_customers dc
                LEFT JOIN active_orders ao ON dc.customer_id = ao.customer_id
                LEFT JOIN legacy_customers lc ON dc.customer_id = lc.customer_number::integer
                WHERE dc.customer_segment IN ('High Value', 'VIP')
                GROUP BY dc.customer_name, dc.customer_segment, dc.lifetime_value, 
                         lc.credit_limit, lc.current_balance
                ORDER BY historical_ltv DESC''',
                
                '''-- Weather-Influenced Product Performance
                SELECT 
                    DATE_TRUNC('day', fs.date_key::date) as sale_date,
                    p.category,
                    SUM(fs.total_amount) as daily_sales,
                    wd.temperature,
                    wd.conditions,
                    CASE 
                        WHEN wd.temperature > 25 THEN 'Hot'
                        WHEN wd.temperature < 10 THEN 'Cold'
                        ELSE 'Moderate'
                    END as temp_category
                FROM fact_sales fs
                JOIN dim_products p ON fs.product_key = p.product_key
                LEFT JOIN weather_data wd ON fs.date_key::date = wd.timestamp::date
                WHERE fs.date_key >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', fs.date_key::date), p.category, 
                         wd.temperature, wd.conditions
                ORDER BY sale_date DESC, daily_sales DESC'''
            ]
            
            for i, query in enumerate(unified_queries, 1):
                print(f"\n   Query {i} Preview:")
                lines = query.strip().split('\n')
                print(f"   {lines[0].strip()}")
                print(f"   {lines[1].strip()}")
                print(f"   ... ({len(lines)} total lines)")
            
        finally:
            await connector.close()
        
    except Exception as e:
        print(f"❌ Mixed sources demo failed: {e}")
        logger.exception("Mixed sources error details")


async def example_3_dynamic_configuration_management():
    """Example 3: Dynamic configuration management and validation."""
    print("\n" + "="*70)
    print("⚙️  Example 3: Dynamic Configuration Management")
    print("="*70)
    
    config_dir = Path(__file__).parent.parent / 'config' / 'examples'
    config_manager = ConfigurationManager(config_dir)
    
    # List all available configurations
    available_configs = config_manager.list_available_configs()
    print(f"📋 Available Configurations: {len(available_configs)}")
    
    for config_name in available_configs:
        try:
            summary = config_manager.get_config_summary(config_name)
            print(f"\n   📄 {config_name}:")
            print(f"      Name: {summary['name']}")
            print(f"      Description: {summary['description'][:80]}...")
            print(f"      Sources: {summary['source_count']} ({', '.join(summary['source_types'])})")
            print(f"      Tables: {summary['total_tables']}")
            
        except Exception as e:
            print(f"   ❌ {config_name}: Error loading - {e}")
    
    # Demonstrate configuration validation
    print(f"\n🔍 Configuration Validation:")
    
    def validate_config(config: Dict) -> List[str]:
        """Validate configuration structure and requirements."""
        issues = []
        
        # Check required top-level fields
        required_fields = ['version', 'sources']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Validate sources
        sources = config.get('sources', {})
        if not sources:
            issues.append("No sources defined")
        
        for source_name, source_config in sources.items():
            # Check required source fields
            if 'type' not in source_config:
                issues.append(f"Source '{source_name}': Missing type")
            
            source_type = source_config.get('type')
            if source_type not in ['postgres_fdw', 'mysql_fdw', 'file_fdw', 'multicorn']:
                issues.append(f"Source '{source_name}': Unknown type '{source_type}'")
            
            # Check tables
            tables = source_config.get('tables', [])
            if not tables:
                issues.append(f"Source '{source_name}': No tables defined")
            
            for table in tables:
                if 'name' not in table:
                    issues.append(f"Source '{source_name}': Table missing name")
                
                columns = table.get('columns', [])
                if not columns:
                    issues.append(f"Source '{source_name}', table '{table.get('name', 'unknown')}': No columns defined")
        
        return issues
    
    # Validate each configuration
    for config_name in available_configs:
        try:
            config = config_manager.load_config(config_name)
            issues = validate_config(config)
            
            if issues:
                print(f"   ⚠️  {config_name}: {len(issues)} validation issues")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"      - {issue}")
                if len(issues) > 3:
                    print(f"      ... and {len(issues) - 3} more")
            else:
                print(f"   ✅ {config_name}: Valid configuration")
                
        except Exception as e:
            print(f"   ❌ {config_name}: Validation failed - {e}")
    
    # Demonstrate environment variable handling
    print(f"\n🌍 Environment Variable Handling:")
    
    # Create test config with various env var patterns
    test_config = {
        'sources': {
            'test_source': {
                'type': 'postgres_fdw',
                'server_options': {
                    'host': '${TEST_DB_HOST}',
                    'port': '${TEST_DB_PORT:5432}',
                    'dbname': '${TEST_DB_NAME}',
                    'timeout': '${CONNECTION_TIMEOUT:30}'
                },
                'user_mapping': {
                    'options': {
                        'user': '${TEST_DB_USER}',
                        'password': '${TEST_DB_PASSWORD:}'
                    }
                }
            }
        }
    }
    
    # Set some test environment variables
    test_env = {
        'TEST_DB_HOST': 'test.example.com',
        'TEST_DB_NAME': 'test_database',
        'TEST_DB_USER': 'test_user'
        # Deliberately omit TEST_DB_PORT and TEST_DB_PASSWORD to test defaults
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    # Process environment variables
    processed_config = config_manager._substitute_env_vars(test_config)
    server_opts = processed_config['sources']['test_source']['server_options']
    user_opts = processed_config['sources']['test_source']['user_mapping']['options']
    
    print(f"   Original: host='${{TEST_DB_HOST}}' -> Processed: host='{server_opts['host']}'")
    print(f"   Original: port='${{TEST_DB_PORT:5432}}' -> Processed: port='{server_opts['port']}'")
    print(f"   Original: dbname='${{TEST_DB_NAME}}' -> Processed: dbname='{server_opts['dbname']}'")
    print(f"   Original: user='${{TEST_DB_USER}}' -> Processed: user='{user_opts['user']}'")
    print(f"   Original: password='${{TEST_DB_PASSWORD:}}' -> Processed: password='{user_opts['password']}'")
    
    # Clean up test environment variables
    for key in test_env:
        if key in os.environ:
            del os.environ[key]
    
    # Performance benchmarking
    print(f"\n⚡ Configuration Performance Benchmarking:")
    
    performance_results = []
    
    for config_name in available_configs:
        try:
            start_time = time.time()
            config = config_manager.load_config(config_name)
            load_time = time.time() - start_time
            
            config_size = len(str(config))
            source_count = len(config.get('sources', {}))
            table_count = sum(len(src.get('tables', [])) for src in config.get('sources', {}).values())
            
            performance_results.append({
                'name': config_name,
                'load_time': load_time,
                'config_size': config_size,
                'sources': source_count,
                'tables': table_count,
                'tables_per_second': table_count / load_time if load_time > 0 else 0
            })
            
        except Exception as e:
            print(f"   ❌ {config_name}: Performance test failed - {e}")
    
    # Sort by complexity (table count)
    performance_results.sort(key=lambda x: x['tables'], reverse=True)
    
    print(f"   📊 Configuration Load Performance:")
    print(f"   {'Name':<25} {'Load Time':<10} {'Sources':<8} {'Tables':<7} {'Tables/s':<9}")
    print(f"   {'-'*25} {'-'*10} {'-'*8} {'-'*7} {'-'*9}")
    
    for result in performance_results:
        print(f"   {result['name']:<25} {result['load_time']:.3f}s    {result['sources']:<8} {result['tables']:<7} {result['tables_per_second']:.1f}")
    
    if performance_results:
        avg_load_time = sum(r['load_time'] for r in performance_results) / len(performance_results)
        total_tables = sum(r['tables'] for r in performance_results)
        total_load_time = sum(r['load_time'] for r in performance_results)
        
        print(f"\n   Summary:")
        print(f"   Average load time: {avg_load_time:.3f}s")
        print(f"   Total tables processed: {total_tables}")
        print(f"   Overall throughput: {total_tables / total_load_time:.1f} tables/second")


async def example_4_production_deployment_patterns():
    """Example 4: Production deployment patterns and best practices."""
    print("\n" + "="*70)
    print("🚀 Example 4: Production Deployment Patterns")
    print("="*70)
    
    # Demonstrate production configuration patterns
    production_patterns = {
        'High Availability': {
            'description': 'Multiple read replicas with failover',
            'config_template': {
                'sources': {
                    'primary_db': {
                        'type': 'postgres_fdw',
                        'server_options': {
                            'host': '${PRIMARY_DB_HOST}',
                            'port': '5432',
                            'dbname': 'production'
                        },
                        'enabled': True
                    },
                    'read_replica_1': {
                        'type': 'postgres_fdw',
                        'server_options': {
                            'host': '${REPLICA_1_HOST}',
                            'port': '5432',
                            'dbname': 'production'
                        },
                        'enabled': True
                    },
                    'read_replica_2': {
                        'type': 'postgres_fdw',
                        'server_options': {
                            'host': '${REPLICA_2_HOST}',
                            'port': '5432',
                            'dbname': 'production'
                        },
                        'enabled': True
                    }
                }
            },
            'benefits': ['Load distribution', 'Failover capability', 'Geographic distribution']
        },
        
        'Multi-Environment': {
            'description': 'Separate configurations for dev/staging/prod',
            'config_template': {
                'global_settings': {
                    'connection_timeout': '${CONNECTION_TIMEOUT:30}',
                    'query_timeout': '${QUERY_TIMEOUT:600}',
                    'enable_push_down': '${ENABLE_PUSH_DOWN:true}'
                },
                'sources': {
                    'app_database': {
                        'type': 'postgres_fdw',
                        'server_options': {
                            'host': '${APP_DB_HOST}',
                            'port': '${APP_DB_PORT:5432}',
                            'dbname': '${APP_DB_NAME}',
                            'use_remote_estimate': '${USE_REMOTE_ESTIMATE:true}'
                        }
                    }
                }
            },
            'benefits': ['Environment isolation', 'Configuration flexibility', 'Consistent deployment']
        },
        
        'Security First': {
            'description': 'Encrypted connections and credential management',
            'config_template': {
                'sources': {
                    'secure_db': {
                        'type': 'postgres_fdw',
                        'server_options': {
                            'host': '${SECURE_DB_HOST}',
                            'port': '5432',
                            'dbname': 'secure_prod',
                            'sslmode': 'require',
                            'sslcert': '/etc/ssl/certs/client.crt',
                            'sslkey': '/etc/ssl/private/client.key',
                            'sslrootcert': '/etc/ssl/certs/ca.crt'
                        },
                        'user_mapping': {
                            'options': {
                                'user': '${SECURE_DB_USER}',
                                'password': 'credential:secure_db_password'  # From credential manager
                            }
                        }
                    }
                }
            },
            'benefits': ['Encrypted communication', 'Secure credential storage', 'Certificate-based auth']
        },
        
        'Performance Optimized': {
            'description': 'Tuned for high-performance analytics',
            'config_template': {
                'global_settings': {
                    'connection_timeout': 60,
                    'query_timeout': 3600,  # 1 hour for long analytics queries
                    'enable_push_down': True
                },
                'sources': {
                    'analytics_db': {
                        'type': 'postgres_fdw',
                        'server_options': {
                            'host': '${ANALYTICS_DB_HOST}',
                            'port': '5432',
                            'dbname': 'analytics',
                            'use_remote_estimate': True,
                            'fdw_startup_cost': 10.0,  # Low startup cost
                            'fdw_tuple_cost': 0.01,    # Low per-tuple cost
                            'fetch_size': 10000        # Large fetch size
                        },
                        'tables': [
                            {
                                'name': 'fact_sales',
                                'options': {
                                    'use_remote_estimate': True,
                                    'fetch_size': 50000  # Even larger for fact tables
                                }
                            }
                        ]
                    }
                }
            },
            'benefits': ['Optimized query planning', 'Bulk data transfer', 'Reduced round trips']
        }
    }
    
    # Show each pattern
    for pattern_name, pattern_info in production_patterns.items():
        print(f"\n📋 Pattern: {pattern_name}")
        print(f"   Description: {pattern_info['description']}")
        print(f"   Benefits: {', '.join(pattern_info['benefits'])}")
        
        # Calculate complexity metrics
        config = pattern_info['config_template']
        source_count = len(config.get('sources', {}))
        has_global_settings = 'global_settings' in config
        uses_credentials = any(
            'password' in src.get('user_mapping', {}).get('options', {})
            for src in config.get('sources', {}).values()
        )
        
        print(f"   Complexity: {source_count} sources, "
              f"{'global settings' if has_global_settings else 'no global settings'}, "
              f"{'credential management' if uses_credentials else 'basic auth'}")
    
    # Deployment checklist
    print(f"\n✅ Production Deployment Checklist:")
    
    checklist_items = [
        ('Security', [
            'SSL/TLS encryption enabled for all connections',
            'Credentials stored in secure credential manager',
            'Network access restricted to authorized IPs',
            'Certificate-based authentication where possible',
            'Regular credential rotation implemented'
        ]),
        ('Performance', [
            'Connection pooling configured appropriately',
            'Query timeouts set for workload requirements',
            'Push-down optimization enabled',
            'Fetch sizes tuned for data volume',
            'Remote estimation enabled for cost-based optimization'
        ]),
        ('Reliability', [
            'Multiple read replicas configured',
            'Automatic failover mechanisms in place',
            'Health checks and monitoring configured',
            'Retry logic implemented for transient failures',
            'Circuit breaker patterns for external dependencies'
        ]),
        ('Monitoring', [
            'Query performance metrics collected',
            'Connection pool status monitored',
            'Error rates and failure patterns tracked',
            'Resource utilization alerts configured',
            'Data freshness monitoring implemented'
        ]),
        ('Maintenance', [
            'Configuration version control established',
            'Automated deployment pipelines configured',
            'Database schema change management process',
            'Performance baseline and regression testing',
            'Documentation and runbooks maintained'
        ])
    ]
    
    for category, items in checklist_items:
        print(f"\n   {category}:")
        for item in items:
            print(f"     ☐ {item}")
    
    # Configuration management recommendations
    print(f"\n🛠️  Configuration Management Recommendations:")
    
    recommendations = [
        {
            'area': 'Version Control',
            'recommendation': 'Store configurations in Git with proper branching strategy',
            'example': 'config/environments/production.yaml, config/environments/staging.yaml'
        },
        {
            'area': 'Environment Variables',
            'recommendation': 'Use environment-specific variable files',
            'example': '.env.production, .env.staging with validation'
        },
        {
            'area': 'Secrets Management',
            'recommendation': 'Integrate with enterprise secret managers',
            'example': 'HashiCorp Vault, AWS Secrets Manager, Azure Key Vault'
        },
        {
            'area': 'Configuration Validation',
            'recommendation': 'Implement pre-deployment configuration testing',
            'example': 'Schema validation, connectivity tests, permission checks'
        },
        {
            'area': 'Deployment Automation',
            'recommendation': 'Automate configuration deployment with rollback capability',
            'example': 'CI/CD pipelines with configuration diff analysis'
        }
    ]
    
    for rec in recommendations:
        print(f"\n   {rec['area']}:")
        print(f"     Recommendation: {rec['recommendation']}")
        print(f"     Example: {rec['example']}")
    
    print(f"\n💡 Next Steps for Production Deployment:")
    print(f"   1. Choose appropriate deployment pattern for your architecture")
    print(f"   2. Implement comprehensive monitoring and alerting")
    print(f"   3. Set up automated testing for configuration changes")
    print(f"   4. Establish incident response procedures")
    print(f"   5. Plan for disaster recovery and business continuity")


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
    """Run all configuration demo examples."""
    print("🔧 Configuration-Driven Demo - PostgreSQL FDW Data Connector")
    print("=" * 70)
    print("Demonstrates production-ready configuration management for")
    print("multi-source data integration using YAML configurations")
    print()
    
    setup_environment()
    
    examples = [
        ("PostgreSQL FDW Configuration", example_1_postgres_fdw_configuration),
        ("Mixed Sources Configuration", example_2_mixed_sources_configuration),
        ("Dynamic Configuration Management", example_3_dynamic_configuration_management),
        ("Production Deployment Patterns", example_4_production_deployment_patterns)
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
    print("🎉 Configuration Demo Completed!")
    print()
    print("💡 Key configuration management features demonstrated:")
    print("  🔧 YAML-based declarative configuration")
    print("  🌍 Environment variable substitution")
    print("  🔍 Configuration validation and testing")
    print("  🚀 Production deployment patterns")
    print("  ⚡ Performance optimization strategies")
    print("  🔒 Security best practices")
    print()
    print("🌟 Next steps:")
    print("  1. Customize configuration files for your data sources")
    print("  2. Set up environment-specific variable files")
    print("  3. Implement configuration validation in CI/CD")
    print("  4. Deploy with appropriate monitoring and alerting")
    print("="*70)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Configuration demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo error details")