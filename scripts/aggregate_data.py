#!/usr/bin/env python3
"""Main script for aggregating data from multiple sources using FDW-based connector."""

import argparse
import asyncio
import logging
import sys
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import InstantDataConnector, LegacyInstantDataConnector, PickleManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def run_fdw_mode(args):
    """Run using the new FDW-based connector."""
    print("🚀 Using FDW-based InstantDataConnector (recommended)")
    
    # Determine PostgreSQL connection config
    postgres_config = {}
    if args.postgres_host:
        postgres_config['host'] = args.postgres_host
    if args.postgres_port:
        postgres_config['port'] = args.postgres_port
    if args.postgres_database:
        postgres_config['database'] = args.postgres_database
    if args.postgres_user:
        postgres_config['username'] = args.postgres_user
    if args.postgres_password:
        postgres_config['password'] = args.postgres_password
    
    # Create FDW-based connector
    async with InstantDataConnector(
        config_path=args.config,
        postgres_config=postgres_config,
        enable_caching=not args.no_cache
    ) as connector:
        
        # Setup FDW infrastructure
        await connector.setup_fdw_infrastructure(
            force_refresh=args.force_refresh,
            validate_connections=not args.skip_validation
        )
        
        # Handle different operations
        if args.command == 'list-tables':
            tables = await connector.list_available_tables(refresh=True)
            print(f"\n📋 Available tables ({len(tables)}):")
            for name, info in tables.items():
                print(f"  • {name} ({info.get('source_type', 'unknown')}): {info.get('description', 'No description')}")
        
        elif args.command == 'describe-table':
            if not args.table_name:
                print("❌ --table-name is required for describe-table command")
                sys.exit(1)
            
            schema = await connector.get_table_schema(args.table_name)
            print(f"\n📊 Schema for table '{args.table_name}':")
            for col in schema:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                print(f"  • {col['name']}: {col['type']} {nullable}")
        
        elif args.command == 'query':
            if not args.sql_query:
                print("❌ --sql-query is required for query command")
                sys.exit(1)
            
            result = await connector.execute_query(args.sql_query)
            print(f"\n📊 Query results ({len(result)} rows):")
            print(result.head(args.preview_rows) if args.preview_rows else result)
            
            # Save to file if requested
            if args.output:
                if args.output.endswith('.csv'):
                    result.to_csv(args.output, index=False)
                elif args.output.endswith('.json'):
                    result.to_json(args.output, orient='records')
                else:
                    # Save as pickle
                    result.to_pickle(args.output)
                print(f"💾 Saved results to {args.output}")
        
        elif args.command == 'extract-table':
            if not args.table_name:
                print("❌ --table-name is required for extract-table command")
                sys.exit(1)
            
            # Build filters if provided
            filters = {}
            if args.filters:
                for filter_expr in args.filters:
                    if '=' in filter_expr:
                        key, value = filter_expr.split('=', 1)
                        filters[key.strip()] = value.strip()
            
            result = await connector.lazy_load_table(
                args.table_name,
                filters=filters if filters else None,
                columns=args.columns,
                limit=args.limit,
                order_by=args.order_by
            )
            
            print(f"\n📊 Extracted {len(result)} rows from table '{args.table_name}'")
            print(result.head(args.preview_rows) if args.preview_rows else result)
            
            # Save to file if requested
            if args.output:
                if args.output.endswith('.csv'):
                    result.to_csv(args.output, index=False)
                elif args.output.endswith('.json'):
                    result.to_json(args.output, orient='records')
                else:
                    result.to_pickle(args.output)
                print(f"💾 Saved results to {args.output}")
        
        elif args.command == 'health-check':
            health = await connector.health_check()
            print(f"\n🏥 Health Check Results:")
            print(f"Overall Status: {'✅ Healthy' if health['overall_healthy'] else '❌ Unhealthy'}")
            for component, status in health['components'].items():
                status_icon = "✅" if status['healthy'] else "❌"
                print(f"  {status_icon} {component}: {status['details']}")
        
        elif args.command == 'refresh-tables':
            tables_to_refresh = [args.table_name] if args.table_name else None
            results = await connector.refresh_virtual_tables(tables_to_refresh)
            
            print(f"\n🔄 Table refresh results:")
            for table, success in results.items():
                status = "✅ Success" if success else "❌ Failed"
                print(f"  {status} {table}")


def run_legacy_mode(args):
    """Run using the legacy connector with deprecation warning."""
    warnings.warn(
        "Legacy mode is deprecated and will be removed in v1.0.0. "
        "Please migrate to FDW-based mode using --use-fdw flag.",
        DeprecationWarning
    )
    
    print("⚠️  Using legacy InstantDataConnector (deprecated)")
    
    # Create legacy aggregator
    aggregator = LegacyInstantDataConnector(config_path=args.config)
    
    # Add sources from command line (legacy way)
    if args.database:
        if len(args.database) >= 2:
            connection_string = args.database[0]
            query = ' '.join(args.database[1:])
            
            params = {'connection_string': connection_string}
            if 'postgresql://' in connection_string:
                params['db_type'] = 'postgresql'
            elif 'mysql://' in connection_string:
                params['db_type'] = 'mysql'
            elif 'sqlite://' in connection_string:
                params['db_type'] = 'sqlite'
            
            aggregator.add_database_source(
                'cli_database',
                params,
                queries={'query': query}
            )
    
    if args.files:
        for i, file_path in enumerate(args.files):
            aggregator.add_file_source(f'cli_file_{i}', file_path)
    
    if args.api:
        base_url, endpoint, method = args.api
        aggregator.add_api_source(
            'cli_api',
            base_url,
            endpoints={'endpoint': {'endpoint': endpoint, 'method': method}}
        )
    
    # Check if any sources were added
    if not aggregator.sources and not args.config:
        print("❌ No data sources specified. Use --config, --database, --files, or --api")
        sys.exit(1)
    
    # Run legacy aggregation pipeline
    try:
        aggregator.aggregate_all()
        save_stats = aggregator.save_pickle(args.output)
        
        # Print summary
        print(f"\n✅ Legacy data aggregation complete!")
        print(f"📁 Output file: {save_stats.get('file_path', args.output)}")
        if 'file_size_mb' in save_stats:
            print(f"📊 File size: {save_stats['file_size_mb']:.2f} MB")
        if 'compression_ratio' in save_stats:
            print(f"🗜️  Compression ratio: {save_stats['compression_ratio']:.2f}x")
        
    except Exception as e:
        logging.error(f"Legacy aggregation failed: {e}")
        sys.exit(1)


def main():
    """Main CLI script with support for both legacy and FDW modes."""
    parser = argparse.ArgumentParser(
        description='Aggregate data from multiple sources using InstantDataConnector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FDW mode (recommended) - list available tables
  %(prog)s list-tables --config config.yaml --postgres-host localhost
  
  # FDW mode - extract data from a table
  %(prog)s extract-table --table-name users --output users.csv --limit 1000
  
  # FDW mode - run custom SQL query
  %(prog)s query --sql-query "SELECT * FROM users WHERE age > 25" --output results.json
  
  # Legacy mode (deprecated)
  %(prog)s legacy --config legacy_config.yaml output.pkl
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--use-fdw',
        action='store_true',
        help='Use FDW-based connector (recommended, default for most commands)'
    )
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy connector (deprecated)'
    )
    
    # Command selection for FDW mode
    parser.add_argument(
        'command',
        nargs='?',
        choices=['list-tables', 'describe-table', 'query', 'extract-table', 'health-check', 'refresh-tables', 'legacy'],
        help='Command to execute'
    )
    
    # Configuration
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Configuration file path (YAML)'
    )
    
    # PostgreSQL connection parameters for FDW mode
    parser.add_argument(
        '--postgres-host',
        type=str,
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )
    parser.add_argument(
        '--postgres-port',
        type=int,
        default=5432,
        help='PostgreSQL port (default: 5432)'
    )
    parser.add_argument(
        '--postgres-database',
        type=str,
        default='postgres',
        help='PostgreSQL database (default: postgres)'
    )
    parser.add_argument(
        '--postgres-user',
        type=str,
        help='PostgreSQL username'
    )
    parser.add_argument(
        '--postgres-password',
        type=str,
        help='PostgreSQL password'
    )
    
    # FDW-specific arguments
    parser.add_argument(
        '--table-name',
        type=str,
        help='Table name for table-specific operations'
    )
    parser.add_argument(
        '--sql-query',
        type=str,
        help='SQL query to execute'
    )
    parser.add_argument(
        '--columns',
        nargs='+',
        help='Columns to select'
    )
    parser.add_argument(
        '--filters',
        nargs='+',
        help='Filters in format column=value'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of rows to return'
    )
    parser.add_argument(
        '--order-by',
        type=str,
        help='Column to order by'
    )
    parser.add_argument(
        '--preview-rows',
        type=int,
        default=10,
        help='Number of rows to preview (default: 10, 0 for all)'
    )
    
    # Output arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (.csv, .json, or .pkl)'
    )
    
    # Control arguments
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh of FDW infrastructure'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip connection validation during setup'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable result caching'
    )
    
    # Legacy mode arguments (for backward compatibility)
    parser.add_argument(
        '--database',
        nargs='+',
        metavar=('CONNECTION_STRING', 'QUERY'),
        help='[Legacy] Database source: connection_string "query"'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='[Legacy] File paths to aggregate'
    )
    parser.add_argument(
        '--api',
        nargs=3,
        metavar=('BASE_URL', 'ENDPOINT', 'METHOD'),
        help='[Legacy] API source: base_url endpoint method'
    )
    
    # Other arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Determine mode
    if args.command == 'legacy' or args.legacy:
        # Legacy mode
        if not args.output:
            parser.error('Output file path is required for legacy mode')
        run_legacy_mode(args)
    else:
        # FDW mode (default for new features)
        if not args.command:
            parser.error('Command is required. Use --help for available commands.')
        
        # Run FDW mode
        try:
            asyncio.run(run_fdw_mode(args))
        except KeyboardInterrupt:
            print("\n🛑 Operation cancelled by user")
            sys.exit(1)
        except Exception as e:
            logging.error(f"FDW operation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()