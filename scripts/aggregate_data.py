#!/usr/bin/env python3
"""Main script for aggregating data from multiple sources."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instant_connector import DataAggregator, MLOptimizer, PickleManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main aggregation script."""
    parser = argparse.ArgumentParser(
        description='Aggregate data from multiple sources into ML-ready pickle files'
    )
    
    # Input/output arguments
    parser.add_argument(
        'output',
        type=str,
        help='Output pickle file path'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Configuration file path (YAML)'
    )
    
    # Source arguments
    parser.add_argument(
        '--database',
        nargs='+',
        metavar=('CONNECTION_STRING', 'QUERY'),
        help='Database source: connection_string "query"'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='File paths to aggregate'
    )
    parser.add_argument(
        '--api',
        nargs=3,
        metavar=('BASE_URL', 'ENDPOINT', 'METHOD'),
        help='API source: base_url endpoint method'
    )
    
    # Optimization arguments
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Skip ML optimization'
    )
    parser.add_argument(
        '--handle-missing',
        choices=['auto', 'drop', 'mean', 'median', 'mode', 'forward_fill'],
        default='auto',
        help='Strategy for handling missing values'
    )
    parser.add_argument(
        '--encode-categorical',
        choices=['auto', 'label', 'onehot', 'none'],
        default='auto',
        help='Strategy for encoding categorical variables'
    )
    parser.add_argument(
        '--scale-numeric',
        choices=['standard', 'minmax', 'robust', 'none'],
        default=None,
        help='Strategy for scaling numeric features'
    )
    
    # Compression arguments
    parser.add_argument(
        '--compression',
        choices=['gzip', 'lz4', 'bz2', 'none'],
        default='lz4',
        help='Compression method for pickle file'
    )
    parser.add_argument(
        '--compression-level',
        type=int,
        default=0,
        help='Compression level (0-9 for gzip/bz2, 0-16 for lz4)'
    )
    
    # Other arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Create aggregator
    aggregator = DataAggregator(config_path=args.config)
    
    # Add sources from command line
    if args.database:
        if len(args.database) >= 2:
            connection_string = args.database[0]
            query = ' '.join(args.database[1:])
            
            # Parse connection string to get parameters
            # Simple parsing - in production use proper URL parsing
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
                {'query': query}
            )
    
    if args.files:
        aggregator.add_file_source('cli_files', args.files)
    
    if args.api:
        base_url, endpoint, method = args.api
        aggregator.add_api_source(
            'cli_api',
            base_url,
            {'endpoint': {'endpoint': endpoint, 'method': method}}
        )
    
    # Check if any sources were added
    if not aggregator.sources and not args.config:
        parser.error('No data sources specified. Use --config, --database, --files, or --api')
    
    # Set up optimization parameters
    optimizer_kwargs = {
        'handle_missing': args.handle_missing,
        'encode_categorical': args.encode_categorical if args.encode_categorical != 'none' else None,
        'scale_numeric': args.scale_numeric if args.scale_numeric != 'none' else None,
    }
    
    # Run aggregation pipeline
    try:
        save_stats = aggregator.aggregate_and_save(
            args.output,
            optimize=not args.no_optimize,
            compression=args.compression,
            compression_level=args.compression_level,
            optimizer_kwargs=optimizer_kwargs
        )
        
        # Print summary
        print(f"\n✅ Data aggregation complete!")
        print(f"📁 Output file: {save_stats['file_path']}")
        print(f"📊 File size: {save_stats['file_size_mb']:.2f} MB")
        print(f"🗜️  Compression ratio: {save_stats['compression_ratio']:.2f}x")
        print(f"⏱️  Save time: {save_stats['save_time_seconds']:.2f} seconds")
        print(f"🔑 Checksum: {save_stats['checksum']}")
        
    except Exception as e:
        logging.error(f"Aggregation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()