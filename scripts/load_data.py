#!/usr/bin/env python3
"""Simple utility for loading data from connector pickle files."""

import argparse
import sys
from pathlib import Path
from pprint import pprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Use secure serialization instead of unsafe pickle
from instant_connector import load_data_connector  # Secure version
# from instant_connector.pickle_manager import load_data_connector  # UNSAFE - DEPRECATED


def main():
    """Main loading script."""
    parser = argparse.ArgumentParser(
        description='Load data from instant data connector pickle files'
    )
    
    parser.add_argument(
        'pickle_file',
        type=str,
        help='Path to connector pickle file'
    )
    parser.add_argument(
        '-d', '--datasets',
        nargs='+',
        help='Specific datasets to load (default: all)'
    )
    parser.add_argument(
        '-i', '--info',
        action='store_true',
        help='Show metadata information'
    )
    parser.add_argument(
        '-s', '--summary',
        action='store_true',
        help='Show summary of loaded data'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=0,
        help='Show first N rows of each dataset'
    )
    parser.add_argument(
        '--describe',
        action='store_true',
        help='Show statistical description of numeric columns'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    pickle_path = Path(args.pickle_file)
    if not pickle_path.exists():
        print(f"❌ Error: File not found: {pickle_path}")
        sys.exit(1)
    
    try:
        # Load data
        import time
        start_time = time.time()
        
        data = load_data_connector(str(pickle_path))
        
        # Filter datasets if specified
        if args.datasets:
            data = {name: df for name, df in data.items() if name in args.datasets}
        
        load_time = time.time() - start_time
        
        if args.info:
            print("\n📊 Connector Information:")
            print("-" * 50)
            print(f"Load time: {load_time:.3f} seconds")
            print(f"Total datasets: {len(data)}")
            print(f"File size: {pickle_path.stat().st_size / 1024**2:.2f} MB")
            print("-" * 50)
        
        print(f"\n✅ Loaded {len(data)} datasets from {pickle_path.name}")
        
        # Show summary
        if args.summary or args.head or args.describe:
            for name, df in data.items():
                print(f"\n📁 Dataset: {name}")
                print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                if args.summary:
                    print(f"   Columns: {', '.join(df.columns[:10])}")
                    if len(df.columns) > 10:
                        print(f"            ... and {len(df.columns) - 10} more")
                    
                    print("   Data types:")
                    for dtype, count in df.dtypes.value_counts().items():
                        print(f"     - {dtype}: {count} columns")
                
                if args.head and args.head > 0:
                    print(f"\n   First {args.head} rows:")
                    print(df.head(args.head))
                
                if args.describe:
                    print("\n   Statistical summary:")
                    print(df.describe())
        
        print("\n💡 To use this data in your code:")
        print(f"   # Use secure serialization instead of unsafe pickle
from instant_connector import load_data_connector  # Secure version
# from instant_connector.pickle_manager import load_data_connector  # UNSAFE - DEPRECATED")
        print(f"   data = load_data_connector('{args.pickle_file}')")
        print(f"   df = data['{list(data.keys())[0]}']  # Get first dataset")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()