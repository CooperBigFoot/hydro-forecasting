#!/usr/bin/env python3
"""
Script to check validity of all parquet files in a directory.
Attempts to read each file with pandas and reports any failures.
Also checks file permissions.
"""

import os
import pandas as pd
from pathlib import Path
import stat
import sys
from datetime import datetime

def check_file_permissions(file_path):
    """Check if file has unusual permissions."""
    try:
        file_stat = os.stat(file_path)
        mode = file_stat.st_mode
        
        # Check if file is readable by owner
        if not (mode & stat.S_IRUSR):
            return False, "Not readable by owner"
        
        # Check if file has any content
        if file_stat.st_size == 0:
            return False, "File is empty (0 bytes)"
            
        return True, "OK"
    except Exception as e:
        return False, f"Permission check error: {str(e)}"

def check_parquet_file(file_path):
    """Try to read a parquet file and return status."""
    try:
        # Try to read the parquet file
        df = pd.read_parquet(file_path)
        
        # Basic validation
        if df.empty:
            return False, "DataFrame is empty", None
        
        return True, f"OK - {len(df)} rows, {len(df.columns)} columns", df.shape
    except Exception as e:
        return False, f"Read error: {type(e).__name__}: {str(e)}", None

def main():
    # Directory to check
    parquet_dir = Path("/Users/cooper/Desktop/CaravanifyParquet/CA/post_processed/timeseries/csv/CA/")
    
    if not parquet_dir.exists():
        print(f"Error: Directory does not exist: {parquet_dir}")
        sys.exit(1)
    
    # Find all parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No .parquet files found in {parquet_dir}")
        sys.exit(0)
    
    print(f"Found {len(parquet_files)} parquet files in {parquet_dir}")
    print("=" * 80)
    
    # Track statistics
    total_files = len(parquet_files)
    valid_files = 0
    invalid_files = 0
    permission_issues = 0
    error_details = []
    
    # Check each file
    for i, file_path in enumerate(parquet_files, 1):
        file_name = file_path.name
        print(f"\n[{i}/{total_files}] Checking: {file_name}")
        
        # Check permissions
        perm_ok, perm_msg = check_file_permissions(file_path)
        if not perm_ok:
            print(f"  ❌ Permission issue: {perm_msg}")
            permission_issues += 1
            error_details.append({
                'file': file_name,
                'error_type': 'permission',
                'error': perm_msg
            })
        else:
            print(f"  ✓ Permissions: {perm_msg}")
        
        # Check if we can read the parquet file
        read_ok, read_msg, shape = check_parquet_file(file_path)
        if read_ok:
            print(f"  ✓ Read status: {read_msg}")
            valid_files += 1
        else:
            print(f"  ❌ Read failed: {read_msg}")
            invalid_files += 1
            error_details.append({
                'file': file_name,
                'error_type': 'read',
                'error': read_msg
            })
    
    # Summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print(f"Total files checked: {total_files}")
    print(f"Valid files: {valid_files} ({valid_files/total_files*100:.1f}%)")
    print(f"Invalid files: {invalid_files} ({invalid_files/total_files*100:.1f}%)")
    print(f"Permission issues: {permission_issues}")
    
    # Detailed error report
    if error_details:
        print("\n" + "=" * 80)
        print("ERROR DETAILS")
        print("=" * 80)
        
        # Group errors by type
        read_errors = [e for e in error_details if e['error_type'] == 'read']
        perm_errors = [e for e in error_details if e['error_type'] == 'permission']
        
        if read_errors:
            print(f"\nFiles with read errors ({len(read_errors)}):")
            for err in read_errors:
                print(f"  - {err['file']}: {err['error']}")
        
        if perm_errors:
            print(f"\nFiles with permission issues ({len(perm_errors)}):")
            for err in perm_errors:
                print(f"  - {err['file']}: {err['error']}")
        
        # Save error report
        error_report_path = Path("parquet_check_errors.txt")
        with open(error_report_path, 'w') as f:
            f.write(f"Parquet File Check Report - {datetime.now()}\n")
            f.write(f"Directory: {parquet_dir}\n")
            f.write("=" * 80 + "\n\n")
            
            for err in error_details:
                f.write(f"File: {err['file']}\n")
                f.write(f"Error Type: {err['error_type']}\n")
                f.write(f"Error: {err['error']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"\nError report saved to: {error_report_path.absolute()}")
    
    # Exit with appropriate code
    if invalid_files > 0 or permission_issues > 0:
        sys.exit(1)
    else:
        print("\n✅ All parquet files are valid and readable!")
        sys.exit(0)

if __name__ == "__main__":
    main()