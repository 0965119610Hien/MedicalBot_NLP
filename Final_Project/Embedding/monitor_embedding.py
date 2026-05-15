#!/usr/bin/env python3
"""Monitor embedding.py progress in real-time"""
import subprocess
import time
import sys
import os
from pathlib import Path

output_dir = Path("./medical_search_index")
log_file = output_dir / "embedding.log"

print(f"Monitoring embedding process...")
print(f"Output directory: {output_dir}")
print(f"Log file: {log_file}")
print("-" * 80)

# Watch for key completion files
completed_files = {
    'chroma_db': output_dir / "chroma_db" / "chroma.sqlite3",
    'bm25_index': output_dir / "bm25_index.pkl",
    'chunk_metadata': output_dir / "chunk_metadata.json",
    'abbreviation_map': output_dir / "abbreviation_map.json"
}

print("\nExpected completion files:")
for name, path in completed_files.items():
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists} {name}: {path}")

print("\nPipeline stages:")
print("  1. Preparation (normalize & tokenize) - ~30 seconds")
print("  2. E5 Encoding on GPU - ~35 minutes")
print("  3. ChromaDB Storage (batch_size=128) - ~5 minutes")
print("  4. BM25 Index Building - ~2 minutes")
print("  5. Metadata Saving - ~1 minute")
print("\nTotal estimated time: ~45 minutes")
print("-" * 80)

# Monitor every 30 seconds
last_check = time.time()
while True:
    try:
        if time.time() - last_check >= 30:
            print(f"\n[{time.strftime('%H:%M:%S')}] Status check:")
            
            # Check file sizes for progress indication
            for name, path in completed_files.items():
                if path.exists():
                    try:
                        size = path.stat().st_size / (1024 * 1024)
                        print(f"  ✓ {name}: {size:.1f} MB")
                    except:
                        print(f"  ✓ {name}: exists")
                else:
                    print(f"  ✗ {name}: not created yet")
            
            last_check = time.time()
        
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
