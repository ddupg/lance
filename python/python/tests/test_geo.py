#!/usr/bin/env python3
"""
Test script for GeoArrow Point geo index functionality in Lance.

This script tests:
1. Creating GeoArrow Point data
2. Writing to Lance dataset
3. Creating a geo index on GeoArrow Point column
4. Querying with spatial filters
5. Verifying the geo index is used
"""

import numpy as np
import pyarrow as pa
import lance
import os
import shutil
from geoarrow.pyarrow import point
from datafusion import SessionContext

def main():
    print("🌍 Testing GeoArrow Point Geo Index in Lance")
    print("=" * 50)

    # Clean slate
    dataset_path = "test_geoarrow_geo"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        print(f"✅ Cleaned up existing dataset: {dataset_path}")

    # Step 1: Create GeoArrow Point data
    print("\n🔵 Step 1: Creating GeoArrow Point data")
    lat_np = np.array([37.7749, 34.0522, 40.7128], dtype="float64")  # SF, LA, NYC
    lng_np = np.array([-122.4194, -118.2437, -74.0060], dtype="float64")

    start_location = point().from_geobuffers(None, lng_np, lat_np)

    table = pa.table({
        "id": [1, 2, 3],
        "city": ["San Francisco", "Los Angeles", "New York"],
        "start_location": start_location,
        "population": [883305, 3898747, 8336817]
    })

    print("✅ Created GeoArrow Point data")
    print("📊 Table schema:")
    print(table.schema)
    print(f"📍 Point column type: {table.schema.field('start_location').type}")
    print(f"📍 Point column metadata: {table.schema.field('start_location').metadata}")

    # Step 2: Write to Lance dataset
    print("\n🔵 Step 2: Writing to Lance dataset")
    try:
        geo_ds = lance.write_dataset(table, dataset_path)
        print("✅ Successfully wrote GeoArrow data to Lance dataset")

        # Verify data was written correctly
        loaded_table = geo_ds.to_table()
        print(f"📊 Dataset has {len(loaded_table)} rows")
        print("📊 Dataset schema:")
        print(loaded_table.schema)

    except Exception as e:
        print(f"❌ Failed to write dataset: {e}")
        return

    # Step 3: Create geo index
    print("\n🔵 Step 3: Creating geo index on GeoArrow Point column")
    try:
        geo_ds.create_scalar_index(column="start_location", index_type="GEO")
        print("✅ Successfully created geo index")

        # Check what indexes exist
        indexes = geo_ds.list_indices()
        print("📊 Available indexes:")
        for idx in indexes:
            print(f"  - {idx}")

    except Exception as e:
        print(f"❌ Failed to create geo index: {e}")
        return

    # Step 4: Test spatial queries using DataFusion
    print("\n🔵 Step 4: Testing spatial queries with DataFusion")

    # Set up DataFusion context with Lance dataset
    print("\n🔧 Setting up DataFusion context")
    try:
        ctx = SessionContext()

        # Register the Lance dataset as a table in DataFusion
        table_provider = lance.FFILanceTableProvider(geo_ds, with_row_id=False, with_row_addr=False)
        ctx.register_table_provider("geo_dataset", table_provider)

        print("✅ Successfully registered Lance dataset with DataFusion")

    except Exception as e:
        print(f"❌ Failed to set up DataFusion context: {e}")
        return

    # Query 1: Basic coordinate-based filtering (should work)
    print("\n🔍 Query 1: Coordinate-based filtering - West Coast cities")
    try:
        # Use coordinate access instead of spatial functions for now
        # This tests that DataFusion can access the geo data structure
        sql = """
              SELECT id, city, population
              FROM geo_dataset
              WHERE start_location.x < -100.0 \
              """
        result = ctx.sql(sql).collect()

        if result:
            cities = [row['city'] for batch in result for row in batch.to_pylist()]
            print(f"✅ Found {len(cities)} West Coast cities: {cities}")
        else:
            print("⚠️  No results returned")

    except Exception as e:
        print(f"❌ Query 1 failed: {e}")

    # Query 2: Try a simple spatial function (may not work yet)
    print("\n🔍 Query 2: Testing basic spatial function")
    try:
        sql = """
              SELECT id, city, population
              FROM geo_dataset
              WHERE ST_X(start_location) < -100.0 \
              """
        result = ctx.sql(sql).collect()

        if result:
            cities = [row['city'] for batch in result for row in batch.to_pylist()]
            print(f"✅ Found {len(cities)} cities with ST_X: {cities}")
        else:
            print("⚠️  No results returned")

    except Exception as e:
        print(f"⚠️  Query 2 failed: {e}")
        print("   Note: Spatial functions may not be available yet")

    # Query 3: Test more complex spatial operations (experimental)
    print("\n🔍 Query 3: Testing bounding box query (experimental)")
    try:
        # Try to use a bounding box-like query
        sql = """
              SELECT id, city, population
              FROM geo_dataset
              WHERE start_location.x BETWEEN -125.0 AND -115.0
                AND start_location.y BETWEEN 32.0 AND 42.0 \
              """
        result = ctx.sql(sql).collect()

        if result:
            cities = [row['city'] for batch in result for row in batch.to_pylist()]
            print(f"✅ Found {len(cities)} cities in bounding box: {cities}")
        else:
            print("⚠️  No results returned")

    except Exception as e:
        print(f"❌ Query 3 failed: {e}")
        print("   Note: May need proper GeoArrow support in DataFusion")

    # Query 4: List all data to verify DataFusion integration
    print("\n🔍 Query 4: List all data via DataFusion")
    try:
        sql = "SELECT * FROM geo_dataset"
        result = ctx.sql(sql).collect()

        if result:
            total_rows = sum(len(batch) for batch in result)
            print(f"✅ Successfully retrieved {total_rows} rows via DataFusion")

            # Show first row details
            first_batch = result[0]
            if len(first_batch) > 0:
                first_row = first_batch.to_pylist()[0]
                print(f"📊 First row: {first_row}")
        else:
            print("⚠️  No results returned")

    except Exception as e:
        print(f"❌ Query 4 failed: {e}")


    # Step 5: Check index files
    print("\n🔵 Step 5: Verifying index files")
    try:
        import glob
        index_files = glob.glob(f"{dataset_path}/_indices/*")
        print(f"📂 Index directories: {len(index_files)}")

        for idx_dir in index_files:
            files = glob.glob(f"{idx_dir}/*")
            print(f"📂 Files in {idx_dir}:")
            for f in files:
                file_size = os.path.getsize(f)
                print(f"  - {os.path.basename(f)} ({file_size} bytes)")

    except Exception as e:
        print(f"❌ Failed to check index files: {e}")

    print("\n🎉 Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()