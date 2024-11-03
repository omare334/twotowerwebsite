#!/bin/sh

# Wait for MinIO to be ready
until mc alias set local http://localhost:9000 omareweis ommaha260801; do
    echo "Waiting for MinIO to be available..."
    sleep 2
done

# Create a bucket named "mybucket" (or any name you prefer)
mc mb local/mybucket

# Optional: Set bucket policy (for example, public read access)
mc policy set public local/mybucket
