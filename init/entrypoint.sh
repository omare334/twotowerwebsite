#!/bin/sh

# Start MinIO server in the background
minio server /data &

# Wait for MinIO to start up
# Instead of a fixed sleep time, we can use a loop to check if MinIO is ready
MINIO_READY=false
RETRIES=10
while [ $RETRIES -gt 0 ]; do
    # Check if MinIO's health endpoint is available
    if curl -s http://localhost:9000/minio/health/live > /dev/null; then
        MINIO_READY=true
        break
    fi
    echo "Waiting for MinIO to start..."
    sleep 2
    RETRIES=$((RETRIES-1))
done

if [ "$MINIO_READY" = true ]; then
    echo "MinIO is ready. Running initialization script."
    # Run your initialization script
    /init/init-minio.sh
else
    echo "MinIO did not start within the expected time."
    exit 1
fi

# Wait for MinIO server to stop (optional)
wait
