# Use the official Python image as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Command to run your application
CMD ["streamlit", "run", "pipeline_searchengine.py", "--server.address=0.0.0.0", "--server.port=8501"]

