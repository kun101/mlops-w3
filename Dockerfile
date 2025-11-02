# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script into the container
COPY api.py .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using Gunicorn
# This is the command that will run when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]