# Use the official Python 3.13 image
FROM python:3.13-slim

# Set the working directory
WORKDIR /src

# Copy the requirements file if you have one
COPY requirements.txt .

# Install any required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code to the working directory
COPY app/ .

# Set the command to run your script
CMD ["python", "ai_modelling/cnn_lstm.py"]
