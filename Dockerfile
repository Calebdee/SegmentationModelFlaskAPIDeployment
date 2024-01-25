# Use an official Python runtime for 3.7.1 as parent image
FROM python:3.7-slim

# Set working directory for container
WORKDIR /app

# Copy current directory contents into our app container
COPY . /app

# Install needed packages
RUN pip install -r requirements.txt

# Make port 80 and 1313 available to the world outside our container
EXPOSE 1313

# Run our API when the container launches
CMD ["python", "api.txt"]