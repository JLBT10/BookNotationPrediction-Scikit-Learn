# Use the official Ubuntu as the base image
FROM ubuntu:latest

# Update package list and install Python 3.9 and necessary packages
RUN apt-get update -y && \
    apt-get install -y python3.9 && \
    apt-get install -y python3-pip &&\
    pip install --upgrade pip 

# Set the working directory
WORKDIR /app/src

# Copy the code into the image
COPY . /app
 # Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Specify the default command to run when the container starts
#CMD ["python3", "train.py"]
