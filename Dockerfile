# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

# Copy the contents of the current directory to the /services directory in the container
COPY . /services

# Install the Python dependencies listed in the requirements.txt file
RUN pip3 install -r services/requirements.txt

# Expose port 8000 to allow external access to the application
EXPOSE 8000

# Set the entry point to run a shell command
ENTRYPOINT ["sh", "-c"]

# Define the default command to run the Uvicorn server with the application module specified by the APP_MODULE environment variable
CMD ["uvicorn ${APP_MODULE} --host 0.0.0.0 --port 8000"]
