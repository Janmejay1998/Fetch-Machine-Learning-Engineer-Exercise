FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run train.py to train the model 
RUN python model_train.py

# Run inference.py to generate model predictions 
RUN python inference.py

# Run app.py when the container launches
CMD ["python", "app.py"]