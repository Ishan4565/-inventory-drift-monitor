# Use a lightweight Python image
FROM python:3.9-slim

# Install system-level database libraries (The missing piece!)
RUN apt-get update && apt-get install -y libpq-dev gcc

# Set the working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Run the app exactly like your .bat file would
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]