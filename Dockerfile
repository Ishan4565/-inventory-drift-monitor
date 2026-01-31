FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libpq-dev gcc

WORKDIR /app

# Set the Environment Variable INSIDE the container
# Replace the URL below with your actual 'External Database URL' from Render
ENV DATABASE_URL="postgresql://user:password@host:port/dbname?sslmode=require"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
