FROM python:3.9-slim

# Install system dependencies (including SWIG)
RUN apt-get update && apt-get install -y \
    swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["streamlit", "run", "single_app.py"]  # Changed "your_app.py" to "app.py" (or your actual filename)
