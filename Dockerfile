FROM python:3.9-slim

WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .
COPY main.py .
COPY src/ src/
COPY models/ models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]