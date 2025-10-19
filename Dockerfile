# Use a lightweight CPU-only PyTorch image to avoid slow/large pip installs
FROM pytorch/pytorch:2.3.1-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first (leveraging Docker layer cache)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render provides PORT; Gunicorn config reads it
EXPOSE 5000

# Start the application with Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:application"]
