# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install MetaTrader5 setup
RUN wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
RUN chmod +x mt5setup.exe

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV DISPLAY=:99
ENV MT5_PATH=/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe

# Create entrypoint script
RUN echo '#!/bin/bash\nXvfb :99 -screen 0 1024x768x16 &\nsleep 1\npython main.py train --symbol EURUSD --timeframe 5 --episodes 10000' > /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Run the bot
ENTRYPOINT ["/app/entrypoint.sh"]