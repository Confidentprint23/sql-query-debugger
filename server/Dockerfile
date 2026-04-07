FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code + shared models
COPY server/app.py         ./server/app.py
COPY server/environment.py ./server/environment.py
COPY models.py             ./models.py

# HF Spaces expects the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]