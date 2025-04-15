FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py .
COPY data/ ./data/

EXPOSE 8000
EXPOSE 7860

ENV API_URL=http://localhost:8000
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "main.py"]