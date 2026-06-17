FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8090
ENV PYTHONUNBUFFERED=1

EXPOSE 8090

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
