FROM python:3.12-slim

WORKDIR /app

COPY mock-allegro/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mock-allegro/mock_api.py .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "mock_api.py"]
