FROM python:3.12-slim

WORKDIR /tests

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

COPY . .

# ALLEASYSTENT_URL  — adres backendu (wymagane, np. https://twoja-app.railway.app)
# ALLEGRO_AUTHED    — ustaw "1" aby uruchomić testy wymagające autoryzacji Allegro
ENV ALLEASYSTENT_URL=""
ENV ALLEGRO_AUTHED="0"
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn runner:app --host 0.0.0.0 --port ${PORT}"]
