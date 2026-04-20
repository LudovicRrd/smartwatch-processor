FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p results

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "--timeout", "120", "app:app"]
