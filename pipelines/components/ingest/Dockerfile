FROM python:3.9-slim
WORKDIR /app
COPY ingest.py .
RUN pip install pandas==2.0.3
ENTRYPOINT ["python", "ingest.py"]