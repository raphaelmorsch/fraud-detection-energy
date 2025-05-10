FROM python:3.9-slim
WORKDIR /app
COPY preprocess.py .
RUN pip install pandas scikit-learn
ENTRYPOINT ["python", "preprocess.py"]