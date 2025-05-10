FROM python:3.9-slim
WORKDIR /app
COPY train.py .
RUN pip install scikit-learn pandas joblib
ENTRYPOINT ["python", "train.py"]