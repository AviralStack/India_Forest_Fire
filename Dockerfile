
FROM python:3.12-slim

WORKDIR /code

COPY xgboost*.whl .


RUN pip install xgboost*.whl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]