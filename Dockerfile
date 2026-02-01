# 1. Base Image
FROM python:3.12-slim

# 2. Set Working Directory
WORKDIR /code

# 3. COPY THE WHEEL FILE (The Manual Carry)
# We copy any .whl file to the container
COPY xgboost*.whl .

# 4. INSTALL THE WHEEL MANUALLY
RUN pip install xgboost*.whl

# 5. Install Dependencies (Now missing xgboost, which is fine)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy App Code
COPY app ./app
COPY models ./models

# 7. Start Command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]