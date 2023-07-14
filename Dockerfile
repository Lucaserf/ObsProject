


FROM python:3-alpine
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . ./
EXPOSE 8080
ENTRYPOINT ["python3", "app/src/app.py"]