FROM python:3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . ./
# EXPOSE 8080
ENV PYTHONUNBUFFERED=1


CMD ["python3","-u", "app/src/pull.py"]