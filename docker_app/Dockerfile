FROM python:3-alpine
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . ./
# EXPOSE 8080
ENV PYTHONUNBUFFERED=1
# CMD ["python3","-u", "app/src/app.py"]
CMD ["python3","-u", "app/src/log_generator.py"]