FROM python:3
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . ./

ENV PYTHONUNBUFFERED=1

CMD ["python3","-u", "app/src/reading_agent_cloudevents.py"]