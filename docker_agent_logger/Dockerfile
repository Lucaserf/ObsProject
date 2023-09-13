FROM python:3-alpine
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . ./

ENV PYTHONUNBUFFERED=1

CMD ["python3","-u", "app/src/logging_agent_kube.py"]