FROM python:3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN apt-get update -y && apt-get upgrade -y
# RUN apt-get install tcpdump -y
COPY . ./

ENV PYTHONUNBUFFERED=1

CMD ["python3","-u", "app/src/logging_agent.py"]
