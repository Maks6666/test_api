FROM python:3.11.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    # software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

CMD ["python3", "app_api.py"]
