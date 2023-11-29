FROM python:3.9

COPY requirements.txt .

COPY . /data

RUN pip install --no-cache-dir -r requirements.txt

COPY . /Prosept_script

WORKDIR /Prosept_script

CMD ["python", "Prosept_script.py"]
