# Укажите базовый образ
FROM python:3.9

# Скопируйте requirements.txt в контейнер
COPY requirements.txt .

COPY . /data

# Установите зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Скопируйте исходный код приложения в контейнер
COPY . /Prosept_script

# Установите рабочую директорию для контейнера
WORKDIR /Prosept_script

# Запустите приложение при старте контейнера
CMD ["python", "Prosept_script.py"]