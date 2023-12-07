# Укажите базовый образ
FROM python:3.9

# Установим рабочую директорию для контейнера
WORKDIR /app

# Скопируем файлы main.py, Prosept_func.py и requirements.txt внутрь контейнера
COPY main.py .
COPY Prosept_func.py .
COPY requirements.txt .

# Установим зависимости Python только при первом запуске контейнера
RUN pip install --no-cache-dir -r requirements.txt

# Запустим приложение при старте контейнера
CMD ["python", "main.py"]