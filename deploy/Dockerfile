# Укажите базовый образ
FROM python:3.9

# Установим рабочую директорию для контейнера
WORKDIR /app

# Скопируем файлы main.py, Prosept_func.py и requirements.txt внутрь контейнера
COPY main.py .
COPY Prosept_func.py .
COPY requirements.txt .

# Скопируем папку data с marketing_productdealerkey.csv внутри
COPY . data/marketing_productdealerkey.csv /app/

# Установим зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Запустим приложение при старте контейнера
CMD ["python", "main.py"]