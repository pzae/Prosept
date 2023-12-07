## Prosept. Cоздание сервиса для полуавтоматической разметки товаров.

### Группа DS команды №12 

**Задача: разработка решения, которое автоматизирует процесс сопоставления товаров заказчика с размещаемыми товарами дилеров.**

За выполнение поставленной задачи со стороны DS отвечали Александр (TL), Милана и Рената. Использованный язык программирования - **python**.

### Инструкция к репозиторию

* `Prosept_reasearch.ipynb` - тетрадка с подробным описанием хода выполнения проекта. Рекомендуется к ознакомлению.
  
Папка `deploy` предназначена для передачи бэкенду и сборки docker контейнера.:
* `main.py` - основной выполняемый скрипт;
* `Prosept_func.py` - скрипт, содержащий созданный класс `Prosept_func`, в котором описаны все необходимые для выполнения `main.py` функции;
* `requirements.txt` - файл для создания докера;
* `Dockerfile` - файл для создания докера.

### Описание решения

За основную гипотезу выбрано предположение о том, что названий товаров достаточно для получения мэтча с приемлемой точностью. Для реализации проверки этой гипотезы выбран метод `cosine_similarity`, применённый к векторам названий, полученным с помощью `TfidfVectorizer()` и `SentenceTransformer('LaBSE')`. Сопоставляются вектора столбцов `df_dealerprice.product_name` и `df_product.name`. Остальные фичи не используются, так по итогам проверки гипотезы получен удовлетворительный результат.

### Проделанная работа

Милана начала работу с получения эмбеддингов с помощью предобученной модели `BERT`. Однако было принято решение отложить это направление в пользу наиболее актуальных задач: предобработки названий, лемматизации, исправления ошибок в названиях. После решения этих задач и до конца хакатона  продолжала работу с `tiny-BERT`. Рената решила попробовать применить метод `ALS`, который по итогу командным решением признали не целесообразным. После этого продолжила работу Миланы по предобработке текста. Её подход добавил 11% точности к показаниям нашей метрики. Так же рассмотрела векторизацию с помощью `SentenceTransformer('LaBSE')`. Александр прорабатывал бэйзлайн решения: метод `cosine_similarity` для векторов, рассчитанных с помощью `TfidfVectorizer`, собирал код, оформалял тетрадку с исследованием, разрабатывал функции:
1. Серия функций `preprocess` для препроцессинга текста на основе кода, составленного совместно Миланой и Ренатой. Представлена в коде скрипта.
2. Функция векторизации `vectorize`.
3. Функция предсказания `prediction`, которая рассчитывает `cos_sim` для всех уникальных пар `название дилера - название заказчика`
4. Функция `get_id_key` для сопоставления используемых `id` настоящим.
5. Функция `result_to_json` для сохранения данных в файл с расширением `.json` в скрипте и `result_to_df` в тетрадке.
6. Функция `save_json` для сохранения файла на сервер.
7. Функции метрик `metric_top_5` и `mean_reciprocal_rank`. Не входят в финальный скрипт.
8. Функция `first_n_no_match` для изучения неправильно полученных мэтчей. Так же не входит в финальный скрипт.

По итогам работы освоена технология `Docker`, упакованный в контейнер код выполняется корректно. Контейнер общается с сервером посредством `requests.get()` и `requests.post()`.

### Результаты

На данный момент наилучший результат: 
- **86%** правильных названий продуктов от заказчика попадают в топ-5 предложенных, <br>
- **0.6375** - значение `mean_reciprocal_rank`, <br>
- **менее 1** секунды - время выполнения в тетрадке функции предсказания - самой ресурсоёмкой функции - для всего набора данных.

### Выводы и планы

Выбранная базовая гипотеза подтвердилась, предложенный для её реализации метод дал отличные результаты. При этом в используемых названиях от заказчика и дилеров всё еще есть недостатки, ликвидируя которые можно дополнительно поднять точность мэтчей.

В ходе выполнения проекта релизована предобработка данных. Разработан метод сопоставления топ n названий заказчиков названиям дилеров. Получены отличная скорость выполнения кода и отличное значение метрики.

### Инструкция к коду

1. При необходимости изменить `url_dealerprice` и `url_product` для входных данных и `url_save_json` для выходных данных в файле `main.py`.
2. Упаковать содержимое папки в docker контейнер.
3. Интегрировать в бэкенд
4. Запустить контейнер
5. На выходе получить файл `result.json` по указанному в `url_save_json` адерсу.
