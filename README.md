### Символьная RNN для генерации текста

Этот проект реализует символьную рекуррентную нейронную сеть (RNN) для предсказания следующего символа на основе полученного корпуса текстов. 
C помощью модели можно генерировать текстовые последовательности заданной длины, используя фиксированную начальную фразу.

data: Папка с текстовыми корпусами для обучения моделей. На данный момент в ней содержится файл onegin.txt.

output: Папка для хранения сгенерированных текстов. В текущей версии проекта в ней содержатся 12 файлов (text_1.txt, text_2.txt и т.д.), полученных во время тестирования моделей.

model: Папка для файлов с параметрами обученных моделей. Подробности о модели и параметрах см. в ноутбуке rnn_work_1.ipynb.

rnn_work_1.ipynb: Jupyter ноутбук проекта, содержащий алгоритм обучения RNN-моделей.

app.py: FastAPI-приложение, которое предоставляет доступ к модели через API.

request_proba.py: Пример POST-запроса для работы с FastAPI.

requirements.txt: Файл с зависимостями для установки окружения.

### API Приложение

Приложение на FastAPI (app.py) принимает POST-запрос по адресу http://0.0.0.0:8000/forms/rnn/inferense. Параметры запроса:

```  
    {
         'name_model':'model_2', # название  модели ( доступны 3 обученные модели model_1, model_2, model_3)
         'prime_str':  'ПРИВЕТ', # ачальная фраза
         'predict_len': 300,     # длинапоследоательности
         'temperature': 0.5      # температура (1 не изменяет генерацию)
       }
```

 Ответ от API возвращает путь и URL для доступа к сгенерированному тексту:
 
 ```
  {
    "save_url": "save_url",   # URL для доступа к файлу
    "save_path": "save_path"  # Локальный путь к файлу
 }   
``` 
###Сборка и запуск в Docker

Сборка Docker image:  make  run    
                                 или  docker build --tag rnn .


Запуск  контенера:make  run 
                            или  docker run --rm -it -p 8000:8000 \
		                     -v ./model:/app/model \
		                     -v ./output:/app/output \
		                     rnn                          
локальные  папки model, output смонтированы  в контейнер.

Контейнер после остановки будет удален.  

### Пример запроса к API
После запуска приложения в контейнере или локально можно отправить запрос к API. Пример запроса находится в request_proba.py


#### Дополнительно
папка task 

Прикладываю решение задачи из курса Тренировки по алгоритмам 6.0 от Яндекса.
- task_3.ipnb -реализованный алгоритм  решения и условие задачи;
- task_.py -скрипт  с реализацией алгоритма решения задачи;
- input.txt - пример входных данных для тестирования скрипта.


