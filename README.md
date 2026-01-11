# MLOps_БЯМ: Чешские БЯМы или как заставить матрицы mluvit česky. 
Главной целью данной работы является разработка и обучение Большой Языковой Модели (БЯМ, a.k.a. Velký Jazykový Model), предназначенной для общения на **чешском** (český!) языке.
Модель должна быть доступной для "обычного пользователя", поэтому должна удовлетворять следующим требованиям:
- Для обучения и использование должно быть достаточно **12** ГБ видеопамяти (VRAM).
- Время ответа модели не должно превышать **1000 мс** на 256 токенов (tokenů!) на GPU.
- Итоговый ответ модели всегда должен должен быть на чешском языке и определяться таковым с уверенностью (jistota!) (по fasttext) **> 0.8**.
Качество обученной модели должно удовлетворять следующим требованиям:
- **CELoss** должен быть не выше **3.0** на обучающих данных.
- Něco se musí přidat v dalších fázích....

## Данные
На данном этапе существования проекта (projektu!) реализовано только предобучение БЯМ, для которого использовались тексты на чешском языке, полученные из набора данных [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).
- В исходном наборе данных (dat!) представлено более 100 ГБ текстов - для обучения использовалось порядка **200 мб**.
Предобработка данных включала в себя следующие этапы:
- Фильтрация по уверенности в чешском языке при помощи fasttext (тексты с уверенностью **< 0.9** отбрасывались),
- Фильтрация ссылок, смайлов (emoticons!), служебных символов. 
- Токенизация и разбиение на блоки по **256** токенов. 

## План экспериментов
В план экспериментов входят:
1. Подготовка, изучение и обработка данных для обучения.
2. Подбор и анализ подходящего токенизатора.
3. Обучение токенизатора на подготовленных данных.
4. Подбор архитектуры БЯМ.
5. Обучение (školení!) БЯМ на подготовленных данных.
6. Проверка инференса обученной модели.

В ходе обучения дополнительно ведётся логирование промежуточных результатов (výsledek!) и валидация на отдельных примерах.

## Техническая составляющая
- Проект реализован на языке Python 3.10. 
- Данные, токенизатор и модель реализованы при помощи инструментов Hugging Face.
- Fвтоматическая проверка кода (CI/CD) реализована при помощи GitHub Actions.
- Необходимые требования по пакетам представлены в **requirements.txt** и **requirements_no_deps.txt** (bohužel, есть проблемы с совместимостью datatrove...).

Установка пакетов может быть выполнена следующим образом:

```bash
pip install -r requirements.txt
pip install --no-deps -r .\requirements_no_deps.txt
```
- Представлено два скрипта: main.py для полного обучения и inference.py для проверки инференса.

Запуск может быть выполнен следующим образом:

```bash
python main.py --config_params_path 'config_params.yaml' --verbose 1
```
и

```bash
python inference.py --config_params_path 'config_params.yaml' --input_sequence 'Počasí v Praze '
```

## Версионирование при помощи DVC + удалённое хранилище данных с Dagshub
- DVC используется для версионирования (verze!) следующих данных: сырых и обработанных датасетов, токенизатора и модели.
- Эти данные расположены в удалённом хранилище Dagshub и не хранятся в Git репозитории.
Для загрузки всех данных с репозитория:
Клонирование (Klonování!) репозитория
```bash
git clone https://github.com/Prokudin-Dmitrii/MLOps_byam.git
```
Установка пакетов:
```bash
pip install -r requirements.txt
pip install --no-deps -r .\requirements_no_deps.txt
```
Подготовка (Příprava!) DVC:
```bash
dvc remote modify dagshub auth basic
dvc remote modify dagshub user $DAGSHUB_USER
dvc remote modify dagshub password $DAGSHUB_TOKEN
```
Подтягивание данных с удалённого хранилища:
```bash
dvc pull
```
При изменении (změna!) пайплайна обучения:
```bash
dvc repro
dvc push
```
Все большие файлы данных всегда должны восстанавливаться командой dvc pull. 

Пайплайн DVC состоит из 5 стадий:
- fetch_raw_data, сбор сырых данных
- fetch_translator, сбор данных для переводчика
- prepare, подготовка данных для обучения
- train, обучение модели
- evaluate, оценка работы модели

Результаты работы каждой стадии отслеживаются DVC.

## Логгирование экспериментов при помощи MLFlow
- Весь процесс обучения из train.py логгируется при помощи MLFlow.
- Каждый запуск train.py создаёт отдельный (samostatný!) run, по умолчанию хранящийся в /mlruns.
- Все логи доступны через интерфейс (rozhraní!) MLFlow:
```bash
mlflow ui
```
- MLFlow привязан к DVC: dvc.lock и dvc.yaml хранятся как артефакты.
- Дополнительно в качестве тега каждого запуска используется хэш текущих данных (dat!) для обучения из DVC.

## Docker
- Имеется Dockerfile для создания образа с запуском predict'а модели (развёртывание модели для офлайн? инференса).
- Модель читает тексты (textů!) из input.txt, дополняет их и записывает в output.txt.
Создание образа/контейнера (собирается долго!):
```bash
docker build -f Dockerfile -t ml-app:v1 .
```
Запуск контейнера (kontejner!):
```bash
docker run --rm \
  -v $(pwd)/input.txt:/input.txt \
  -v $(pwd)/output.txt:/output.txt \
  ml-app:v1 \
  --config_params_path config_params.yaml \
  --input_path /input.txt \
  --output_path /output.txt
```

Входные данные ожидаются (se očekávají!) в файле input.txt, где каждая строка - новый текст для завершения, например:
```bash
Počasí v Praze 
Všechny síly byly vynaloženy 
Studenti jsou unavení
```
Результат работы модели сохраняется в output.txt в соответствующем построчном (podle řádek!) виде.

## TorchServe
- Имеется Dockerfile.torchserve для развёртывания (vystružování!) модели как онлайн сервиса.
- Модель принимает на вход текст и возвращает его завершённый вариант.
- Модель и токенизатор запакованы в .mar архив (archiv!).
Сборка и запуск:
```bash
docker build -f Dockerfile.torchserve -t mymodel-serve:v1 .
docker run -d -p 8080:8080 -p 8081:8081 mymodel-serve:v1
```
Проверка состояния сервиса (služby!):
```bash
curl http://localhost:8080/ping
```
Пример REST-запроса:
```bash
curl -X POST http://localhost:8080/predictions/mymodel \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Počasí v Praze ",
        "max_length": 50,
        "do_sample": true,
        "top_p": 0.9,
        "temperature": 0.9,
        "repetition_penalty": 1.2
      }'
```
Пересборка .mar архива:
```bash
torch-model-archiver   --model-name mymodel   --version 1.0   --handler src/handler.py   --extra-files "model/byam_step_final/,data/tokenizer/"   --export-path model-store   --force
```
- Параметры сервиса описаны в конфигурационном файле (soubor!): config.properties.
