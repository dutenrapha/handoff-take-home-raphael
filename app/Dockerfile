FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt
COPY main.py main.py
COPY config.yaml config.yaml
COPY evaluators/ evaluators/
COPY adapters/ adapters/
COPY observers/ observers/
COPY reports/ reports/
COPY data/ data/
COPY utils/ utils/

RUN pip install --upgrade pip &&pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
