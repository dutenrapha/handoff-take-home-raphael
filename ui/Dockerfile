FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt
COPY ui.py ui.py
COPY ./.streamlit/ ./.streamlit/

RUN pip install --upgrade pip &&pip install --no-cache-dir -r requirements.txt

CMD ["python", "ui.py"]
