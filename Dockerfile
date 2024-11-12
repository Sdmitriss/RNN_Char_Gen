FROM python:3.12.5

COPY app.py /app/
COPY requirements.txt /app/ 
COPY model /app/model
COPY output /app/output


WORKDIR /app
RUN pip install -r requirements.txt

CMD [ "python", "app.py" ]