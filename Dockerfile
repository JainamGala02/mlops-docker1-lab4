FROM python:3.10

# setting the working dir
WORKDIR /app

# copy code into the container
COPY src/ .

# install all req
RUN pip install -r requirements.txt

# run the script as soon as the container starts
CMD ["python", "main.py"]
