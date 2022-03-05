FROM python:3.8.12-slim-bullseye

RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1 

WORKDIR /work

RUN python -m pip install --upgrade pip
RUN pip3 install pandas python-dotenv slack_sdk
RUN pip3 install -U scikit-learn
RUN pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

CMD ["python", "main.py"]