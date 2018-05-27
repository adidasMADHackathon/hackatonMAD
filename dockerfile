FROM frolvlad/alpine-miniconda3
COPY . /app
WORKDIR /app
RUN apk add --no-cache bash libstdc++
RUN conda env create -f environment.yml 
EXPOSE 5000
WORKDIR /app/src
ENTRYPOINT ["bash", "-c", "source activate bot && FLASK_APP=api.py flask run --host=0.0.0.0 --port=5000"]
