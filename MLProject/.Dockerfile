FROM continuumio/miniconda3:latest

WORKDIR /app

COPY conda.yaml .
RUN conda env create -f conda.yaml

COPY . .

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "avocado_ripeness_env", "/bin/bash", "-c"]

# The code to run when container is started
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "avocado_ripeness_env", "python", "modelling.py"]