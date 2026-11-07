FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY taxi_graph ./taxi_graph
COPY tests ./tests

# Dataset is mounted at /data (not baked into the image).
VOLUME ["/data"]
USER app

# Pure stdlib: no dependencies to install.
ENTRYPOINT ["python", "-m", "taxi_graph"]
CMD ["--help"]
