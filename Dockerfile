FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY packs/ packs/

RUN pip install --no-cache-dir .

ENTRYPOINT ["evalrun"]
