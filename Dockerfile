FROM docker.io/library/python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home app && mkdir /app/data && chown app:app /app/data
USER app
EXPOSE 8000
CMD ["rag-demo", "serve"]
