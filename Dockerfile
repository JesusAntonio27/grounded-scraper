FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# poppler renders PDF pages for OCR; tesseract reads images (English + Spanish).
RUN apt-get update \
    && apt-get install --no-install-recommends -y poppler-utils tesseract-ocr tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

RUN groupadd --system scraper && useradd --system --gid scraper --create-home scraper
COPY app ./app
COPY runner ./runner
RUN chown -R scraper:scraper /app

USER scraper
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
