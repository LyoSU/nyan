FROM python:3.9-slim

# Встановлюємо системні залежності
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Створюємо робочу директорію
WORKDIR /app

# Копіюємо файли з залежностями
COPY requirements.txt .

# Встановлюємо Python залежності
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код проекту
COPY . .

# Завантажуємо моделі
RUN chmod +x download_models.sh && ./download_models.sh

# Створюємо директорію для даних
RUN mkdir -p data

# Копіюємо та налаштовуємо entrypoint і скрипти
COPY docker-entrypoint.sh /usr/local/bin/
COPY create_mongo_config.py /app/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Експортуємо порти (якщо потрібно)
EXPOSE 8000

# Встановлюємо entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# За замовчуванням запускаємо bash
CMD ["/bin/bash"]
