#!/bin/bash

# Скрипт для створення правильної mongo_config.json з типами даних

echo "🔧 Створення MongoDB конфігурації..."

# Запускаємо Python скрипт для створення конфігурації
python3 /app/create_mongo_config.py

# Перевіряємо чи файл створено
if [ ! -f "/app/configs/mongo_config.json" ]; then
    echo "❌ Не вдалося створити mongo_config.json"
    exit 1
fi

echo "✅ MongoDB конфігурація готова"

# Перевіряємо та завантажуємо моделі, якщо вони відсутні
echo "🔍 Перевірка наявності моделей..."
if [ ! -f "/app/models/lid.176.bin" ]; then
    echo "⚠️ Модель lid.176.bin відсутня, завантажуємо..."
    cd /app && ./download_models.sh
else
    echo "✅ Моделі знайдені"
fi

# Запускаємо команду, передану як аргументи
exec "$@"
