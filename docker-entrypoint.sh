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

# Запускаємо команду, передану як аргументи
exec "$@"
