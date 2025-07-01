#!/bin/bash

# Скрипт для зупинки НЯН Docker

echo "🛑 Зупинка НЯН..."

docker-compose down

echo "🧹 Очищення контейнерів..."
docker-compose ps -q | xargs -r docker rm -f

echo "✅ НЯН зупинено!"

# Показуємо статус
echo "📊 Статус контейнерів:"
docker ps | grep nyan || echo "Жодних НЯН контейнерів не запущено"
