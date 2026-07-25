#!/bin/bash
# Зупиняє НЯН.
set -uo pipefail

# `docker compose` — сучасна вбудована команда; docker-compose лишився для
# старих встановлень.
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "❌ Docker Compose не встановлено!"
    exit 1
fi

echo "🛑 Зупинка НЯН..."

$COMPOSE down

echo "🧹 Очищення контейнерів..."
$COMPOSE ps -q | xargs -r docker rm -f

echo "✅ НЯН зупинено!"

# Показуємо статус
echo "📊 Статус контейнерів:"
docker ps | grep nyan || echo "Жодних НЯН контейнерів не запущено"
