#!/bin/bash
# Запускає НЯН у Docker (MongoDB — зовнішня).
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

echo "🐱 Запуск НЯН в Docker..."

# Перевіряємо чи існують необхідні файли
if [ ! -f "channels.json" ]; then
    echo "❌ Файл channels.json не знайдено!"
    exit 1
fi

if [ ! -f "configs/client_config.json" ]; then
    echo "❌ Файл configs/client_config.json не знайдено!"
    echo "   Додайте ваші Telegram API credentials"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "⚠️  Файл .env не знайдено!"
    echo "   Створіть .env файл з налаштуваннями MongoDB:"
    echo "   MONGO_HOST=your-mongo-server.com"
    echo "   MONGO_PORT=27017"
    echo "   MONGO_USERNAME=your_username"
    echo "   MONGO_PASSWORD=your_password"
    echo ""
    read -p "Продовжити без .env файлу? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Створюємо необхідні директорії
mkdir -p data models

echo "🔨 Збираємо Docker образи..."
$COMPOSE build

echo "🚀 Запускаємо сервіси..."
$COMPOSE up -d

echo "✅ НЯН запущено!"
echo ""
echo "📊 Для перегляду логів:"
echo "   $COMPOSE logs -f"
echo ""
echo "🛑 Для зупинки:"
echo "   $COMPOSE down"
echo ""
echo "🔍 Статус сервісів:"
$COMPOSE ps
