#!/bin/bash

# Скрипт для тестування НЯН Docker setup

echo "🐱 Тестування НЯН Docker налаштувань..."

# Перевіряємо Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не встановлено!"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не встановлено!"
    exit 1
fi

# Перевіряємо файли
echo "📋 Перевірка файлів..."

if [ ! -f "channels.json" ]; then
    echo "❌ Файл channels.json не знайдено!"
    exit 1
else
    echo "✅ channels.json знайдено"
fi

if [ ! -f "configs/client_config.json" ]; then
    echo "❌ configs/client_config.json не знайдено!"
    exit 1
else
    echo "✅ client_config.json знайдено"
fi

if [ ! -f ".env" ]; then
    echo "⚠️  .env файл не знайдено!"
    echo "   Створіть .env з налаштуваннями MongoDB"
else
    echo "✅ .env файл знайдено"
    echo "📝 Змінні середовища:"
    grep -E "^MONGO_" .env | sed 's/=.*/=***/' || echo "   Немає MONGO_ змінних"
fi

# Збираємо образ
echo "🔨 Збірка Docker образу..."
if docker-compose build --quiet; then
    echo "✅ Docker образ зібрано успішно"
else
    echo "❌ Помилка збірки Docker образу"
    exit 1
fi

# Тестуємо MongoDB підключення
echo "🔌 Тестування MongoDB підключення..."
docker-compose run --rm nyan-app bash -c "
echo 'Змінні середовища:'
echo 'MONGO_HOST='$MONGO_HOST
echo 'MONGO_PORT='$MONGO_PORT
echo 'MONGO_USERNAME='${MONGO_USERNAME:+***}
echo 'MONGO_PASSWORD='${MONGO_PASSWORD:+***}
echo ''
echo 'Створена конфігурація:'
cat configs/mongo_config.json
echo ''
echo 'Тестування підключення...'
python3 -c \"
from nyan.mongo import get_documents_collection
try:
    collection = get_documents_collection('configs/mongo_config.json')
    count = collection.count_documents({})
    print(f'✅ MongoDB з\\'єднання успішне')
    print(f'📊 Документів у колекції: {count}')
except Exception as e:
    print(f'❌ Помилка з\\'єднання з MongoDB: {e}')
    import traceback
    traceback.print_exc()
    raise
\"
"

if [ $? -eq 0 ]; then
    echo "🎉 Всі тести пройшли успішно!"
    echo "🚀 Запускайте НЯН командою: ./start-docker.sh"
else
    echo "❌ Тести не пройшли. Перевірте налаштування."
    exit 1
fi
