#!/bin/bash
# Перевіряє, що образ збирається і що з контейнера видно MongoDB.
set -uo pipefail

echo "🐱 Тестування НЯН Docker налаштувань..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker не встановлено!"
    exit 1
fi

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

echo "📋 Перевірка файлів..."
# configs/client_config.json тут немає навмисно: його створює
# docker-entrypoint.sh із CLIENT_CONFIG_JSON, тому у свіжому клоні його й не
# мусить бути. Наявність самої змінної перевіряється нижче, разом з .env.
for required in channels.json; do
    if [ ! -f "$required" ]; then
        echo "❌ Файл $required не знайдено!"
        exit 1
    fi
    echo "✅ $required знайдено"
done

if [ ! -f ".env" ]; then
    echo "⚠️  .env не знайдено — MongoDB використає значення за замовчуванням"
else
    echo "✅ .env знайдено, змінні:"
    grep -E "^(MONGO|LLM)_" .env | sed 's/=.*/=***/' || echo "   Немає MONGO_/LLM_ змінних"

    # Без неї entrypoint зупинить кожен контейнер: publishing-конфіг більше не
    # лежить файлом на хості, а народжується з цієї змінної.
    if grep -qE "^CLIENT_CONFIG_JSON=.+" .env; then
        echo "✅ CLIENT_CONFIG_JSON задано"
    else
        echo "❌ CLIENT_CONFIG_JSON не задано — контейнери не стартують"
        exit 1
    fi
fi

echo "🔨 Збірка Docker образу..."
if ! $COMPOSE build --quiet; then
    echo "❌ Помилка збірки Docker образу"
    exit 1
fi
echo "✅ Docker образ зібрано успішно"

echo "🔌 Тестування MongoDB підключення..."
# Скрипт передається через stdin, а не в подвійних лапках: інакше $MONGO_HOST
# та інші змінні розкриваються у хостовому шелі й до контейнера доходять
# порожніми, тобто перевірка нічого не перевіряє.
$COMPOSE run --rm -T nyan-app bash -s <<'CONTAINER'
set -u
echo "Змінні середовища в контейнері:"
echo "  MONGO_HOST=${MONGO_HOST:-<не задано>}"
echo "  MONGO_PORT=${MONGO_PORT:-<не задано>}"
echo "  MONGO_USERNAME=${MONGO_USERNAME:+***}"
echo "  MONGO_PASSWORD=${MONGO_PASSWORD:+***}"
echo "  LLM_MODEL=${LLM_MODEL:-<не задано>}"
echo
echo "Створена конфігурація:"
cat configs/mongo_config.json
echo
echo "Підключення:"
python3 - <<'PYTHON'
from nyan.mongo import get_documents_collection

collection = get_documents_collection("configs/mongo_config.json")
print(f"✅ MongoDB з'єднання успішне")
print(f"📊 Документів у колекції: {collection.count_documents({})}")
PYTHON
CONTAINER

if [ $? -eq 0 ]; then
    echo "🎉 Всі тести пройшли успішно!"
    echo "🚀 Запускайте НЯН командою: ./start-docker.sh"
else
    echo "❌ Тести не пройшли. Перевірте налаштування."
    exit 1
fi
