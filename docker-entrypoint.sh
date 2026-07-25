#!/bin/bash
# Готує конфігурацію та моделі, потім передає керування команді контейнера.
set -euo pipefail

cd /app

echo "🔧 Створення MongoDB конфігурації..."
python3 create_mongo_config.py --output-path configs/mongo_config.json

# Моделі не входять в образ: том ./models монтується з хоста, тож при першому
# запуску він порожній.
if [ ! -f "models/lid.176.bin" ]; then
    echo "⚠️ Моделі відсутні, завантажуємо..."
    ./download_models.sh
else
    echo "✅ Моделі знайдені"
fi

exec "$@"
