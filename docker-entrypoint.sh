#!/bin/bash
# Готує конфігурацію та моделі, потім передає керування команді контейнера.
set -euo pipefail

cd /app

echo "🔧 Створення MongoDB конфігурації..."
python3 create_mongo_config.py --output-path configs/mongo_config.json

# Решта configs/ їде в образі разом із кодом, тож сюди попадають лише ті два
# файли, яких не можна тримати в git. Падає, якщо значення непридатне: краще
# зупинитися на старті, ніж дати демону крутитися в циклі рестартів.
echo "🔧 Створення конфігурації публікації..."
python3 create_client_config.py --output-path configs/client_config.json

# Моделі не входять в образ: том ./models монтується з хоста, тож при першому
# запуску він порожній.
if [ ! -f "models/lid.176.bin" ]; then
    echo "⚠️ Моделі відсутні, завантажуємо..."
    ./download_models.sh
else
    echo "✅ Моделі знайдені"
fi

exec "$@"
