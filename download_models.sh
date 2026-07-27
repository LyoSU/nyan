#!/bin/bash

set -e  # Зупинитися при будь-якій помилці

echo "🔧 Завантаження моделей..."

mkdir -p models

# Детектор мови (models/lid.176.bin) і класифікатор категорій
# (models/multilingual_e5_base_cat_detect.joblib) лежать в одному архіві релізу.
#
# Новіші fastText-моделі для визначення мови — OpenLID-v3 і GlotLID — важать
# ~1.2–1.7 ГБ проти 131 МБ у lid.176, і на наших даних переваги не показали:
# точність обох 100% у всіх діапазонах довжини поста. Якщо колись знадобиться,
# nyan/lang_detector.py уже вміє їхній формат міток (`ukr_Cyrl`) — досить
# завантажити файл і вказати шлях у configs/annotator_config.json:
#   curl -L -o models/openlid-v3.bin \
#     https://huggingface.co/HPLT/OpenLID-v3/resolve/main/openlid-v3.bin
if [ -f "models/lid.176.bin" ] && [ -f "models/multilingual_e5_base_cat_detect.joblib" ]; then
    echo "✅ Моделі вже на місці, пропускаємо завантаження"
    exit 0
fi

echo "📥 Завантажуємо моделі з GitHub..."
curl -L --fail -o models/nyan_models.tar.gz \
    "https://github.com/NyanNyanovich/nyan/releases/download/v0.3/nyan_models.tar.gz"

echo "📦 Розпаковуємо..."
tar -xzf models/nyan_models.tar.gz -C models
rm models/nyan_models.tar.gz

if [ ! -f "models/lid.176.bin" ]; then
    echo "❌ Помилка: models/lid.176.bin не знайдено після розпакування"
    exit 1
fi

# SigLIP 2 і multilingual-e5-base transformers тягне у свій кеш при першому
# запуску, тому окремого кроку для них тут немає.

echo "✅ Моделі успішно завантажені"
