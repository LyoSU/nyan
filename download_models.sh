#!/bin/bash

set -e  # Зупинитися при будь-якій помилці

echo "🔧 Завантаження моделей..."

# Створюємо директорію для моделей
mkdir -p models

# Перевіряємо, чи модель вже існує
if [ -f "models/lid.176.bin" ]; then
    echo "✅ Модель lid.176.bin вже існує, пропускаємо завантаження"
    exit 0
fi

echo "📥 Завантажуємо моделі з GitHub..."

# Завантажуємо архів з моделями
curl -L -o models/nyan_models.tar.gz https://github.com/NyanNyanovich/nyan/releases/download/v0.3/nyan_models.tar.gz

# Перевіряємо, чи файл завантажився
if [ ! -f "models/nyan_models.tar.gz" ]; then
    echo "❌ Помилка: файл моделей не завантажився"
    exit 1
fi

echo "📦 Розпаковуємо моделі..."

# Розпаковуємо архів
cd models && tar -xzvf nyan_models.tar.gz && rm nyan_models.tar.gz

# Перевіряємо, чи модель успішно розпакувалася
if [ -f "lid.176.bin" ]; then
    echo "✅ Моделі успішно завантажені та розпаковані"
else
    echo "❌ Помилка: модель lid.176.bin не знайдена після розпакування"
    exit 1
fi
