#!/bin/bash

# Скрипт для підстановки змінних середовища в mongo_config.json

CONFIG_FILE="/app/configs/mongo_config.json"
TEMPLATE_FILE="/app/configs/mongo_config.json.template"

# Створюємо конфігурацію з підстановкою змінних
envsubst < "$TEMPLATE_FILE" > "$CONFIG_FILE"

echo "📝 MongoDB конфігурація створена:"
cat "$CONFIG_FILE"

# Запускаємо команду, передану як аргументи
exec "$@"
