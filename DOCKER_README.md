# Docker розгортання НЯН

## 🚀 Швидкий старт

### 1. Підготовка конфігурації

Створіть `.env` файл з налаштуваннями вашого MongoDB сервера:

```bash
cp .env.example .env
# Відредагуйте .env файл з вашими налаштуваннями
```

### 2. Тестування

Перевірте налаштування перед запуском:

```bash
./test-docker.sh
```

### 3. Запуск

```bash
# Простий спосіб
./start-docker.sh

# Або вручну
docker-compose up -d
```

### 4. Зупинка

```bash
# Простий спосіб  
./stop-docker.sh

# Або вручну
docker-compose down
```

## ⚙️ Конфігурація

### MongoDB налаштування

У файлі `.env` вкажіть параметри вашого зовнішнього MongoDB сервера:

```env
MONGO_HOST=your-mongo-server.com
MONGO_PORT=27017
MONGO_USERNAME=your_username
MONGO_PASSWORD=your_password
MONGO_AUTH_SOURCE=admin
```

### Обов'язкові файли

- `configs/client_config.json` - Telegram API credentials
- `channels.json` - список каналів для парсингу
- `.env` - налаштування MongoDB та інші змінні

## 📦 Сервіси

- **nyan-app** - основний контейнер для дебагу і ручних команд
- **nyan-crawler** - автоматично парсить канали кожні 24 години  
- **nyan-sender** - обробляє та відправляє повідомлення

## 🔧 Управління

### Моніторинг

```bash
# Всі логи
docker-compose logs -f

# Логи конкретного сервісу
docker-compose logs -f nyan-crawler
docker-compose logs -f nyan-sender
```

### Керування сервісами

```bash
# Зупинити всі сервіси
docker-compose down

# Перезапустити сервіс
docker-compose restart nyan-crawler

# Виконати команду в контейнері
docker-compose exec nyan-app bash
```

### Перезбірка

```bash
# Повна перезбірка
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🐛 Troubleshooting

### MongoDB з'єднання
- Переконайтеся що ваш MongoDB сервер доступний з Docker контейнерів
- Перевірте налаштування файрволу
- Для AWS/Cloud серверів може знадобитися додати IP адреси Docker мережі до whitelist

### Логи помилок
```bash
# Детальні логи
docker-compose logs -f --tail=100 nyan-sender
```

### Тестування з'єднання  
```bash
# Тест MongoDB з'єднання
docker-compose exec nyan-app python3 -c "
from nyan.mongo import get_documents_collection
try:
    collection = get_documents_collection('configs/mongo_config.json')
    print('✅ MongoDB з\'єднання успішне')
    print(f'📊 Документів у колекції: {collection.count_documents({})}')
except Exception as e:
    print(f'❌ Помилка з\'єднання з MongoDB: {e}')
"
```
