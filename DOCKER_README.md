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
docker compose up -d
```

### 4. Зупинка

```bash
# Простий спосіб
./stop-docker.sh

# Або вручну
docker compose down
```

## ⚙️ Конфігурація

### MongoDB налаштування

У файлі `.env` вкажіть параметри вашого зовнішнього MongoDB сервера:

```env
MONGO_HOST=your-mongo-server.com
MONGO_PORT=27017
MONGO_DATABASE=main
MONGO_USERNAME=your_username
MONGO_PASSWORD=your_password
MONGO_AUTH_SOURCE=admin
```

### LLM налаштування

НЯН звертається до LLM для заголовків постів, розбіжностей між джерелами та
дайджестів. Підходить будь-який OpenAI-сумісний шлюз — OpenRouter, LiteLLM тощо:

```env
LLM_API_KEY=your_llm_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-5.4-mini
LLM_TIMEOUT=120
LLM_MAX_RETRIES=3
LLM_REASONING_EFFORT=low
```

Якщо шлюз не підтримує якийсь параметр (наприклад `reasoning_effort`), НЯН
прочитає це з тексту помилки, повторить запит без нього і запам'ятає обмеження
до кінця роботи процесу — тож ламатися на кожному виклику воно не буде.

Це відновлення коштує один невдалий запит на процес. Якщо ви точно знаєте, що
ваш шлюз не приймає `reasoning_effort` (типово для LiteLLM з
`custom_openai`), поставте `LLM_REASONING_EFFORT=none` — параметр не піде взагалі.

### Формат постів

`configs/renderer_config.json` — `"post_format"`:

- `"rich"` — структуровані блоки через `sendRichMessage` (Bot API 10.1+): заголовок,
  медіа окремим блоком, цитати з атрибуцією, згортаний список джерел.
- `"legacy"` — старий HTML-текст із фото як `caption`. Тримається для швидкого
  відкату без редеплою.

`"sources_open": true` розгортає список джерел за замовчуванням.

### Обов'язкові файли

- `configs/client_config.json` - Telegram API credentials
- `channels.json` - список каналів для парсингу
- `.env` - налаштування MongoDB та інші змінні

**Важливо:** `configs/mongo_config.json` автоматично генерується з `.env` змінних, тому його не потрібно створювати вручну.

### Стан краулера

`data/fetch_times.json` — коли який канал читали останній раз. Файл створюється
сам і живе в томі `./data`. Раніше він лежав у `crawler/`, через що доводилося
монтувати папку з кодом — і змонтований код перекривав образ, тобто
`docker compose build` оновлював залежності, але не код. Якщо у вас є старий
`crawler/fetch_times.json`, `crawl.sh` перенесе його автоматично при першому запуску.

## 📦 Сервіси

- **nyan-app** — контейнер для дебагу і ручних команд, нічого не робить сам
- **nyan-crawler** — `crawl.sh`: безперервно парсить канали (темп задає
  `recrawl_time` кожного каналу), з наростаючою паузою після падінь
- **nyan-sender** — `send.sh`: кластеризує й публікує, так само з backoff

## 🔧 Управління

### Моніторинг

```bash
# Всі логи
docker compose logs -f

# Логи конкретного сервісу
docker compose logs -f nyan-crawler
docker compose logs -f nyan-sender
```

### Керування сервісами

```bash
# Зупинити всі сервіси
docker compose down

# Перезапустити сервіс
docker compose restart nyan-crawler

# Виконати команду в контейнері
docker compose exec nyan-app bash
```

### Перезбірка

```bash
# Повна перезбірка
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 🐛 Troubleshooting

### MongoDB з'єднання
- Переконайтеся що ваш MongoDB сервер доступний з Docker контейнерів
- Перевірте налаштування файрволу
- Для AWS/Cloud серверів може знадобитися додати IP адреси Docker мережі до whitelist

### Краулер не збирає пости

Якщо в логах `nyan-crawler` є

```
ERROR: Crawl scraped no posts at all (0 requests, reason: finished)
```

то прохід завершився без жодного поста. Причини за частотою:

1. **Усі канали пропущені** через `recrawl_time` — у логу буде
   `Requesting 0 of N channels`. Нормально, якщо прохід був щойно.
2. **Змінилася верстка t.me** — `Requesting N of N`, запити є, постів нуль.
   Запустіть `pytest tests/test_crawler.py`: тести працюють проти збереженого
   фрагмента розмітки й покажуть, який селектор відвалився.
3. **Несумісна версія Scrapy.** Потрібна `>= 2.13`: у ній `start_requests()`
   замінили на `async start()`, і спайдер, що визначає лише старий метод,
   не робить жодного запиту й не повідомляє про помилку.

Ознаки того, що в контейнері свіжий код: у логу є рядок
`Requesting N of M channels`, а в `Overridden settings` —
`TWISTED_DNS_RESOLVER`, не `DNS_RESOLVER`. Якщо бачите старі —
перезберіть образ (`docker compose build`).

### Логи помилок
```bash
# Детальні логи
docker compose logs -f --tail=100 nyan-sender
```

Рівень логування задає `LOG_LEVEL` (типово `INFO`; `DEBUG` показує пропуски
каналів у краулері). Помилки публікації йдуть як `ERROR`, відновлювані збої
LLM — як `INFO`.

### Тестування з'єднання  
```bash
# Тест MongoDB з'єднання
docker compose exec nyan-app python3 -c "
from nyan.mongo import get_documents_collection
try:
    collection = get_documents_collection('configs/mongo_config.json')
    print('✅ MongoDB з\'єднання успішне')
    print(f'📊 Документів у колекції: {collection.count_documents({})}')
except Exception as e:
    print(f'❌ Помилка з\'єднання з MongoDB: {e}')
"
```
