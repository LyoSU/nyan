# НЯН

[![Tests](https://github.com/LyoSU/nyan/actions/workflows/python.yml/badge.svg)](https://github.com/LyoSU/nyan/actions/workflows/python.yml)
[![Telegram](https://img.shields.io/badge/Telegram-UAliveNews-blue.svg?logo=telegram)](https://t.me/UAliveNews)
[![License](https://img.shields.io/github/license/LyoSU/nyan)](LICENSE)

Новинний агрегатор для Telegram: збирає пости з українських каналів,
об'єднує повідомлення про одну й ту саму подію в кластери й публікує спільну
стрічку. Кожне джерело належить до групи довіри, тож читач бачить, наскільки
можна довіряти повідомленню.

Канал: [UAliveNews](https://t.me/UAliveNews)

Форк [NyanNyanovich/nyan](https://github.com/NyanNyanovich/nyan), адаптований
під українську мову: власні групи джерел, українські заголовки від LLM і пости
у форматі rich messages (Bot API 10.1+).

## Як це працює

1. **Crawler** (`crawler/`) читає веб-версію каналів через Scrapy й складає
   пости в MongoDB.
2. **Annotator** (`nyan/annotator.py`) чистить текст, визначає мову й категорію,
   рахує ембединги, викидає рядки, які канал повторює в більшості постів
   (підписи «Підпишись на…»).
3. **Clusterer** (`nyan/clusterer.py`) групує пости про одну подію за косинусною
   відстанню між ембедингами, зі штрафами за час і за однаковий канал.
4. **Ranker** (`nyan/ranker.py`) відбирає, що варте публікації, за швидкістю
   набору переглядів.
5. **Renderer** (`nyan/renderer.py`) будує пост із типізованих блоків. Подію,
   про яку написало кілька каналів, LLM зводить в один текст (`nyan/summary.py`);
   якщо джерело одне або виклик не вдався — цитується текст обраного каналу.
6. **Client** (`nyan/client.py`) публікує його через `sendRichMessage`.

## Запуск у Docker

Найпростіший шлях. Потрібна зовнішня MongoDB.

```bash
cp .env.example .env      # впишіть свої MongoDB та LLM налаштування
./test-docker.sh          # перевірка конфігурації та з'єднання
./start-docker.sh         # запуск crawler + sender
```

Деталі, змінні середовища й типові проблеми — у [DOCKER_README.md](DOCKER_README.md).

## Запуск без Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash download_models.sh
```

Вкажіть Telegram-креденшели у `configs/client_config.json`
(шаблон — `configs/client_config.json.example`) і MongoDB у
`configs/mongo_config.json`.

```bash
./crawl.sh    # збір постів
./send.sh     # обробка й публікація
```

## Налаштування

| Файл | Що задає |
|---|---|
| `channels.json` | Список каналів, їхні групи довіри та стрічки |
| `configs/renderer_config.json` | Формат поста (`rich`/`legacy`), таймзона |
| `configs/ranker_config.json` | Пороги публікації для кожної стрічки |
| `configs/clusterer_config.json` | Штрафи й поріг кластеризації |
| `configs/annotator_config.json` | Моделі, чистка тексту, детекція шаблонних рядків |
| `configs/daemon_config.json` | Вікна часу та інтервали демона |
| `.env` | MongoDB, LLM, канал публікації |

LLM використовується для заголовків, зведення новини з кількох джерел і
дайджесту. Підходить будь-який OpenAI-сумісний шлюз — задайте `LLM_API_KEY`,
`LLM_BASE_URL` і `LLM_MODEL`. Без ключа НЯН працює: пости будуть без заголовків
і з текстом одного каналу замість зведеного.

## Дайджест

Добірка публікується в **окремий канал** — задайте `DIGEST_CHANNEL_ID` у `.env`
(можна юзернеймом, `@ShortUA`, або числовим id, `-1001925661350`). Якщо в
`configs/client_config.json` є issue з назвою `digest`, виграє він: там мав би
лежати окремий `bot_token`, якщо добірки постить інший бот.

```bash
python3 -m nyan.digest --mongo-config-path configs/mongo_config.json \
    --client-config-path configs/client_config.json --duration-hours 8
```

Без `--auto` покаже JSON добірки й спитає підтвердження.

Вікно рахується від останньої **опублікованої** добірки, а не від «зараз мінус
8 годин». Тому якщо новин було мало й добірка не вийшла, ці пости не губляться:
вони потраплять у наступну.

## Розробка

```bash
pip install -r requirements-dev.txt
pre-commit install

ruff check .    # лінтер
mypy            # типи
pytest -q       # тести
```

Тести порівнюють вихід із записаними снапшотами в `tests/data/` (Git LFS).
Якщо зміна свідомо змінює результат, перегенеруйте їх:

```bash
python3 -m tests.canonize
```
