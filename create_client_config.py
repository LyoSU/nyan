#!/usr/bin/env python3
"""Генерує configs/client_config.json зі змінної CLIENT_CONFIG_JSON.

Близнюк create_mongo_config.py, і з тієї ж причини: `configs/` тепер їде в
образі разом із кодом, тому два файли, яких не можна тримати в git — доступи до
Mongo і токени публікації — народжуються при старті контейнера зі середовища.

Одна змінна з цілим JSON, а не окремі змінні на кожне поле: `issues` — це
список, і схема зі змінних, яка мусила б рости разом зі списком, рано чи пізно
від нього відстане.

Значення перевіряється тут, поки контейнер лише стартує. Найімовірніший спосіб
це зламати — вставити обрізаний JSON у поле Coolify, а помилка, знайдена в
entrypoint, коштує один рядок у логах; та сама помилка всередині демона — цикл
рестартів із трейсбеком, який ніхто не читає.
"""

import argparse
import json
import os
import sys
from typing import Any


DEFAULT_OUTPUT_PATH = "configs/client_config.json"
ENV_NAME = "CLIENT_CONFIG_JSON"

#: `discussion_id` не тут: канал із вимкненими комментарями не має групи
#: обговорення, і вимагати її означало б вимагати вигадане значення.
REQUIRED_ISSUE_FIELDS = ("name", "channel_id", "bot_token")


class ClientConfigError(Exception):
    """Значення непридатне. Повідомлення йде в логи, тому без токенів."""


def build_client_config(raw: str | None) -> dict[str, Any]:
    # Не `if raw is None`: compose віддає незадану змінну порожнім рядком, а
    # значення, вставлене у веб-форму, приходить із доданим браузером переносом.
    if raw is None or not raw.strip():
        raise ClientConfigError(
            f"Змінна {ENV_NAME} не задана. Без неї нічого не опублікувати: "
            "додайте її в оточення застосунку."
        )

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ClientConfigError(
            f"Значення {ENV_NAME} не є коректним JSON ({e.msg}, позиція {e.pos}). "
            "Найчастіше значення обрізалося при вставлянні."
        ) from e

    if not isinstance(config, dict):
        raise ClientConfigError(
            f"{ENV_NAME} має бути об'єкт із ключем 'issues', а не "
            f"{type(config).__name__}. Схоже, список issues вставлено без обгортки."
        )

    issues = config.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ClientConfigError(
            f"{ENV_NAME} має містити непорожній список 'issues': інакше жоден "
            "пост не має куди піти."
        )

    for position, issue in enumerate(issues, start=1):
        if not isinstance(issue, dict):
            raise ClientConfigError(
                f"issues[{position}] має бути об'єктом, а не {type(issue).__name__}."
            )
        # Назвати issue по імені, якщо воно є: у списку з трьох каналів номер
        # позиції нічого не підказує тому, хто це виправляє.
        named = issue.get("name")
        where = f"issue '{named}'" if isinstance(named, str) else f"issues[{position}]"
        for field in REQUIRED_ISSUE_FIELDS:
            if field not in issue:
                raise ClientConfigError(f"У {where} немає обов'язкового '{field}'.")

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        config = build_client_config(os.getenv(ENV_NAME))
    except ClientConfigError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    with open(args.output_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    names = ", ".join(str(issue["name"]) for issue in config["issues"])
    print(f"📝 Конфігурація публікації: {names} → {args.output_path}")


if __name__ == "__main__":
    main()
