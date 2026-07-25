#!/usr/bin/env python3
"""Генерує configs/mongo_config.json зі змінних середовища.

Викликається docker-entrypoint.sh перед запуском будь-якої команди: pymongo
чекає на типізовані значення (port — число, ssl — булеве), а середовище
віддає лише рядки.
"""

import argparse
import json
import os
import sys
from typing import Any


DEFAULT_OUTPUT_PATH = "configs/mongo_config.json"


def env_flag(name: str, default: str) -> bool:
    return (os.getenv(name) or default).strip().lower() in ("true", "1", "yes")


def env_str(name: str, default: str = "") -> str:
    # `or default`, не getenv-default: compose передає незадані змінні як
    # порожні рядки, які default не замінив би.
    return (os.getenv(name) or default).strip()


def create_mongo_config() -> dict[str, Any]:
    client: dict[str, Any] = {
        "host": env_str("MONGO_HOST", "localhost"),
        "port": int(env_str("MONGO_PORT", "27017")),
        "readPreference": env_str("MONGO_READ_PREFERENCE", "primary"),
        "ssl": env_flag("MONGO_SSL", "false"),
        "directConnection": env_flag("MONGO_DIRECT_CONNECTION", "true"),
    }

    username = env_str("MONGO_USERNAME")
    password = env_str("MONGO_PASSWORD")
    auth_source = env_str("MONGO_AUTH_SOURCE", "admin")
    if username:
        client["username"] = username
        if auth_source:
            client["authSource"] = auth_source
    if password:
        client["password"] = password

    return {
        "client": client,
        "database_name": env_str("MONGO_DATABASE", "main"),
        "documents_collection_name": "documents",
        "annotated_documents_collection_name": "annotated_documents",
        "clusters_collection_name": "clusters",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        config = create_mongo_config()
        with open(args.output_path, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")
    except Exception as e:
        print(f"❌ Помилка створення конфігурації: {e}", file=sys.stderr)
        sys.exit(1)

    client = config["client"]
    auth = "з автентифікацією" if "username" in client else "без автентифікації"
    print(
        "📝 MongoDB конфігурація: {}:{}/{} ({}) → {}".format(
            client["host"],
            client["port"],
            config["database_name"],
            auth,
            args.output_path,
        )
    )


if __name__ == "__main__":
    main()
