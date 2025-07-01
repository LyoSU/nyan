#!/usr/bin/env python3

import json
import os
import sys

def create_mongo_config():
    """Створює mongo_config.json з правильними типами даних"""
    
    config = {
        "client": {
            "host": os.getenv("MONGO_HOST", "localhost"),
            "port": int(os.getenv("MONGO_PORT", "27017"))
        },
        "database_name": "main",
        "documents_collection_name": "documents", 
        "annotated_documents_collection_name": "annotated_documents",
        "clusters_collection_name": "clusters"
    }
    
    # Додаємо username і password тільки якщо вони задані
    username = os.getenv("MONGO_USERNAME", "").strip()
    password = os.getenv("MONGO_PASSWORD", "").strip()
    auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin").strip()
    
    if username:
        config["client"]["username"] = username
    if password:
        config["client"]["password"] = password
    if username and auth_source:
        config["client"]["authSource"] = auth_source
    
    return config

if __name__ == "__main__":
    try:
        config = create_mongo_config()
        
        # Записуємо конфігурацію
        with open("/app/configs/mongo_config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        print("📝 MongoDB конфігурація створена:")
        print(json.dumps(config, indent=4))
        
    except Exception as e:
        print(f"❌ Помилка створення конфігурації: {e}")
        sys.exit(1)
