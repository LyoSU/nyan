import argparse
from collections import defaultdict

from nyan.channels import Channels
from nyan.client import TelegramClient


def list_channels(client_config_path, channels_path):
    client = TelegramClient(client_config_path)
    channels = Channels(channels_path)

    groups = defaultdict(list)
    for _, ch in channels:
        main_group = ch.groups.get("main", "unknown")
        groups[main_group].append(ch)

    text = ""
    for _, group in groups.items():
        if not group:
            continue
        first_channel = group[0]
        emoji = first_channel.emojis.get("main", "") if first_channel.emojis else ""
        text += emoji
        for ch in group:
            text += '<a href="https://t.me/{}">{}</a> • '.format(ch.name, ch.alias)
        text += "\n\n"
    print(text)
    client.send_message(text, "main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels-path", type=str, required=True)
    parser.add_argument("--client-config-path", type=str, default="configs/client_config.json")
    args = parser.parse_args()
    list_channels(**vars(args))
