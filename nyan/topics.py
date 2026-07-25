import argparse
import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Template

from nyan import rich
from nyan.clusters import Clusters
from nyan.client import TelegramClient
from nyan.util import format_dt_uk, get_current_ts, ts_to_dt, PUBLISH_CHANNEL_URL
from nyan.openai import openai_completion, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
from nyan.mongo import get_topics_collection


BASE_DIR = Path(__file__).parent

# The digest is also mirrored to this issue, which collects long-form posts.
SUMMARY_ISSUE = "summary"


def extract_topics(
    clusters: list[dict[str, Any]],
    prompt_path: str,
    model_name: str,
) -> list[dict[str, Any]]:
    with open(prompt_path) as f:
        template = Template(f.read())

    prompt = template.render(clusters=clusters).strip() + "\n"

    messages = [{"role": "user", "content": prompt}]
    content = openai_completion(
        messages=messages,
        model_name=model_name,
        response_format={"type": "json_object"},
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )

    content = content[content.find("{") : content.rfind("}") + 1]
    topics: list[dict[str, Any]] = json.loads(content)["topics"]
    return [topic for topic in topics if topic.get("name") and topic.get("titles")]


def render_topics(topics: list[dict[str, Any]], duration_hours: int) -> list[rich.Block]:
    """The digest as a block tree.

    Each headline is a link in its entirety, which is why the prompt no longer
    has to name a verb for a link to be spliced onto: matching a model-provided
    word back into its own sentence was the most fragile step in this file.
    """
    blocks: list[rich.Block] = [
        rich.heading(f"Головне за {duration_hours} годин", size=2)
    ]
    for topic in topics:
        title = " ".join(
            part for part in (topic.get("emojis", ""), topic["name"]) if part
        )
        blocks.append(rich.heading(title, size=4))
        items = []
        for entry in topic["titles"]:
            text = (entry.get("title") or "").strip()
            url = entry.get("url")
            if not text:
                continue
            items.append([rich.paragraph(rich.link(text, url) if url else text)])
        if items:
            blocks.append(rich.bullet_list(*items))

    blocks.append(rich.divider())
    blocks.append(
        rich.footer(format_dt_uk(ts_to_dt(get_current_ts())))
    )
    return blocks


def collect_clusters(
    mongo_config_path: str, duration_hours: int, issue_name: str
) -> list[dict[str, Any]]:
    duration = int(duration_hours * 3600)
    clusters_obj = Clusters.load_from_mongo(
        mongo_config_path, get_current_ts(), duration
    )
    clusters = list(clusters_obj.clid2cluster.values())
    clusters.sort(key=lambda cl: cl.create_time if cl.create_time else 0)

    fixed_clusters = []
    for cluster in clusters:
        messages = [m for m in cluster.messages if m.issue == issue_name]
        if not messages:
            continue
        message = messages[0]
        date_str = ""
        if cluster.create_time:
            date_str = format_dt_uk(ts_to_dt(cluster.create_time))
        fixed_clusters.append(
            {
                "url": f"{PUBLISH_CHANNEL_URL}/{message.message_id}",
                "dt": date_str,
                "views": cluster.views,
                "sources_count": len(cluster.channels),
                "text": cluster.annotation_doc.patched_text,
            }
        )
    return fixed_clusters


def main(
    mongo_config_path: str,
    client_config_path: str,
    duration_hours: int,
    max_news_count: int,
    min_news_count: int,
    issue_name: str,
    prompt_path: str,
    model_name: str,
    auto: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fixed_clusters = collect_clusters(mongo_config_path, duration_hours, issue_name)
    if len(fixed_clusters) < min_news_count:
        logging.info(
            "Only %d posts in the last %d hours, need %d",
            len(fixed_clusters),
            duration_hours,
            min_news_count,
        )
        return
    fixed_clusters = fixed_clusters[-max_news_count:]

    topics = extract_topics(
        fixed_clusters, prompt_path=prompt_path, model_name=model_name
    )
    if not topics:
        logging.warning("No topics extracted")
        return

    blocks = render_topics(topics, int(duration_hours))
    for topic in topics:
        logging.info(
            "%s: %d headlines", topic["name"], len(topic.get("titles", []))
        )

    should_publish = auto
    if not auto:
        should_publish = input("Publish? y/n ").strip() == "y"

    if should_publish:
        with TelegramClient(client_config_path) as client:
            client.send_rich_message(blocks, issue_name=issue_name)
            client.send_rich_message(blocks, issue_name=SUMMARY_ISSUE)

    collection = get_topics_collection(mongo_config_path)
    collection.insert_one({"clusters": fixed_clusters, "topics": topics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-config-path", type=str, required=True)
    parser.add_argument("--client-config-path", type=str, required=True)
    parser.add_argument("--duration-hours", type=int, default=8)
    parser.add_argument("--max-news-count", type=int, default=30)
    parser.add_argument("--min-news-count", type=int, default=5)
    parser.add_argument("--issue-name", type=str, default="main")
    parser.add_argument(
        "--prompt-path", type=str, default=str(BASE_DIR / "prompts/topics.txt")
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--auto", default=False, action="store_true")
    args = parser.parse_args()
    main(**vars(args))
