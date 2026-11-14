"""Render real posts of a week in the old and the new presentation, side by side.

Nothing is sent and nothing is asked: the posts are taken from a dump of
published clusters with the text they went out with, the LLM is cut off so a
stale generation cannot start a rewrite, and the editor's decisions are given
per post in `CASES` (where each one came from is said there).

    python scripts/presentation_demo.py --dump dump.jsonl --out demo.json
"""

import argparse
import json
import sys
from typing import Any

import nyan.clusters
from nyan.channels import Channels
from nyan.clusters import Cluster
from nyan.renderer import Renderer
from nyan.util import PUBLISH_CHANNEL_URL, ts_to_dt

PRESENTATION = {
    "levels": True,
    "update_line": True,
    "more_on_this": True,
    "quote_parent": True,
    "channel_urls": {"main": PUBLISH_CHANNEL_URL},
}

# clid -> what the editor would have attached to the post, and why.
CASES: list[dict[str, Any]] = [
    {
        "clid": 66325,
        "level": "breaking",
        "updates": [{"time_of": 66332, "text": "Загиблих стало троє, постраждалих — 11"}],
        "note": "Удар по ринку, одна загибла. О 14:38 прод випустив окремий пост-відповідь "
        "66332 «кількість загиблих зросла до трьох»; тут це рядок «оновлено» в тому самому "
        "пості. Текст рядка — поле change з реальної відповіді редактора (ізольований виклик). "
        "Рівень breaking — за правилом промпту (загибель людей); редактор цю історію в "
        "програванні не бачив.",
    },
    {
        "clid": 66332,
        "level": "normal",
        "note": "Те, що прод надіслав окремою відповіддю. З редактором окремого поста немає: "
        "relation=update, і його зміст іде рядком у 66325 (вище). Показано для порівняння.",
        "dropped": True,
    },
    {
        "clid": 66290,
        "level": "normal",
        "storyline": "Переговори в Абу-Дабі",
        "storyline_of": [66245],
        "updates": [{"time": 1794404700, "text": "Домовленість про обмін підтверджено"}],
        "note": "Продовження (відповідь на 66268) з цитатою заголовка батька, «Ще про це» з "
        "постом тієї самої теми, що не є батьком, і рядок «оновлено» з реальної відповіді "
        "редактора 11.11 14:45 (ізольований виклик).",
    },
    {
        "clid": 66334,
        "level": "normal",
        "note": "Продовження, відповідь на 66290 з цитатою його заголовка.",
    },
    {
        "clid": 66318,
        "level": "normal",
        "storyline": "Удари по Харкову",
        "storyline_of": [66271],
        "note": "Удар по Салтівці 11.11 08:45. Редактор у програванні з новим промптом "
        "опублікував його як storyline «Удари по Харкову»; «Ще про це» веде на удар 09.11.",
    },
    {
        "clid": 66317,
        "level": "normal",
        "storyline": "Переговори в Абу-Дабі",
        "storyline_of": [66290, 66268, 66245],
        "note": "Окрема подія теми переговорів: три попередні пости теми, новіші вище.",
    },
    {
        "clid": 66326,
        "level": "minor",
        "note": "Як виглядає minor. Редактор цю монету відправив у skip; рівень тут "
        "ілюстративний, щоб показати формат.",
    },
    {
        "clid": 66330,
        "level": "normal",
        "note": "Звичайний пост без оновлень і без теми: новий вигляд має збігатися зі старим.",
    },
]


def flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(flatten(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "url":
            return f"[{flatten(value.get('text'))}]({value.get('url')})"
        if value.get("type") == "bold":
            return f"**{flatten(value.get('text'))}**"
        if value.get("type") == "italic":
            return f"_{flatten(value.get('text'))}_"
        if value.get("type") == "marked":
            return f"=={flatten(value.get('text'))}=="
        return flatten(value.get("text", ""))
    return ""


def as_text(blocks: list[dict[str, Any]], indent: str = "") -> list[str]:
    """A rough reading of the block tree, for a person; the JSON is what is sent."""
    lines: list[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            lines.append(f"{indent}# {flatten(block['text'])}")
        elif kind in ("paragraph", "footer"):
            lines.append(f"{indent}{flatten(block['text'])}")
        elif kind == "divider":
            lines.append(f"{indent}———")
        elif kind == "details":
            lines.append(f"{indent}▸ {flatten(block['summary'])} (згорнуто)")
            lines.extend(as_text(block["blocks"], indent + "    "))
        elif kind == "expandable_blockquote":
            credit = f" — {flatten(block['credit'])}" if block.get("credit") else ""
            lines.append(f"{indent}❝ (згорнута цитата) {flatten(block['text'])}{credit}")
        elif kind == "list":
            for item in block["items"]:
                inner = as_text(item["blocks"], indent + "  ")
                if inner:
                    lines.append(f"{indent}• {inner[0].strip()}")
                    lines.extend(inner[1:])
        elif kind == "table":
            if block.get("caption"):
                lines.append(f"{indent}[таблиця: {flatten(block['caption'])}]")
            for row in block["cells"]:
                lines.append(f"{indent}| " + " | ".join(flatten(c.get("text")) for c in row) + " |")
        elif kind in ("photo", "video", "animation"):
            lines.append(f"{indent}[{kind}]")
        elif kind == "slideshow":
            lines.append(f"{indent}[слайдшоу: {len(block['blocks'])}]")
        else:
            lines.append(f"{indent}[{kind}]")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--renderer-config", default="configs/renderer_config.json")
    ap.add_argument("--channels", default="channels.json")
    args = ap.parse_args()

    def no_llm(**kwargs: Any) -> str:
        raise RuntimeError("the demo asks nothing")

    nyan.clusters.openai_completion = no_llm  # type: ignore[attr-defined]

    wanted = {case["clid"] for case in CASES}
    for case in CASES:
        wanted.update(case.get("storyline_of", []))
        wanted.update(u["time_of"] for u in case.get("updates", []) if "time_of" in u)
    posts: dict[int, Cluster] = {}
    with open(args.dump) as r:
        for line in r:
            record = json.loads(line)
            if record.get("clid") in wanted:
                posts[record["clid"]] = Cluster.fromdict(record)
    parents = {p.reply_to_clid for p in posts.values() if p.reply_to_clid}
    with open(args.dump) as r:
        for line in r:
            record = json.loads(line)
            if record.get("clid") in parents and record["clid"] not in posts:
                posts[record["clid"]] = Cluster.fromdict(record)

    channels = Channels(args.channels)
    old = Renderer(args.renderer_config, channels)
    with open(args.renderer_config) as r:
        config = json.load(r)
    config["presentation"] = PRESENTATION
    new_config = args.out + ".renderer_config.json"
    with open(new_config, "w") as w:
        json.dump(config, w, ensure_ascii=False)
    new = Renderer(new_config, channels)

    out = []
    text: list[str] = []
    for case in CASES:
        post = posts[case["clid"]]
        before = old.render_cluster(post, "main")
        if post.reply_to_clid and post.reply_to_clid in posts:
            parent = posts[post.reply_to_clid]
            post.reply_to_headline = parent.stored_headline or ""
        post.level = case.get("level", "normal")
        post.storyline = case.get("storyline")
        post.updates = [
            {
                "time": posts[u["time_of"]].create_time if "time_of" in u else u["time"],
                "text": u["text"],
            }
            for u in case.get("updates", [])
        ]
        post.storyline_posts = [
            {
                "clid": other.clid,
                "message_id": other.messages[0].message_id,
                "issue": "main",
                "time": other.create_time,
                "headline": other.stored_headline,
            }
            for other in (posts[clid] for clid in case.get("storyline_of", []))
        ]
        after = new.render_cluster(post, "main")
        assert before is not None and after is not None
        assert before.blocks is not None and after.blocks is not None
        reply = None
        if post.reply_to_clid and post.reply_to_clid in posts:
            reply = {"message_id": posts[post.reply_to_clid].messages[0].message_id}
            if after.reply_quote:
                reply["quote_rich_texts"] = after.reply_quote
        entry = {
            "clid": post.clid,
            "published": ts_to_dt(post.create_time or 0).strftime("%d.%m %H:%M"),
            "headline": post.stored_headline,
            "note": case["note"],
            "editor": {
                "level": post.level,
                "storyline": post.storyline,
                "updates": post.updates,
                "storyline_posts": post.storyline_posts,
            },
            "old": {"rich_message": {"blocks": before.blocks}, "disable_notification": True},
            "new": None
            if case.get("dropped")
            else {
                "rich_message": {"blocks": after.blocks},
                "disable_notification": not after.notify,
                "reply_parameters": reply,
            },
        }
        out.append(entry)
        text.append("=" * 78)
        text.append(f"{post.clid} · {entry['published']} · {post.stored_headline}")
        text.append(case["note"])
        text.append("--- СТАРИЙ ВИГЛЯД (як у проді)")
        text.extend(as_text(before.blocks))
        text.append("--- НОВИЙ ВИГЛЯД")
        if case.get("dropped"):
            text.append("(окремого поста немає — див. рядок «оновлено» у 66325)")
        else:
            if reply:
                quote = flatten(reply.get("quote_rich_texts") or "")
                text.append(f"[відповідь на {reply['message_id']}; цитата: «{quote}»]")
            text.append(f"[звук: {'так' if after.notify else 'ні'}]")
            text.extend(as_text(after.blocks))
    with open(args.out, "w") as w:
        json.dump(out, w, ensure_ascii=False, indent=1)
    with open(args.out.removesuffix(".json") + ".txt", "w") as w:
        w.write("\n".join(text) + "\n")
    print(f"{len(out)} posts -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
