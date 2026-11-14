"""Render real posts of a week in the old and the new presentation, side by side.

Nothing is sent and nothing is asked: the posts are taken from a dump of
published clusters with the text they went out with, the LLM is cut off so a
stale generation cannot start a rewrite, and the editor's decisions are given
per post in `CASES` (where each one came from is said there).

    python scripts/presentation_demo.py --dump dump.jsonl --out demo.json
"""

import argparse
import json
import os
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
    "short_legend": True,
    "layout": True,
    "channel_urls": {"main": PUBLISH_CHANNEL_URL},
}

# clid -> what the editor would have attached to the post, and why.
CASES: list[dict[str, Any]] = [
    {
        "clid": 66325,
        "level": "breaking",
        "updates": [{"time_of": 66332, "text": "Загиблих стало троє, постраждалих — 11"}],
        "rewritten_by": 66332,
        "author_file": "author_update_66325.json",
        "note": "Удар по ринку, одна загибла. О 14:38 прод випустив окремий пост-відповідь "
        "66332 «кількість загиблих зросла до трьох»; тут це те саме повідомлення 66325, "
        "відредаговане: update переписує пост цілком. Заголовок і текст написав автор "
        "(реальний виклик із новим правилом про заголовок оновлення і з найсвіжішими "
        "постами кожного каналу). Рядок «оновлено» — поле change з реальної відповіді "
        "редактора (ізольований виклик). Рівень breaking («Важливо») — за правилом промпту "
        "(загибель людей); редактор цю історію в програванні не бачив.",
    },
    {
        "clid": 66332,
        "level": "normal",
        "note": "Те, що прод надіслав окремою відповіддю. З редактором окремого поста немає: "
        "relation=update, і його зміст іде рядком у 66325 (вище). Показано для порівняння.",
        "dropped": True,
    },
    {
        "clid": 66361,
        "level": "normal",
        "updates": [
            {"time_of": 66362, "text": "постраждала ще одна людина"},
            {"time_of": 66365, "text": "одна з постраждалих померла"},
        ],
        "rewritten_by": 66365,
        "author_file": "author_update_66361.json",
        "note": "Два оновлення одного поста. 12.11 прод випустив три пости за 16 хвилин "
        "(09:06 «постраждала людина», 09:15 «постраждалих двоє», 09:22 «загинула жінка»); "
        "тут це один пост. Заголовок і текст написав автор двома послідовними переписуваннями "
        "(реальні виклики). Рядок «оновлено» про останню зміну, історія згорнута в "
        "«Оновлення: 2» і починається з початкового заголовка. Тексти рядків оновлення "
        "написав я в стилі нового правила для change (різниця з постом), не редактор.",
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
        "ілюстративний, щоб показати формат. «Пишуть окремі джерела» в minor не показуємо. "
        "Фото посту тижневої давності, посилання протухли — тут без фото; стан із фото "
        "див. нижче.",
    },
    {
        "clid": 66488,
        "level": "minor",
        "also_without_media": True,
        "note": "Minor зі свіжим фото (14.11) без верстки автора: фото не в тілі, а першим у "
        "«Докладніше» (details приймає будь-які InputRichBlock, медіа теж). Нижче — той "
        "самий пост без фото взагалі, для порівняння. Рівень ілюстративний. (Автор у "
        "своїй верстці для цієї історії сказав place=show — тоді фото лишилося б у тілі.)",
    },
    {
        "clid": 66470,
        "level": "breaking",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА, важливий зі свіжим фото в тілі. Автор бачив 3 фото (low) і "
        "поставив у тіло кадр із Зеленським. Рівень breaking — мій вибір для зразка (рішення "
        "держави, що стосується багатьох: обмін 1200 на 1200). Зауваження: кластер у проді "
        "змішав кілька новин (Starshield, FREYJA, Данія), і автор склав їх у список — це "
        "вада кластеризації, не верстки.",
    },
    {
        "clid": 66485,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА, normal зі свіжим фото в тілі. Автор побачив, що перше фото — "
        "портрет із текстом поверх, і поставив першим архівне фото з колегами (first=2). "
        "Згоден.",
    },
    {
        "clid": 66473,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА: пряма мова. Автор узяв quote зі словами Зеленського — "
        "заради них новина. Згоден.",
    },
    {
        "clid": 66463,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА: подія з відео. Автор бачив прев'ю трьох відео з підписом "
        "«відео, 0:58» тощо, третє правильно назвав кадром телеефіру (kind=text). Список із "
        "трьох окремих фактів доречний.",
    },
    {
        "clid": 66447,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА: графік відключень. Абзац і список із двох пунктів (населення / "
        "промисловість); медіа автор не взяв (place=none). Згоден: картинка-заставка "
        "графіка нічого не дає без самих черг.",
    },
    {
        "clid": 66462,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА: ціни на пальне. Автор побачив, що фото — скриншоти таблиць "
        "цін, і згорнув їх під «Фото й відео». Тут я з автором НЕ згоден у заголовку: "
        "«Бензин подешевшав на 3 гривні» загальніше за факт (подешевшав лише А-95 Energy в "
        "«Укрнафті»); у проді заголовок точніший. Таблицю він не взяв — для одного числа "
        "й не треба.",
    },
    {
        "clid": 66486,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА: удар по Харкову, два відео (прев'ю + «відео, 0:05»). Абзац і "
        "attributed. Згоден.",
    },
    {
        "clid": 66237,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА без зображень (фото тижневої давності не завантажилися): "
        "розслідування. Один абзац із найпомітнішим (басейн, Rolls-Royce, крокодил) і "
        "attributed. Для розслідування коротко, але не переверстано.",
    },
    {
        "clid": 66115,
        "level": "normal",
        "author_layout": True,
        "note": "ВЕРСТКА АВТОРА без зображень: числа по складах. Абзац + список із двох "
        "наслідків + attributed; таблицю не взяв (чисел для порівняння по рядках немає). "
        "Згоден.",
    },
    {
        "clid": 66421,
        "level": "minor",
        "note": "Minor-цитата: пост одного каналу його словами, довгий текст згорнуто в "
        "expandable_blockquote з підписом каналу. Рівень ілюстративний.",
    },
    {
        "clid": 66330,
        "level": "normal",
        "note": "Звичайний пост без оновлень і без теми: новий вигляд збігається зі старим, "
        "крім легенди значків у джерелах — три значки й більше тепер ідуть двома рядками.",
    },
]


# How old a post's media may be and still load.
FRESH_MEDIA = 30 * 3600


def strip_media(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same blocks with every photo, video and slideshow taken out, at any depth."""
    kept = []
    for block in blocks:
        if block.get("type") in ("photo", "video", "animation", "slideshow", "collage"):
            continue
        block = dict(block)
        if isinstance(block.get("blocks"), list):
            block["blocks"] = strip_media(block["blocks"])
            if block.get("type") == "details" and not block["blocks"]:
                # A minor post's "Докладніше" that held only the photo.
                continue
        if isinstance(block.get("items"), list):
            block["items"] = [
                {**item, "blocks": strip_media(item["blocks"])} for item in block["items"]
            ]
        kept.append(block)
    return kept


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
        if "rewritten_by" in case:
            wanted.add(case["rewritten_by"])
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

    folder = os.path.dirname(os.path.abspath(args.out))
    layouts: dict[int, dict[str, Any]] = {}
    if os.path.exists(f"{folder}/layout_answers.jsonl"):
        with open(f"{folder}/layout_answers.jsonl") as r:
            for line in r:
                answer = json.loads(line)
                if answer["analysis"].get("headline"):
                    layouts[answer["clid"]] = answer["analysis"]
    # Media links of a few days ago no longer load; such posts go without media.
    newest = max(post.create_time or 0 for post in posts.values())
    out = []
    text: list[str] = []
    for case in CASES:
        post = posts[case["clid"]]
        fresh = (post.create_time or 0) >= newest - FRESH_MEDIA
        before = old.render_cluster(post, "main")
        assert before is not None and before.blocks is not None
        headline = post.stored_headline or flatten(before.blocks[0].get("text"))
        if "rewritten_by" in case:
            # What the forced rewrite would have made of it: the later post's
            # text over the sources of both.
            later = posts[case["rewritten_by"]]
            for doc in later.docs:
                post.add(doc)
            post.saved_analysis = dict(later.saved_analysis or {})
            if "author_file" in case:
                with open(f"{folder}/{case['author_file']}") as r:
                    post.saved_analysis = json.load(r)
        if case.get("author_layout"):
            # The author's own layout of this story, from a real call.
            post.saved_analysis = dict(layouts[post.clid])
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
        variants = [("НОВИЙ ВИГЛЯД", after.blocks if fresh else strip_media(after.blocks))]
        if case.get("also_without_media"):
            variants.append(("НОВИЙ ВИГЛЯД, БЕЗ ФОТО", strip_media(after.blocks)))
        if not fresh:
            before.blocks = strip_media(before.blocks)
        reply = None
        if post.reply_to_clid and post.reply_to_clid in posts:
            reply = {"message_id": posts[post.reply_to_clid].messages[0].message_id}
            if after.reply_quote:
                reply["quote_rich_texts"] = after.reply_quote
        entry = {
            "clid": post.clid,
            "published": ts_to_dt(post.create_time or 0).strftime("%d.%m %H:%M"),
            "headline": headline,
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
                "rich_message": {"blocks": variants[0][1]},
                "disable_notification": not after.notify,
                "reply_parameters": reply,
            },
            "media": "kept" if fresh else "removed: links of a few days ago no longer load",
        }
        if after.fallback and not case.get("dropped"):
            # What goes out instead if Telegram refuses the author's layout.
            entry["new_fallback"] = {
                "rich_message": {"blocks": after.fallback if fresh else strip_media(after.fallback)}
            }
        if len(variants) > 1:
            entry["new_without_media"] = {
                "rich_message": {"blocks": variants[1][1]},
                "disable_notification": not after.notify,
                "reply_parameters": reply,
            }
        out.append(entry)
        text.append("=" * 78)
        text.append(f"{post.clid} · {entry['published']} · {headline}")
        text.append(case["note"])
        if not fresh:
            text.append("(медіа прибрано в обох версіях: посилання кількаденної давності протухли)")
        text.append("--- СТАРИЙ ВИГЛЯД (як у проді)")
        text.extend(as_text(before.blocks))
        if case.get("dropped"):
            text.append("--- НОВИЙ ВИГЛЯД")
            text.append("(окремого поста немає — див. рядок «оновлено» у 66325)")
            continue
        for title, blocks in variants:
            text.append(f"--- {title}")
            if reply:
                quote = flatten(reply.get("quote_rich_texts") or "")
                text.append(f"[відповідь на {reply['message_id']}; цитата: «{quote}»]")
            text.append(f"[звук: {'так' if after.notify else 'ні'}]")
            text.extend(as_text(blocks))
    with open(args.out, "w") as w:
        json.dump(out, w, ensure_ascii=False, indent=1)
    with open(args.out.removesuffix(".json") + ".txt", "w") as w:
        w.write("\n".join(text) + "\n")
    print(f"{len(out)} posts -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
