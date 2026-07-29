"""Names the rubrics nobody has written a pattern for yet.

A rubric is recurring by definition: the same channel writes the same opening at
the same hour every day. So the documents can be asked which openings recur, and
the answer turns out to name the rubrics on its own — the minute of silence, the
siren, the digest, the horoscope — including the ones no pattern in
`annotator_config.json` reaches.

What this deliberately does NOT do is filter anything. Recurrence alone is a bad
judge: the general staff's daily situation report and Ukrenergo's grid status
recur exactly as reliably as the horoscope, and they are the news. Left to
decide, this would quietly delete a channel whose routine is its reporting.
So it proposes and a human disposes: run it, read the list, and add to the config
only the lines that are genuinely a channel talking about itself.

    python -m scripts.find_rubrics --days 7 --min-days 4

Reads `annotated_documents`, which is the right collection to ask: it holds
exactly the text the detector was given, so a shape reported as uncaught here is
uncaught in production too.
"""

import argparse
import datetime
import json
import re
from collections import defaultdict

from bson import ObjectId

from nyan.mongo import get_annotated_documents_collection
from nyan.rubrics import RubricDetector


# Everything that varies between two runs of the same rubric: the date, the
# clock, the day of the week, the emoji. What is left is the shape of the line,
# and two posts of one rubric share it.
MONTHS = (
    r"січн|лют|берез|квітн|трав|черв|липн|серп|вересн|жовтн|листопад|грудн"
)
WEEKDAYS = r"понеділк|вівторк|середи|четверг|четверк|п.ятниц|субот|неділ"
DIGITS = re.compile(r"\d+")
NON_WORD = re.compile(r"[^\w\s]+")
SPACES = re.compile(r"\s+")

#: A first line shorter than this is a bare marker, longer than this is a lede
#: that happens to be one sentence. Neither is a heading worth grouping.
MIN_HEADING = 8
MAX_HEADING = 90


def shape(line: str) -> str:
    text = line.lower()
    text = re.sub(MONTHS, "@", text)
    text = re.sub(WEEKDAYS, "@", text)
    text = DIGITS.sub("#", text)
    text = NON_WORD.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def find_rubrics(
    mongo_config_path: str,
    annotator_config_path: str,
    days: int,
    min_days: int,
    include_caught: bool,
) -> None:
    with open(annotator_config_path) as r:
        detector = RubricDetector(json.load(r).get("rubric_detector", {}))

    collection = get_annotated_documents_collection(mongo_config_path)
    # Filtered on _id rather than on fetch_time: fetch_time has no index, and
    # asking half a million documents for it times out.
    since = ObjectId.from_datetime(
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    )

    days_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    example: dict[tuple[str, str], str] = dict()
    total = 0
    for doc in collection.find(
        {"_id": {"$gt": since}},
        {"patched_text": 1, "channel_id": 1, "fetch_time": 1},
    ):
        text = (doc.get("patched_text") or "").strip()
        channel_id = doc.get("channel_id")
        if not text or not channel_id:
            continue
        total += 1
        first_line = text.split("\n", 1)[0].strip()
        if not MIN_HEADING <= len(first_line) <= MAX_HEADING:
            continue
        key = (channel_id, shape(first_line))
        if len(key[1]) < MIN_HEADING:
            continue
        day = datetime.datetime.fromtimestamp(doc["fetch_time"]).strftime("%Y-%m-%d")
        days_seen[key].add(day)
        example.setdefault(key, text)

    recurring = [
        (len(seen), key) for key, seen in days_seen.items() if len(seen) >= min_days
    ]
    recurring.sort(reverse=True)

    print(f"Scanned {total} documents over {days} days.")
    print(
        f"{len(recurring)} first-line shapes recur on {min_days}+ separate days, "
        f"across {len({key[0] for _, key in recurring})} channels.\n"
    )

    uncaught = [(n, key) for n, key in recurring if not detector(example[key])]
    caught = [(n, key) for n, key in recurring if detector(example[key])]

    print(f"--- NOT caught by the current patterns ({len(uncaught)}):")
    print("    Candidates. Read each one and decide; recurrence is not a verdict.\n")
    for count, key in uncaught:
        first_line = " ".join(example[key].split())[:78]
        print(f"  [{count}d] {key[0][:24]:24s} | {first_line}")

    if include_caught:
        print(f"\n--- already caught ({len(caught)}):")
        for count, key in caught:
            first_line = " ".join(example[key].split())[:60]
            why = detector.explain(example[key]) or ""
            print(f"  [{count}d] {key[0][:20]:20s} | {first_line}\n{'':34s}→ {why[:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-config-path", type=str, default="configs/mongo_config.json")
    parser.add_argument(
        "--annotator-config-path", type=str, default="configs/annotator_config.json"
    )
    parser.add_argument("--days", type=int, default=7, help="how far back to look")
    parser.add_argument(
        "--min-days",
        type=int,
        default=4,
        help="on how many separate days a shape has to appear to count as a rubric",
    )
    parser.add_argument(
        "--include-caught",
        action="store_true",
        help="also list the recurring shapes the patterns already handle",
    )
    args = parser.parse_args()
    find_rubrics(**vars(args))


if __name__ == "__main__":
    main()
