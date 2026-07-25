from datetime import datetime, UTC
from unittest.mock import patch

from nyan.util import (
    format_date_uk,
    format_dt_uk,
    get_current_ts,
    get_timezone,
    normalize_channel_id,
    normalize_url,
    ts_to_dt,
)


def test_current_ts_does_not_depend_on_the_host_timezone() -> None:
    """The regression this guards against shifted every freshness calculation.

    `datetime.now().replace(tzinfo=utc)` relabels local wall-clock time as UTC,
    so on a host in Kyiv it returned a timestamp hours away from the real one.
    """
    with patch("nyan.util.datetime") as fake_datetime:
        moment = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
        fake_datetime.now.return_value = moment

        assert get_current_ts() == int(moment.timestamp())
        fake_datetime.now.assert_called_once_with(UTC)


def test_ts_to_dt_follows_daylight_saving_time() -> None:
    # Kyiv is UTC+2 in January and UTC+3 in July.
    winter = ts_to_dt(int(datetime(2026, 1, 15, 10, 0, tzinfo=UTC).timestamp()))
    summer = ts_to_dt(int(datetime(2026, 7, 15, 10, 0, tzinfo=UTC).timestamp()))

    assert winter.strftime("%H:%M") == "12:00"
    assert summer.strftime("%H:%M") == "13:00"


def test_ts_to_dt_accepts_a_zone_name_and_a_plain_offset() -> None:
    ts = int(datetime(2026, 7, 15, 10, 0, tzinfo=UTC).timestamp())

    assert ts_to_dt(ts, "UTC").strftime("%H:%M") == "10:00"
    assert ts_to_dt(ts, 5).strftime("%H:%M") == "15:00"


def test_unknown_timezone_falls_back_to_utc() -> None:
    assert get_timezone("Mars/Olympus_Mons") is UTC


def test_dates_are_formatted_in_ukrainian() -> None:
    dt = datetime(2026, 7, 25, 14, 30, tzinfo=UTC)

    assert format_dt_uk(dt) == "25 липня, 14:30"
    assert format_dt_uk(dt, with_time=False) == "25 липня"


def test_urls_are_normalized_for_comparison() -> None:
    assert normalize_url("https://T.me/Channel/123/?a=1#x") == "https://t.me/channel/123"
    assert normalize_url("") == ""


def test_channel_ids_are_normalized_from_every_form_they_arrive_in() -> None:
    assert normalize_channel_id("@Channel") == "channel"
    assert normalize_channel_id("https://t.me/s/Channel") == "channel"
    assert normalize_channel_id("https://t.me/Channel/123") == "channel"
    assert normalize_channel_id("") == ""


def test_a_prompt_date_carries_the_year() -> None:
    """The year is the point: without it a model writes the same deadline twice.

    "до кінця року" and "до кінця 2026 року" are one deadline, and a model with
    no idea which year it is treats them as two.
    """
    assert format_date_uk(datetime(2026, 7, 25, 18, 30)) == "25 липня 2026 року"


def test_a_prompt_date_uses_the_genitive_month() -> None:
    assert format_date_uk(datetime(2027, 1, 1)) == "1 січня 2027 року"
