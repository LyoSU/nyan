import os
import json
import logging
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace

from httpx import Timeout, Limits, HTTPTransport, Client, Response

from nyan.media import (
    MEDIA_ANIMATION,
    MEDIA_VIDEO,
    MediaItem,
    SentMedia,
    attach_urls,
    extract_sent_media,
)
from nyan.rich import Block, RenderedPost, fix_media_url, media_payloads
from nyan.util import Serializable


ISSUE_WARNING = "Missing issue '%s' in the client config"

# Values of MessageId.post_format.
FORMAT_RICH = "rich"
FORMAT_LEGACY = "legacy"


@dataclass
class IssueConfig:
    name: str
    # A numeric id, or "@username" for a public channel: the Bot API accepts
    # both, and a username is what a human reads off a t.me link.
    channel_id: int | str
    discussion_id: int
    bot_token: str
    last_update_id: int = 0


@dataclass
class MessageId(Serializable):
    message_id: int
    issue: str = "main"
    from_discussion: bool = False
    # How this message was sent, so an update can edit it the same way. Empty
    # for messages sent before this field existed; those are identified from
    # Telegram's own error the first time an update is attempted.
    post_format: str = ""
    # What Telegram stored for every attachment, in the order it was sent. Kept
    # for two readers: an edit, which references a file_id instead of a URL that
    # may already be gone, and the site, which serves the media through the bot
    # and so needs the file_id rather than the CDN link the crawler found.
    media: list[SentMedia] = field(default_factory=list)
    # Whether this message's text lives in a caption. Media was sent with it, or
    # Telegram said so — either way editMessageText cannot touch it. Recorded
    # because guessing from the current render was wrong in both directions: a
    # cluster that gained photos after publication was edited as a caption it
    # never had, and Telegram's refusal stopped the post updating for good.
    is_caption: bool = False

    @classmethod
    def fromdict(cls, d: dict[str, Any]) -> "MessageId":
        # `Serializable.fromdict` is shallow, so the attachments would come back
        # as plain dicts and break the first edit that touched them.
        keys = {f.name for f in fields(cls)}
        values = {k: v for k, v in d.items() if k in keys}
        values["media"] = [
            SentMedia.fromdict(m) if isinstance(m, dict) else m
            for m in values.get("media") or []
        ]
        return cls(**values)

    def file_ids(self) -> dict[str, str]:
        """Known URL -> file_id, for rewriting an edit's attachments."""
        return {item.url: item.file_id for item in self.media if item.url}

    @property
    def has_caption(self) -> bool:
        """Whether editMessageText would be refused on this message.

        Recorded attachments are proof on their own: anything sent with media
        carries a caption. `is_caption` covers the messages that have none
        recorded — those sent before file ids were kept, identified from
        Telegram's own refusal.
        """
        return self.is_caption or bool(self.media)

    def as_tuple(self) -> tuple[str, int]:
        return (self.issue, self.message_id)

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __eq__(self, another: Any) -> bool:
        # NotImplemented, not an exception: Python needs this to fall back to
        # the other operand, and `message == None` must be False rather than a
        # crash.
        if not isinstance(another, MessageId):
            return NotImplemented
        return self.as_tuple() == another.as_tuple()


class TelegramClient:
    def __init__(self, config_path: str) -> None:
        assert os.path.exists(config_path)
        with open(config_path) as r:
            self.config = json.load(r)

        self.host = self.config.get("host", "https://api.telegram.org")
        timeout = Timeout(
            connect=self.config.get("connect_timeout", 30.0),
            read=self.config.get("read_timeout", 30.0),
            write=self.config.get("write_timeout", 30.0),
            pool=self.config.get("pool_timeout", 1.0),
        )
        limits = Limits(
            max_connections=self.config.get("connection_pool_size", 1),
            max_keepalive_connections=self.config.get("connection_pool_size", 1),
        )
        transport = HTTPTransport(retries=self.config.get("retries", 5))
        self.client = Client(timeout=timeout, limits=limits, transport=transport)

        self.issues: dict[str, IssueConfig] = {
            config["name"]: IssueConfig(**config) for config in self.config["issues"]
        }
        # Channel post id -> its mirrored message id in the discussion group.
        self.discussions: dict[str, dict[int, int]] = {
            issue.name: dict() for _, issue in self.issues.items()
        }
        for issue_name in self.issues:
            self.update_discussion_mapping(issue_name)

    def has_issue(self, issue_name: str) -> bool:
        """Whether this issue has a channel to post to.

        Asked by the daemon before it renders anything: without a channel every
        send is refused, and finding that out afterwards means the post has
        already been written.
        """
        return issue_name in self.issues

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TelegramClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def send_post(
        self,
        post: RenderedPost,
        issue_name: str,
        reply_to: int | None = None,
    ) -> MessageId | None:
        """Send a rendered post using whichever API method its format needs."""
        if post.is_rich:
            assert post.blocks is not None
            message = self.send_rich_message(post.blocks, issue_name, reply_to=reply_to)
            if message:
                message.post_format = FORMAT_RICH
            return message
        assert post.text is not None
        message = self.send_message(
            post.text,
            issue_name,
            media=post.media,
            reply_to=reply_to,
        )
        if message:
            message.post_format = FORMAT_LEGACY
        return message

    def update_post(self, message: MessageId, post: RenderedPost) -> None:
        assert not message.from_discussion
        issue = self.issues[message.issue]
        if post.is_rich:
            assert post.blocks is not None
            # Rewritten before sending, so the edit references files Telegram
            # already holds. A URL from an earlier crawl may no longer resolve,
            # and then the edit takes the whole post's media down with it.
            urls = self._use_known_files(post.blocks, message)
            response = self._edit_rich(message.message_id, post.blocks, issue=issue)
            if response.status_code == 200:
                # Attachments an update added arrive with a file_id of their own,
                # and the next edit has to be able to reference those too.
                self._record_media(message, response, urls)
                return
        elif message.has_caption:
            assert post.text is not None
            response = self._edit_caption(message.message_id, post.text, issue=issue)
        else:
            assert post.text is not None
            response = self._edit_text(message.message_id, post.text, issue=issue)

        if response.status_code == 200:
            return

        if self._is_caption_only(response):
            # A message sent as media carries a caption, not text, and cannot be
            # edited with editMessageText. Record what it is so the next
            # iteration renders and edits it as a caption instead. Messages sent
            # before `is_caption` was recorded are the only ones that get here.
            message.post_format = FORMAT_LEGACY
            message.is_caption = True
            logging.warning(
                "Message %d predates the rich format; will update it as a "
                "caption from now on",
                message.message_id,
            )
            return

        logging.error(
            "Update of %s failed (%d): %s",
            message.message_id,
            response.status_code,
            response.text,
        )

    @staticmethod
    def _use_known_files(blocks: Sequence[Block], message: MessageId) -> list[str]:
        """Swap every attachment we have a file_id for, in place.

        Returns the URLs in send order — including the ones left as URLs, which
        is how the response's file_ids get paired back to them.
        """
        known = message.file_ids()
        urls: list[str] = []
        for payload in media_payloads(blocks):
            url = str(payload["media"])
            urls.append(url)
            file_id = known.get(url)
            if file_id:
                payload["media"] = file_id
        return urls

    @staticmethod
    def _record_media(
        message: MessageId, response: Response, urls: Sequence[str]
    ) -> None:
        """Keep what Telegram stored for this message's attachments.

        Read from every send and every successful edit, because the set changes:
        a cluster grows, a photo joins the post, and that one is only referable
        by file_id once Telegram has answered for it.
        """
        result = response.json().get("result")
        if not result:
            return
        media = attach_urls(extract_sent_media(result), list(urls))
        if media:
            message.media = media

    @staticmethod
    def _is_caption_only(response: Response) -> bool:
        if response.status_code != 400:
            return False
        description = response.json().get("description", "")
        return "no text in the message to edit" in description

    def clone_issue(
        self, name: str, channel_id: int | str, like: str = "main"
    ) -> bool:
        """Register `name` as another channel posted to by the same bot.

        For a second channel that needs nothing of its own but an id — the
        digest — this saves editing the client config, so the channel can be
        set from the environment like every other deployment detail. An issue
        already present in the config always wins, since that is where a
        separate bot token would have to live.
        """
        if name in self.issues:
            return True
        if like not in self.issues:
            logging.warning(ISSUE_WARNING, like)
            return False
        template = self.issues[like]
        self.issues[name] = replace(
            template, name=name, channel_id=channel_id, discussion_id=0
        )
        logging.info("Issue '%s' posts to %s, cloned from '%s'", name, channel_id, like)
        return True

    def send_rich_message(
        self,
        blocks: Sequence[Block],
        issue_name: str,
        reply_to: int | None = None,
    ) -> MessageId | None:
        if issue_name not in self.issues:
            logging.warning(ISSUE_WARNING, issue_name)
            return None
        issue = self.issues[issue_name]
        response = self._send_rich(blocks, issue=issue, reply_to=reply_to)

        if response.status_code != 200:
            logging.error(
                "sendRichMessage failed (%d): %s",
                response.status_code,
                response.text,
            )
            return None

        message_id = self._extract_message_id(response)
        if message_id is None:
            return None
        message = MessageId(
            message_id=message_id, issue=issue_name, from_discussion=False
        )
        self._record_media(message, response, self._use_known_files(blocks, message))
        return message

    def send_message(
        self,
        text: str,
        issue_name: str,
        media: Sequence[MediaItem] = tuple(),
        reply_to: int | None = None,
        parse_mode: str = "html",
    ) -> MessageId | None:
        if issue_name not in self.issues:
            logging.warning(ISSUE_WARNING, issue_name)
            return None
        issue = self.issues[issue_name]
        response = None
        # One attachment goes out as itself, several as a group. A group is
        # allowed to mix photos and videos, which is what lets the choice of
        # attachments stay in one place instead of each format ranking the types
        # its own way.
        if len(media) > 1:
            response = self._send_media_group(
                text, media, issue=issue, reply_to=reply_to, parse_mode=parse_mode
            )
        elif len(media) == 1 and media[0].type == MEDIA_VIDEO:
            response = self._send_video(
                text, media[0].url, issue=issue, reply_to=reply_to, parse_mode=parse_mode
            )
        elif len(media) == 1 and media[0].type == MEDIA_ANIMATION:
            response = self._send_animation(
                text, media[0].url, issue=issue, reply_to=reply_to, parse_mode=parse_mode
            )
        elif len(media) == 1:
            response = self._send_photo(
                text, media[0].url, issue=issue, reply_to=reply_to, parse_mode=parse_mode
            )
        else:
            response = self._send_text(
                text, issue=issue, reply_to=reply_to, parse_mode=parse_mode
            )

        sent_with_media = bool(media)
        if response.status_code == 400 and "description" in response.text:
            response_dict = response.json()
            description = response_dict.get("description", "")
            if description == "Bad Request: message caption is too long":
                logging.warning("Caption too long, resending as text only")
                response = self._send_text(text, issue=issue, reply_to=reply_to)
                # The attachments went nowhere, so the text is text: an update
                # has to edit it as one.
                sent_with_media = False

        if response.status_code != 200:
            logging.error(
                "sendMessage failed (%d): %s", response.status_code, response.text
            )
            return None

        message_id = self._extract_message_id(response)
        if message_id is None:
            return None
        message = MessageId(
            message_id=message_id,
            issue=issue_name,
            from_discussion=False,
            is_caption=sent_with_media,
        )
        self._record_media(message, response, [item.url for item in media])
        return message

    @staticmethod
    def _extract_message_id(response: Response) -> int | None:
        result = response.json().get("result")
        if not result:
            return None
        # sendMediaGroup answers with a list of messages, one per attachment.
        if isinstance(result, list):
            result = result[0]
        message_id = int(result.get("message_id", 0))
        if message_id == 0:
            return None
        return message_id

    def update_discussion_mapping(self, issue_name: str) -> None:
        if issue_name not in self.issues:
            logging.warning(ISSUE_WARNING, issue_name)
            return None
        issue = self.issues[issue_name]
        updates = self._get_updates(issue)
        if not updates:
            return
        for update in updates:
            if "message" not in update:
                continue
            message = update["message"]
            if "forward_from_chat" not in message:
                continue
            if issue.channel_id != message["forward_from_chat"]["id"]:
                continue
            if issue.discussion_id != message["chat"]["id"]:
                continue
            orig_message_id = message["forward_from_message_id"]
            discussion_message_id = message["message_id"]
            self.discussions[issue.name][orig_message_id] = discussion_message_id

    def get_discussion(self, message: MessageId) -> MessageId:
        # 0 means "this post has no discussion mirror yet", which every
        # consumer already treats as absent via a falsiness check.
        discussion_message_id = self.discussions[message.issue].get(
            message.message_id, 0
        )
        return MessageId(
            message_id=discussion_message_id, issue=message.issue, from_discussion=True
        )

    def send_discussion_message(
        self,
        text: str,
        discussion_message: MessageId,
        disable_web_page_preview: bool = False,
    ) -> Response | None:
        assert discussion_message.from_discussion
        issue = self.issues[discussion_message.issue]
        if not issue.discussion_id or not discussion_message.message_id:
            return None
        url_template = self.host + "/bot{}/sendMessage"
        params = {
            "chat_id": issue.discussion_id,
            "text": text,
            "parse_mode": "html",
            "disable_web_page_preview": disable_web_page_preview,
            "reply_to_message_id": discussion_message.message_id,
        }
        return self._post(url_template.format(issue.bot_token), params)

    def _send_rich(
        self,
        blocks: Sequence[Block],
        issue: IssueConfig,
        reply_to: int | None = None,
    ) -> Response:
        url_template = self.host + "/bot{}/sendRichMessage"
        params: dict[str, Any] = {
            "chat_id": issue.channel_id,
            "rich_message": json.dumps({"blocks": list(blocks)}, ensure_ascii=False),
            "disable_notification": True,
        }
        if reply_to:
            # sendRichMessage takes a ReplyParameters object rather than the
            # flat reply_to_message_id the older send methods accept.
            params["reply_parameters"] = json.dumps(
                {"message_id": reply_to, "allow_sending_without_reply": True}
            )
        return self._post(url_template.format(issue.bot_token), params)

    def _edit_rich(
        self, message_id: int, blocks: Sequence[Block], issue: IssueConfig
    ) -> Response:
        url_template = self.host + "/bot{}/editMessageText"
        params = {
            "chat_id": issue.channel_id,
            "message_id": message_id,
            "rich_message": json.dumps({"blocks": list(blocks)}, ensure_ascii=False),
        }
        return self._post(url_template.format(issue.bot_token), params)

    def _send_text(
        self,
        text: str,
        issue: IssueConfig,
        reply_to: int | None = None,
        parse_mode: str = "html",
    ) -> Response:
        url_template = self.host + "/bot{}/sendMessage"
        params = {
            "chat_id": issue.channel_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
            "disable_notification": True,
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        return self._post(url_template.format(issue.bot_token), params)

    def _send_photo(
        self,
        text: str,
        photo: str,
        issue: IssueConfig,
        reply_to: int | None = None,
        parse_mode: str = "html",
    ) -> Response:
        url_template = self.host + "/bot{}/sendPhoto"
        params = {
            "chat_id": issue.channel_id,
            "caption": text,
            "photo": fix_media_url(photo),
            "parse_mode": parse_mode,
            "disable_notification": True,
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        return self._post(url_template.format(issue.bot_token), params)

    def _send_animation(
        self,
        text: str,
        animation: str,
        issue: IssueConfig,
        reply_to: int | None = None,
        parse_mode: str = "html",
    ) -> Response:
        url_template = self.host + "/bot{}/sendAnimation"
        params = {
            "chat_id": issue.channel_id,
            "caption": text,
            "animation": animation,
            "parse_mode": parse_mode,
            "disable_notification": True,
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        return self._post(url_template.format(issue.bot_token), params)

    def _send_video(
        self,
        text: str,
        video: str,
        issue: IssueConfig,
        reply_to: int | None = None,
        parse_mode: str = "html",
    ) -> Response:
        url_template = self.host + "/bot{}/sendVideo"
        params = {
            "chat_id": issue.channel_id,
            "caption": text,
            "video": fix_media_url(video),
            "parse_mode": parse_mode,
            "disable_notification": True,
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        return self._post(url_template.format(issue.bot_token), params)

    def _send_media_group(
        self,
        text: str,
        media: Sequence[MediaItem],
        issue: IssueConfig,
        reply_to: int | None = None,
        parse_mode: str = "html",
    ) -> Response:
        url_template = self.host + "/bot{}/sendMediaGroup"
        # In the order chosen, types mixed as they come: a group where a video
        # sits between two photos is exactly what the cluster decided to show.
        group = [
            {
                "type": item.type,
                "media": fix_media_url(item.url),
                # The caption belongs to the first attachment only; repeated, it
                # would be shown once per item.
                "caption": text if i == 0 else "",
                "parse_mode": parse_mode,
            }
            for i, item in enumerate(media)
        ]
        params = {
            "chat_id": issue.channel_id,
            "disable_notification": True,
            "media": json.dumps(group),
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        return self._post(url_template.format(issue.bot_token), params)

    def _edit_text(
        self, message_id: int, text: str, issue: IssueConfig, parse_mode: str = "html"
    ) -> Response:
        url_template = self.host + "/bot{}/editMessageText"
        params = {
            "chat_id": issue.channel_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
            "message_id": message_id,
        }
        return self._post(url_template.format(issue.bot_token), params)

    def _edit_caption(
        self, message_id: int, text: str, issue: IssueConfig, parse_mode: str = "html"
    ) -> Response:
        url_template = self.host + "/bot{}/editMessageCaption"
        params = {
            "chat_id": issue.channel_id,
            "message_id": message_id,
            "caption": text,
            "parse_mode": parse_mode,
        }
        return self._post(url_template.format(issue.bot_token), params)

    def _get_updates(self, issue: IssueConfig) -> list[dict[str, Any]]:
        url_template = self.host + "/bot{}/getUpdates"
        params = {"timeout": 10}
        if issue.last_update_id != 0:
            params["offset"] = issue.last_update_id
        response = self.client.get(
            url_template.format(issue.bot_token), params=params, timeout=20
        )
        if response.status_code != 200:
            return []
        updates: list[dict[str, Any]] = response.json()["result"]
        # The offset must advance past the highest update seen, computed once:
        # adding 1 per iteration overshoots whenever updates arrive out of
        # order, which silently drops the updates in between.
        for update in updates:
            issue.last_update_id = max(issue.last_update_id, update["update_id"] + 1)
        return updates

    def _post(self, url: str, params: dict[str, Any]) -> Response:
        return self.client.post(url, data=params)
