BOT_NAME = "telegram_crawler"
SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"
USER_AGENT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
# Pinned, because the wording of the page is data we parse: "edited" in the meta
# line and "subscribers" in the header counter are both matched as English text.
# Without this header the language is Telegram's choice, and the day it chooses
# another one, `edited` is always False and every subscriber count is 0 — with no
# error anywhere.
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,uk;q=0.8",
}
ROBOTSTXT_OBEY = False
CONCURRENT_REQUESTS = 32
# Every request in this crawl goes to t.me, and Scrapy caps per domain before it
# caps globally: without this the 32 above described a limit nothing could reach,
# because the per-domain default is 8.
CONCURRENT_REQUESTS_PER_DOMAIN = 8
DOWNLOAD_DELAY = 0.3
RANDOMIZE_DOWNLOAD_DELAY = True
# The default is 180s. A pass has 342 channels to get through before the recrawl
# interval comes round again, so one stalled connection holding a slot for three
# minutes costs the freshness of everything behind it.
DOWNLOAD_TIMEOUT = 30
# 429 is in Scrapy's default list, and it is the one that matters here: the whole
# crawl is one host, and being throttled by it is the expected failure, not an
# exotic one. Spelled out rather than inherited so a Scrapy default that changes
# does not quietly change how we behave under throttling.
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [429, 500, 502, 503, 504, 408, 522, 524]
TELNETCONSOLE_ENABLED = False
ITEM_PIPELINES = {
    # Measurements first, so they are stored before the post pipeline waves
    # them through; every pipeline lets the kinds it does not handle pass
    # untouched.
    "crawler.pipelines.ChannelStatsPipeline": 200,
    # Before MongoPipeline for a reason: it reads `views` off the item, which is
    # the value about to be written over the previous one. Order does not matter
    # for correctness — they write to different collections — but keeping the
    # sample ahead of the overwrite is how the two stay readable together.
    "crawler.pipelines.PostHistoryPipeline": 250,
    "crawler.pipelines.MongoPipeline": 300,
}
# Renamed from DNS_RESOLVER in Scrapy 2.13.
TWISTED_DNS_RESOLVER = "scrapy.resolver.CachingHostnameResolver"
LOG_LEVEL = "INFO"
