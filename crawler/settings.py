BOT_NAME = "telegram_crawler"
SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"
USER_AGENT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
ROBOTSTXT_OBEY = False
CONCURRENT_REQUESTS = 32
DOWNLOAD_DELAY = 0.3
RANDOMIZE_DOWNLOAD_DELAY = True
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
