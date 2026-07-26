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
    # them through; both let the other's kind of item pass untouched.
    "crawler.pipelines.ChannelStatsPipeline": 200,
    "crawler.pipelines.MongoPipeline": 300,
}
# Renamed from DNS_RESOLVER in Scrapy 2.13.
TWISTED_DNS_RESOLVER = "scrapy.resolver.CachingHostnameResolver"
LOG_LEVEL = "INFO"
