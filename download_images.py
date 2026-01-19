from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "dataset/images"})
crawler.crawl(keyword="eyeglasses product", max_num=500)
