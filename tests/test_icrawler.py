from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir': 'temp/google'})
google_crawler.crawl(keyword='ct with iv contrast + axial plane + face sinuses, and neck + paraganglioma', max_num=10)