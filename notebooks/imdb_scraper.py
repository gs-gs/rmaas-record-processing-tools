import requests
import pandas as pd
from bs4 import BeautifulSoup


class IMDB(object):
    
    def __init__(self): pass
    
    def read_links(self, file_path):
        return open(file_path, 'r').readlines()
        
    def scrape_movie(self, url):
        synopsis_url = url.strip() + 'synopsis'
        resp = requests.get(synopsis_url)
        soup = BeautifulSoup(resp.content, 'html.parser')
        synopsis_div = soup.find('div', {'id': 'swiki_body'})
        results = {'url': url}
        if synopsis_div:
            synopsis = synopsis_div.text.strip()
            results['synopsis'] = synopsis
            title_link = soup.find('div', {'id': 'tn15title'})
            if title_link:
                title = title_link.a.text
                results['title'] = title
        return results
        
    def scrape_movies(self, links):
        data = []
        for i, link in enumerate(links, 1):
            print('%s of %s - Scraping: %s' % (i, len(links), link.strip()))
            results = self.scrape_movie(link)
            data.append(results)
        return data
        
    def write_csv(self, data, outfile):
        df = pd.DataFrame(data)
        df.to_csv(outfile, index=False)
        print('-- COMPLETED --')
        print('Scraped data has been written into CSV here: %s' % outfile)


if __name__ == '__main__':
    scraper = IMDB()
    links = scraper.read_links('data/imdb_links.txt')
    data = scraper.scrape_movies(links)
    scraper.write_csv(data, 'data/imdb_data.csv')
    