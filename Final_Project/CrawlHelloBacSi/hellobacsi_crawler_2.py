#!/usr/bin/env python3
"""
HelloBacsi Complete Crawler v4 - FIXED URL STRUCTURE
Sửa lỗi: Bài viết trong subcategory có thể thuộc category khác
Ví dụ: /vacxin/vacxin-rubella/ chứa bài từ /thuoc/, /nuoi-day-con/, etc.
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
import re

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hellobacsi_crawler_2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config
BASE_URL = "https://hellobacsi.com"
OUTPUT_DIR = Path("hellobacsi_data_2")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
}

session = requests.Session()
session.headers.update(HEADERS)

# 31 Chuyên mục chính - dùng để thu thập subcategories
MAIN_CATEGORIES = {
    'benh-tim-mach': 'Bệnh tim mạch',
    'ho-va-benh-duong-ho-hap': 'Bệnh hô hấp',
    'ung-thu-ung-buou': 'Ung thư - Ung bướu',
    'benh-tieu-hoa': 'Bệnh tiêu hóa',
}


def setup_driver():
    """Khởi tạo Selenium WebDriver"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(45)
    return driver


def count_url_segments(url):
    """Đếm số segments trong URL path"""
    path = urlparse(url).path.strip('/')
    segments = [s for s in path.split('/') if s]
    return len(segments)


def is_article_url(url):
    """
    Kiểm tra URL có phải là bài viết thực sự không.
    Bài viết thực sự có >= 2 segments VÀ không phải là subcategory page
    """
    path = urlparse(url).path.strip('/')
    segments = [s for s in path.split('/') if s]
    
    # Loại bỏ các pattern không phải bài viết
    skip_patterns = ['categories', 'tag', 'author', 'expert', 'static', 
                     'care', 'shop', 'community', 'health-tools']
    
    if len(segments) < 2:
        return False
    
    for pattern in skip_patterns:
        if pattern in segments:
            return False
    
    return True


def extract_all_article_urls_from_page(soup, source_category):
    """
    Trích xuất TẤT CẢ URLs bài viết từ một trang (không giới hạn category)
    Trả về dict {url: detected_category}
    """
    article_urls = {}
    
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        
        # Chuẩn hóa URL
        if not href.startswith('http'):
            href = urljoin(BASE_URL, href)
        
        # Chỉ lấy hellobacsi.com
        if 'hellobacsi.com' not in href:
            continue
        
        # Loại bỏ query params và fragments
        clean_url = href.split('?')[0].split('#')[0]
        if not clean_url.endswith('/'):
            clean_url += '/'
        
        # Kiểm tra có phải URL bài viết không
        if not is_article_url(clean_url):
            continue
        
        # Lấy category từ URL
        path = urlparse(clean_url).path.strip('/')
        segments = path.split('/')
        detected_category = segments[0] if segments else source_category
        
        # Lấy tên category từ dict hoặc dùng slug
        cat_name = MAIN_CATEGORIES.get(detected_category, detected_category)
        
        article_urls[clean_url] = cat_name
    
    return article_urls


def crawl_page_with_selenium(driver, url):
    """Crawl một trang với Selenium và trả về soup"""
    try:
        driver.get(url)
        time.sleep(2)
        
        # Scroll để load nội dung
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        
        html = driver.page_source
        return BeautifulSoup(html, 'lxml')
    except Exception as e:
        logger.warning(f"Lỗi khi load {url}: {e}")
        return None


def get_subcategory_urls(driver, category_slug):
    """Lấy tất cả subcategory URLs (2 segments) từ một category"""
    subcategory_urls = set()
    category_url = f"{BASE_URL}/{category_slug}/"
    
    soup = crawl_page_with_selenium(driver, category_url)
    if not soup:
        return subcategory_urls
    
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        
        if not href.startswith('http'):
            href = urljoin(BASE_URL, href)
        
        if 'hellobacsi.com' not in href:
            continue
        
        if f'/{category_slug}/' not in href:
            continue
        
        # Loại bỏ các URL không cần
        skip_patterns = ['?page=', 'community', 'care', 'shop', 'static', 'bot', '#', '/tag/', '/author/']
        if any(s in href for s in skip_patterns):
            continue
        
        # Lấy URLs có đúng 2 segments
        segments = count_url_segments(href)
        if segments == 2:
            clean_url = href.split('?')[0].split('#')[0]
            if not clean_url.endswith('/'):
                clean_url += '/'
            subcategory_urls.add(clean_url)
    
    return subcategory_urls


def crawl_subcategory_for_all_articles(driver, subcategory_url, source_category):
    """
    Crawl một subcategory và lấy TẤT CẢ bài viết (từ mọi category)
    """
    all_articles = {}
    
    # Crawl nhiều trang pagination
    for page_num in range(1, 20):
        if page_num == 1:
            page_url = subcategory_url
        else:
            page_url = f"{subcategory_url}?page={page_num}"
        
        soup = crawl_page_with_selenium(driver, page_url)
        if not soup:
            break
        
        # Extract tất cả URLs bài viết
        articles = extract_all_article_urls_from_page(soup, source_category)
        
        # Kiểm tra có bài mới không
        new_count = sum(1 for url in articles if url not in all_articles)
        
        if new_count == 0 and page_num > 1:
            break
        
        all_articles.update(articles)
    
    return all_articles


def crawl_category_complete(driver, category_slug, category_name):
    """Crawl hoàn chỉnh một category"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📁 CRAWLING: {category_name} (/{category_slug}/)")
    logger.info(f"{'='*60}")
    
    all_article_urls = {}  # {url: category_name}
    
    # Bước 1: Lấy subcategories
    logger.info(f"  📂 Bước 1: Tìm subcategories...")
    subcategories = get_subcategory_urls(driver, category_slug)
    logger.info(f"  ✓ Tìm thấy {len(subcategories)} subcategories")
    
    # Bước 2: Crawl trang chính category (pagination)
    logger.info(f"  📄 Bước 2: Lấy bài viết từ trang chính...")
    for page_num in range(1, 15):
        if page_num == 1:
            page_url = f"{BASE_URL}/{category_slug}/"
        else:
            page_url = f"{BASE_URL}/{category_slug}/?page={page_num}"
        
        soup = crawl_page_with_selenium(driver, page_url)
        if not soup:
            break
        
        articles = extract_all_article_urls_from_page(soup, category_slug)
        new_count = sum(1 for url in articles if url not in all_article_urls)
        
        if new_count == 0 and page_num > 1:
            break
        
        all_article_urls.update(articles)
    
    logger.info(f"  ✓ Từ trang chính: {len(all_article_urls)} bài viết")
    
    # Bước 3: Crawl từng subcategory
    logger.info(f"  📂 Bước 3: Crawl {len(subcategories)} subcategories...")
    
    for i, subcat_url in enumerate(subcategories, 1):
        subcat_name = subcat_url.rstrip('/').split('/')[-1]
        
        try:
            subcat_articles = crawl_subcategory_for_all_articles(driver, subcat_url, category_slug)
            
            new_count = sum(1 for url in subcat_articles if url not in all_article_urls)
            all_article_urls.update(subcat_articles)
            
            logger.info(f"    [{i}/{len(subcategories)}] {subcat_name}: +{new_count} mới (tổng: {len(all_article_urls)})")
            
        except Exception as e:
            logger.warning(f"    [{i}/{len(subcategories)}] {subcat_name}: Lỗi - {e}")
    
    logger.info(f"\n  ✅ TỔNG {category_name}: {len(all_article_urls)} URLs bài viết")
    return all_article_urls


def parse_article_content(url, category):
    """Parse nội dung đầy đủ từ một bài viết"""
    try:
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Lấy tiêu đề
        title = ""
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
        else:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True).split(' - ')[0]
        
        if not title or len(title) < 5:
            return None
        
        # Lấy tác giả
        author = ""
        author_patterns = [
            {'class_': re.compile(r'author|writer|reviewer', re.I)},
            {'class_': 'article-author'},
        ]
        for pattern in author_patterns:
            author_tag = soup.find(attrs=pattern)
            if author_tag:
                author = author_tag.get_text(strip=True)
                break
        
        # Lấy ngày đăng
        date = ""
        time_tag = soup.find('time')
        if time_tag:
            date = time_tag.get('datetime', '') or time_tag.get_text(strip=True)
        
        # Lấy nội dung chính
        content_parts = []
        
        article = soup.find('article')
        if not article:
            article = soup.find(class_=re.compile(r'article-content|post-content|entry-content', re.I))
        if not article:
            article = soup.find('main')
        
        if article:
            # Xóa các phần không cần
            for unwanted in article.find_all(['script', 'style', 'nav', 'footer', 
                                              'aside', 'form', 'button', 'iframe']):
                unwanted.decompose()
            
            for unwanted in article.find_all(class_=re.compile(r'related|sidebar|advertisement|social|share|comment', re.I)):
                unwanted.decompose()
            
            # Lấy headings và paragraphs
            for elem in article.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'blockquote']):
                text = elem.get_text(strip=True)
                if text and len(text) > 5:
                    if elem.name in ['h1', 'h2']:
                        content_parts.append(f"\n## {text}\n")
                    elif elem.name in ['h3', 'h4']:
                        content_parts.append(f"\n### {text}\n")
                    else:
                        content_parts.append(text)
        
        content = '\n\n'.join(content_parts)
        
        # Yêu cầu content >= 300 ký tự
        if len(content) < 300:
            return None
        
        return {
            'url': url,
            'title': title,
            'category': category,
            'author': author,
            'date': date,
            'content': content
        }
        
    except Exception as e:
        return None


def save_checkpoint(articles, filename):
    """Lưu checkpoint"""
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved: {len(articles)} articles -> {filepath}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         HELLOBACSI DEEP CRAWLER v4 - FIXED                    ║
║         Lấy bài viết từ mọi category trong subcategory        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Load cache
    cache_file = OUTPUT_DIR / "articles_2.json"
    crawled_articles = []
    crawled_urls = set()
    
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            crawled_articles = json.load(f)
            crawled_urls = {a['url'] for a in crawled_articles}
    
    logger.info(f"Đã có {len(crawled_articles)} bài viết trong cache")
    
    # ========== PHASE 1: Thu thập URL ==========
    print("\n" + "="*60)
    print("📡 PHASE 1: KHÁM PHÁ BÀI VIẾT")
    print("="*60)
    
    logger.info("Khởi tạo Selenium...")
    driver = setup_driver()
    
    all_article_urls = {}  # {url: category_name}
    
    try:
        for cat_slug, cat_name in MAIN_CATEGORIES.items():
            urls = crawl_category_complete(driver, cat_slug, cat_name)
            
            # Thêm vào tổng
            for url, cat in urls.items():
                if url not in all_article_urls:
                    all_article_urls[url] = cat
            
            # Save progress
            with open(OUTPUT_DIR / "discovered_urls_2.json", 'w', encoding='utf-8') as f:
                json.dump(all_article_urls, f, ensure_ascii=False, indent=2)
    
    finally:
        driver.quit()
    
    # Loại bỏ đã crawl
    new_urls = {url: cat for url, cat in all_article_urls.items() if url not in crawled_urls}
    
    logger.info(f"\n📊 Phase 1 Hoàn tất:")
    logger.info(f"   - Tổng URLs: {len(all_article_urls)}")
    logger.info(f"   - Đã crawl: {len(crawled_urls)}")
    logger.info(f"   - URLs mới: {len(new_urls)}")
    
    # ========== PHASE 2: Crawl nội dung ==========
    print("\n" + "="*60)
    print("📥 PHASE 2: CRAWL NỘI DUNG")
    print("="*60)
    
    urls_list = list(new_urls.items())
    total = len(urls_list)
    success = 0
    failed = 0
    
    for i, (url, category) in enumerate(urls_list, 1):
        try:
            article = parse_article_content(url, category)
            
            if article:
                crawled_articles.append(article)
                crawled_urls.add(url)
                success += 1
                
                if success % 20 == 0:
                    logger.info(f"[{i}/{total}] ✓ Đã crawl {success} bài viết")
            else:
                failed += 1
            
            # Checkpoint
            if success % 50 == 0:
                save_checkpoint(crawled_articles, "articles_2.json")
            
            time.sleep(0.3)
            
        except Exception as e:
            failed += 1
    
    # Save final
    save_checkpoint(crawled_articles, "articles_2.json")
    
    # ========== PHASE 3: Training data ==========
    print("\n" + "="*60)
    print("📝 PHASE 3: TẠO TRAINING DATA")
    print("="*60)
    
    training_data = []
    for article in crawled_articles:
        text = f"Tiêu đề: {article['title']}\n\n"
        text += f"Chuyên mục: {article['category']}\n\n"
        text += article['content']
        
        if article.get('author'):
            text += f"\n\n{article['author']}"
        
        training_data.append({
            "text": text,
            "metadata": {
                "url": article['url'],
                "category": article['category'],
                "title": article['title'],
                "author": article.get('author', ''),
                "date": article.get('date', '')
            }
        })
    
    # Save
    with open(OUTPUT_DIR / "training_data_v4.json", 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / "training_data_v4.jsonl", 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # ========== Báo cáo ==========
    print("\n" + "="*60)
    print("📊 BÁO CÁO TỔNG KẾT")
    print("="*60)
    
    print(f"""
    ✅ Tổng bài viết: {len(crawled_articles)}
    ✅ Thành công lần này: {success}
    ❌ Thất bại: {failed}
    
    📁 Files:
       - {OUTPUT_DIR}/articles_2.json
       - {OUTPUT_DIR}/training_data_2.json
       - {OUTPUT_DIR}/training_data_2.jsonl
    """)


if __name__ == "__main__":
    main()
