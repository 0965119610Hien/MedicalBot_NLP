import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from urllib.parse import urljoin, quote
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vinmec_complete.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.vinmec.com"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
}

OUTPUT_DIR = Path("vinmec_complete_data")
OUTPUT_DIR.mkdir(exist_ok=True)

REQUEST_DELAY = 0.5
session = requests.Session()


def safe_request(url, max_retries=3):
    """Make HTTP request with retry logic"""
    for attempt in range(max_retries):
        try:
            time.sleep(REQUEST_DELAY)
            response = session.get(url, headers=HEADERS, timeout=30)
            response.encoding = 'utf-8'
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def get_soup(url):
    """Get BeautifulSoup object from URL"""
    response = safe_request(url)
    if response:
        return BeautifulSoup(response.text, 'html.parser')
    return None


def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def save_json(data, filename):
    """Save data to JSON file"""
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(data)} items to {filepath}")


def load_json(filename):
    """Load data from JSON file"""
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


# ============ DISEASE CRAWLER ============

def crawl_disease_list():
    """Get all disease URLs"""
    logger.info("Discovering disease URLs...")
    disease_urls = set()
    
    # Crawl by alphabet
    letters = 'abcdefghijklmnopqrstuvwxyz'
    
    for letter in letters:
        base_url = f"{BASE_URL}/vie/tra-cuu-benh/{letter}"
        
        page = 1
        while page <= 50:
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}?page={page}"
            
            soup = get_soup(url)
            if not soup:
                break
            
            found_new = False
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/vie/benh/' in href:
                    full_url = urljoin(BASE_URL, href)
                    if re.search(r'-\d+$', full_url):
                        if full_url not in disease_urls:
                            found_new = True
                            disease_urls.add(full_url)
            
            if not found_new:
                break
            
            page += 1
        
        logger.info(f"Letter '{letter}': {len(disease_urls)} total diseases")
    
    # Also try main disease page
    soup = get_soup(f"{BASE_URL}/vie/benh/")
    if soup:
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/vie/benh/' in href and re.search(r'-\d+$', href):
                disease_urls.add(urljoin(BASE_URL, href))
    
    logger.info(f"Total disease URLs: {len(disease_urls)}")
    return list(disease_urls)


def parse_disease_detail(url):
    """Parse disease detail page"""
    soup = get_soup(url)
    if not soup:
        return None
    
    disease = {
        'url': url,
        'crawl_time': datetime.now().isoformat()
    }
    
    # Get disease name
    h1 = soup.find('h1')
    if h1:
        name = clean_text(h1.get_text())
        # Remove common suffixes
        name = re.sub(r':\s*Nguyên nhân.*$', '', name)
        disease['ten_benh'] = name
    
    # Parse content sections
    main_content = soup.find('article') or soup.find('div', {'class': re.compile(r'content|article')})
    
    if main_content:
        sections = {}
        current_section = 'intro'
        sections[current_section] = []
        
        for elem in main_content.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li']):
            if elem.name in ['h2', 'h3', 'h4']:
                section_name = clean_text(elem.get_text())
                if section_name:
                    current_section = section_name
                    sections[current_section] = []
            else:
                text = clean_text(elem.get_text())
                if text and len(text) > 10:
                    sections[current_section].append(text)
        
        # Map to fields
        field_mapping = {
            'tong_quan': ['tổng quan', 'giới thiệu', 'khái niệm', 'intro'],
            'nguyen_nhan': ['nguyên nhân', 'căn nguyên'],
            'yeu_to_nguy_co': ['yếu tố nguy cơ', 'nguy cơ'],
            'trieu_chung': ['triệu chứng', 'biểu hiện', 'dấu hiệu'],
            'chan_doan': ['chẩn đoán', 'xét nghiệm'],
            'dieu_tri': ['điều trị', 'chữa trị', 'phương pháp điều trị'],
            'phong_ngua': ['phòng ngừa', 'dự phòng', 'phòng tránh'],
            'bien_chung': ['biến chứng'],
            'tien_luong': ['tiên lượng'],
        }
        
        for field, keywords in field_mapping.items():
            for section_name, content in sections.items():
                if any(kw in section_name.lower() for kw in keywords):
                    if content:
                        disease[field] = '\n'.join(content)
                    break
        
        # Full content
        all_text = []
        for section_name, content in sections.items():
            if section_name != 'intro':
                all_text.append(f"\n## {section_name}\n")
            all_text.extend(content)
        disease['noi_dung_day_du'] = '\n'.join(all_text)
    
    # Get related topics
    tags = []
    for link in soup.find_all('a', href=lambda h: h and '/vie/chu-de/' in h):
        tag = clean_text(link.get_text())
        if tag:
            tags.append(tag)
    disease['chu_de'] = list(set(tags))
    
    return disease


def crawl_all_diseases():
    """Crawl all diseases"""
    logger.info("=" * 50)
    logger.info("Starting Disease Crawler")
    logger.info("=" * 50)
    
    existing = load_json('diseases.json')
    existing_urls = {d['url'] for d in existing}
    
    all_urls = crawl_disease_list()
    new_urls = [url for url in all_urls if url not in existing_urls]
    
    logger.info(f"Existing: {len(existing_urls)}, New: {len(new_urls)}")
    
    diseases = existing.copy()
    
    for i, url in enumerate(new_urls):
        disease = parse_disease_detail(url)
        if disease:
            diseases.append(disease)
            logger.info(f"[{i+1}/{len(new_urls)}] Crawled: {disease.get('ten_benh', url)}")
        
        if (i + 1) % 10 == 0:
            save_json(diseases, 'diseases.json')
    
    save_json(diseases, 'diseases.json')
    return diseases


# ============ ARTICLE CRAWLER ============

def crawl_article_list():
    """Get article URLs from safe drug usage section"""
    logger.info("Discovering article URLs...")
    article_urls = set()
    
    # Multiple article categories
    categories = [
        f"{BASE_URL}/vie/su-dung-thuoc-an-toan",
        f"{BASE_URL}/vie/bai-viet",
    ]
    
    for cat_url in categories:
        page = 1
        while page <= 200:
            if page == 1:
                url = cat_url
            else:
                url = f"{cat_url}/page_{page}"
            
            soup = get_soup(url)
            if not soup:
                # Try alternative pagination
                url = f"{cat_url}?page={page}"
                soup = get_soup(url)
                if not soup:
                    break
            
            found_new = False
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/vie/bai-viet/' in href:
                    full_url = urljoin(BASE_URL, href)
                    if full_url not in article_urls:
                        found_new = True
                        article_urls.add(full_url)
            
            if not found_new and page > 1:
                break
            
            logger.info(f"Category page {page}: {len(article_urls)} articles")
            page += 1
    
    logger.info(f"Total article URLs: {len(article_urls)}")
    return list(article_urls)


def parse_article_detail(url):
    """Parse article page"""
    soup = get_soup(url)
    if not soup:
        return None
    
    article = {
        'url': url,
        'crawl_time': datetime.now().isoformat()
    }
    
    # Get title
    h1 = soup.find('h1')
    if h1:
        article['tieu_de'] = clean_text(h1.get_text())
    
    # Get description
    meta = soup.find('meta', {'name': 'description'})
    if meta:
        article['mo_ta'] = meta.get('content', '')
    
    # Parse content
    main = soup.find('article') or soup.find('div', {'class': re.compile(r'content|article|post')})
    
    if main:
        sections = []
        current_section = {'title': '', 'content': []}
        
        for elem in main.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li']):
            text = clean_text(elem.get_text())
            
            if elem.name in ['h2', 'h3', 'h4']:
                if current_section['content']:
                    sections.append(current_section)
                current_section = {'title': text, 'content': []}
            elif text and len(text) > 15:
                current_section['content'].append(text)
        
        if current_section['content']:
            sections.append(current_section)
        
        article['phan_doan'] = sections
        
        # Full text
        full_text = []
        for section in sections:
            if section['title']:
                full_text.append(f"\n## {section['title']}\n")
            full_text.extend(section['content'])
        article['noi_dung'] = '\n'.join(full_text)
    
    # Tags
    tags = []
    for link in soup.find_all('a', href=lambda h: h and '/vie/chu-de/' in h):
        tag = clean_text(link.get_text())
        if tag:
            tags.append(tag)
    article['chu_de'] = list(set(tags))
    
    return article


def crawl_all_articles():
    """Crawl all articles"""
    logger.info("=" * 50)
    logger.info("Starting Article Crawler")
    logger.info("=" * 50)
    
    existing = load_json('articles.json')
    existing_urls = {a['url'] for a in existing}
    
    all_urls = crawl_article_list()
    new_urls = [url for url in all_urls if url not in existing_urls]
    
    logger.info(f"Existing: {len(existing_urls)}, New: {len(new_urls)}")
    
    articles = existing.copy()
    
    for i, url in enumerate(new_urls):
        article = parse_article_detail(url)
        if article and article.get('noi_dung'):
            articles.append(article)
            logger.info(f"[{i+1}/{len(new_urls)}] Crawled: {article.get('tieu_de', url)[:50]}...")
        
        if (i + 1) % 10 == 0:
            save_json(articles, 'articles.json')
    
    save_json(articles, 'articles.json')
    return articles


# ============ TRAINING DATA GENERATOR ============

def generate_training_data():
    """Generate training dataset for AI"""
    logger.info("=" * 50)
    logger.info("Generating Training Data")
    logger.info("=" * 50)
    
    training_data = []
    
    # Process drugs
    drugs = load_json('drugs.json')
    for drug in drugs:
        if drug.get('ten_thuoc'):
            entry = {
                'type': 'thuoc',
                'ten': drug.get('ten_thuoc'),
                'text': f"Thông tin thuốc {drug.get('ten_thuoc')}:\n{drug.get('noi_dung_day_du', '')}",
                'metadata': {
                    'url': drug.get('url'),
                    'chi_dinh': drug.get('chi_dinh', ''),
                    'chong_chi_dinh': drug.get('chong_chi_dinh', ''),
                    'tac_dung_phu': drug.get('tac_dung_phu', ''),
                    'lieu_dung': drug.get('lieu_dung', ''),
                }
            }
            training_data.append(entry)
    
    # Process diseases
    diseases = load_json('diseases.json')
    for disease in diseases:
        if disease.get('ten_benh'):
            entry = {
                'type': 'benh',
                'ten': disease.get('ten_benh'),
                'text': f"Thông tin bệnh {disease.get('ten_benh')}:\n{disease.get('noi_dung_day_du', '')}",
                'metadata': {
                    'url': disease.get('url'),
                    'nguyen_nhan': disease.get('nguyen_nhan', ''),
                    'trieu_chung': disease.get('trieu_chung', ''),
                    'dieu_tri': disease.get('dieu_tri', ''),
                    'phong_ngua': disease.get('phong_ngua', ''),
                }
            }
            training_data.append(entry)
    
    # Process articles
    articles = load_json('articles.json')
    for article in articles:
        if article.get('tieu_de') and article.get('noi_dung'):
            entry = {
                'type': 'bai_viet',
                'ten': article.get('tieu_de'),
                'text': f"{article.get('tieu_de')}\n\n{article.get('noi_dung')}",
                'metadata': {
                    'url': article.get('url'),
                    'mo_ta': article.get('mo_ta', ''),
                    'chu_de': article.get('chu_de', []),
                }
            }
            training_data.append(entry)
    
    # Save as JSON
    save_json(training_data, 'training_data.json')
    
    # Save as JSONL (for easier processing)
    jsonl_path = OUTPUT_DIR / 'training_data.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved JSONL to {jsonl_path}")
    
    # Generate Q&A pairs
    qa_pairs = []
    
    for drug in drugs:
        name = drug.get('ten_thuoc', '')
        if not name:
            continue
        
        if drug.get('chi_dinh'):
            qa_pairs.append({
                'question': f"Thuốc {name} dùng để điều trị gì?",
                'answer': drug['chi_dinh'],
                'type': 'drug_indication'
            })
        
        if drug.get('tac_dung_phu'):
            qa_pairs.append({
                'question': f"Tác dụng phụ của thuốc {name} là gì?",
                'answer': drug['tac_dung_phu'],
                'type': 'drug_side_effect'
            })
        
        if drug.get('lieu_dung'):
            qa_pairs.append({
                'question': f"Liều dùng thuốc {name} như thế nào?",
                'answer': drug['lieu_dung'],
                'type': 'drug_dosage'
            })
    
    for disease in diseases:
        name = disease.get('ten_benh', '')
        if not name:
            continue
        
        if disease.get('trieu_chung'):
            qa_pairs.append({
                'question': f"Triệu chứng của bệnh {name} là gì?",
                'answer': disease['trieu_chung'],
                'type': 'disease_symptom'
            })
        
        if disease.get('dieu_tri'):
            qa_pairs.append({
                'question': f"Cách điều trị bệnh {name}?",
                'answer': disease['dieu_tri'],
                'type': 'disease_treatment'
            })
        
        if disease.get('phong_ngua'):
            qa_pairs.append({
                'question': f"Làm thế nào để phòng ngừa bệnh {name}?",
                'answer': disease['phong_ngua'],
                'type': 'disease_prevention'
            })
    
    save_json(qa_pairs, 'qa_pairs.json')
    
    logger.info(f"Generated {len(training_data)} training entries and {len(qa_pairs)} Q&A pairs")
    return training_data, qa_pairs


def print_summary():
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT DỮ LIỆU ĐÃ THU THẬP")
    print("=" * 60)
    
    stats = {}
    files = {
        'drugs.json': 'Thuốc',
        'diseases.json': 'Bệnh',
        'articles.json': 'Bài viết',
        'training_data.json': 'Training entries',
        'qa_pairs.json': 'Q&A pairs',
    }
    
    for filename, label in files.items():
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats[label] = len(data)
                print(f"  ✓ {label}: {len(data)} mục")
        else:
            print(f"  ✗ {label}: Chưa có dữ liệu")
    
    print("=" * 60)
    print(f"📁 Dữ liệu được lưu tại: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║      VINMEC COMPLETE MEDICAL DATA CRAWLER                     ║
    ║      Thu thập dữ liệu y tế để huấn luyện AI                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # Run crawlers
  
    print("\n🏥 [2/4] Thu thập dữ liệu bệnh...")
    crawl_all_diseases()
    
    print("\n📄 [3/4] Thu thập bài viết y tế...")
    crawl_all_articles()
    
    print("\n🤖 [4/4] Tạo dữ liệu huấn luyện AI...")
    generate_training_data()
    
    # Summary
    print_summary()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Tổng thời gian: {elapsed/60:.1f} phút")
    print("✅ Hoàn thành!")


if __name__ == "__main__":
    main()
