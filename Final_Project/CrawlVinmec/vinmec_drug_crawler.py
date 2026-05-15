import json
import time
import re
import os
import logging
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.vinmec.com"
DRUG_LIST_URL = "https://www.vinmec.com/vie/thuoc/"
OUTPUT_DIR = "vinmec_complete_data"

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'})


def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver


def collect_drug_urls_from_page(driver):
    """Thu thập URLs thuốc từ trang hiện tại"""
    urls = set()
    try:
        links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/vie/thuoc/"]')
        for link in links:
            try:
                href = link.get_attribute('href')
                if href and re.search(r'/vie/thuoc/[a-zA-Z0-9\-]+-\d+$', href):
                    urls.add(href)
            except:
                continue
    except Exception as e:
        logger.error(f"Lỗi thu thập URLs: {e}")
    return urls


def discover_drug_urls(driver):
    """Click vào từng chữ cái A-Z để lấy danh sách thuốc"""
    all_urls = set()
    
    print("\n" + "="*60)
    print("   VINMEC DRUG CRAWLER")
    print("   Click vào từng chữ cái A-Z để lấy thuốc")
    print("="*60 + "\n")
    
    # Load trang chính trước
    logger.info("Đang load trang danh mục thuốc...")
    driver.get(DRUG_LIST_URL)
    time.sleep(5)
    
    # Lấy thuốc từ trang mặc định
    default_urls = collect_drug_urls_from_page(driver)
    all_urls.update(default_urls)
    logger.info(f"Trang mặc định: {len(default_urls)} thuốc")
    
    # Tìm và click vào từng chữ cái
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for letter in alphabet:
        try:
            # Reload trang để tránh stale elements
            driver.get(DRUG_LIST_URL)
            time.sleep(3)
            
            # Tìm nút chữ cái - thử nhiều cách
            letter_btn = None
            
            # Cách 1: Tìm bằng text chính xác
            try:
                buttons = driver.find_elements(By.XPATH, f"//a[text()='{letter}'] | //button[text()='{letter}'] | //span[text()='{letter}']/parent::a | //span[text()='{letter}']/parent::button")
                if buttons:
                    letter_btn = buttons[0]
            except:
                pass
            
            # Cách 2: Tìm trong danh sách alphabet
            if not letter_btn:
                try:
                    all_btns = driver.find_elements(By.CSS_SELECTOR, 'a, button, span')
                    for btn in all_btns:
                        if btn.text.strip() == letter:
                            letter_btn = btn
                            break
                except:
                    pass
            
            if letter_btn:
                # Scroll đến nút và click
                driver.execute_script("arguments[0].scrollIntoView(true);", letter_btn)
                time.sleep(0.5)
                
                try:
                    letter_btn.click()
                except:
                    driver.execute_script("arguments[0].click();", letter_btn)
                
                time.sleep(3)
                
                # Thu thập URLs sau khi click
                page_urls = collect_drug_urls_from_page(driver)
                new_urls = page_urls - all_urls
                all_urls.update(page_urls)
                
                logger.info(f"Chữ {letter}: +{len(new_urls)} thuốc mới (Tổng: {len(all_urls)})")
                
                # Kiểm tra pagination
                page_num = 2
                while page_num <= 20:
                    try:
                        # Tìm nút trang tiếp theo
                        next_btns = driver.find_elements(By.XPATH, f"//a[text()='{page_num}'] | //a[contains(@href, 'page={page_num}')]")
                        if not next_btns:
                            break
                        
                        next_btns[0].click()
                        time.sleep(2)
                        
                        page_urls = collect_drug_urls_from_page(driver)
                        new_in_page = page_urls - all_urls
                        
                        if not new_in_page:
                            break
                            
                        all_urls.update(page_urls)
                        logger.info(f"  Chữ {letter} trang {page_num}: +{len(new_in_page)} (Tổng: {len(all_urls)})")
                        page_num += 1
                        
                    except Exception as e:
                        break
            else:
                logger.warning(f"Không tìm thấy nút chữ {letter}")
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý chữ {letter}: {e}")
            continue
    
    logger.info(f"\n=== TỔNG SỐ URLs THUỐC: {len(all_urls)} ===")
    return list(all_urls)


def parse_drug_detail(url):
    """Parse chi tiết thuốc"""
    try:
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, 'lxml')
    except:
        return None
    
    drug_data = {
        'url': url,
        'name': '',
        'formulation': '',
        'drug_group': '',
        'indication': '',
        'contraindication': '',
        'precaution': '',
        'side_effects': '',
        'dosage': '',
        'usage_notes': '',
        'related_topics': [],
    }
    
    # Lấy tên từ title
    title = soup.find('title')
    if title:
        text = title.get_text(strip=True)
        drug_data['name'] = text.split('|')[0].strip() if '|' in text else text
    
    # Parse sections
    section_map = {
        'dạng bào chế': 'formulation',
        'nhóm thuốc': 'drug_group',
        'chỉ định': 'indication',
        'chống chỉ định': 'contraindication',
        'thận trọng': 'precaution',
        'tác dụng không mong muốn': 'side_effects',
        'liều': 'dosage',
        'chú ý': 'usage_notes',
    }
    
    for h2 in soup.find_all('h2'):
        title_text = h2.get_text(strip=True).lower()
        for key, field in section_map.items():
            if key in title_text:
                parts = []
                sibling = h2.find_next_sibling()
                while sibling and sibling.name != 'h2':
                    if sibling.name in ['p', 'div', 'ul', 'ol']:
                        text = sibling.get_text(separator=' ', strip=True)
                        if text:
                            parts.append(text)
                    sibling = sibling.find_next_sibling()
                if parts:
                    drug_data[field] = ' '.join(parts)
                break
    
    # Full text
    parts = [f"Thuốc: {drug_data['name']}"]
    for field, name in [('formulation', 'Dạng bào chế'), ('drug_group', 'Nhóm thuốc'),
                        ('indication', 'Chỉ định'), ('contraindication', 'Chống chỉ định'),
                        ('precaution', 'Thận trọng'), ('side_effects', 'Tác dụng phụ'),
                        ('dosage', 'Liều dùng'), ('usage_notes', 'Chú ý')]:
        if drug_data.get(field):
            parts.append(f"{name}: {drug_data[field]}")
    
    drug_data['full_text'] = '\n'.join(parts)
    return drug_data


def crawl_details(urls):
    """Crawl chi tiết thuốc"""
    drugs = []
    total = len(urls)
    
    for i, url in enumerate(urls, 1):
        drug = parse_drug_detail(url)
        if drug and drug.get('name'):
            drugs.append(drug)
            logger.info(f"[{i}/{total}] ✓ {drug['name'][:40]}...")
        else:
            logger.warning(f"[{i}/{total}] ✗ {url}")
        
        if i % 20 == 0:
            save_drugs(drugs)
        time.sleep(0.3)
    
    return drugs


def save_drugs(drugs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'drugs.json'), 'w', encoding='utf-8') as f:
        json.dump(drugs, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu {len(drugs)} thuốc")


def generate_training_data(drugs):
    training_data = []
    qa_pairs = []
    
    for drug in drugs:
        if not drug.get('name'):
            continue
        
        name = drug['name']
        if drug.get('full_text'):
            training_data.append({'type': 'drug', 'name': name, 'text': drug['full_text'], 'url': drug.get('url', '')})
        
        for field, q_template, a_template in [
            ('indication', f"Thuốc {name} dùng để gì?", f"Chỉ định: {{}}"),
            ('contraindication', f"Chống chỉ định của {name}?", f"Chống chỉ định: {{}}"),
            ('dosage', f"Liều dùng {name}?", f"Liều dùng: {{}}"),
            ('side_effects', f"Tác dụng phụ của {name}?", f"Tác dụng phụ: {{}}"),
        ]:
            if drug.get(field):
                qa_pairs.append({'question': q_template, 'answer': a_template.format(drug[field])})
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'drug_training_data.json'), 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'drug_qa_pairs.json'), 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Training: {len(training_data)} entries, {len(qa_pairs)} Q&A pairs")


def main():
    print("\n" + "="*60)
    print("   VINMEC DRUG CRAWLER")
    print("="*60 + "\n")
    
    # Load existing
    existing = []
    drugs_file = os.path.join(OUTPUT_DIR, 'drugs.json')
    if os.path.exists(drugs_file):
        try:
            with open(drugs_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            logger.info(f"Đã load {len(existing)} thuốc có sẵn")
        except:
            pass
    
    existing_urls = {d.get('url') for d in existing if d.get('url')}
    
    logger.info("Khởi tạo Selenium...")
    driver = setup_driver()
    
    try:
        all_urls = discover_drug_urls(driver)
        new_urls = [u for u in all_urls if u not in existing_urls]
        logger.info(f"URLs mới: {len(new_urls)}")
        
        if new_urls:
            new_drugs = crawl_details(new_urls)
            all_drugs = existing + new_drugs
            save_drugs(all_drugs)
            generate_training_data(all_drugs)
            print(f"\n✅ HOÀN THÀNH! Tổng: {len(all_drugs)} thuốc")
        else:
            logger.info("Không có thuốc mới")
            if existing:
                generate_training_data(existing)
    finally:
        try:
            driver.quit()
        except:
            pass
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
