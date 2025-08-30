from src.crawl_page import crawl_page
from src.data_processing import process_raw_data

def main():
    # Bước 1: Crawl từ các URL dịch vụ
    print("----- [1] Crawling dịch vụ từ config ----")
    crawl_page()

    # Bước 2: Làm sạch và lưu dữ liệu
    print("----- [2] Xử lý và lưu dữ liệu sạch -----")
    process_raw_data()

if __name__ == "__main__":
    main()