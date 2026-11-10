"""Đường dẫn dùng chung cho dữ liệu và báo cáo labeling."""

from pathlib import Path  # xử lý đường dẫn

ROOT = Path(__file__).resolve().parents[1]  # thư mục gốc project
REPORT_DIR = ROOT / "reports"  # thư mục chứa report/cache

REVIEWS_PATH = ROOT / "reviews.jsonl"  # review gốc dạng JSONL
REVIEWS_LABELED_PATH = ROOT / "reviews_labeled.jsonl"  # review đã gán nhãn
REVIEWS_LABELED_CSV_PATH = ROOT / "reviews_labeled.csv"  # bản CSV của nhãn

LABEL_STATS_PATH = REPORT_DIR / "label_stats.json"  # thống kê labeling
UNDERTHESEA_CACHE_PATH = REPORT_DIR / "underthesea_cache.jsonl"  # cache underthesea
