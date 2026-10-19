"""
Automated Test Runner for Online Retail Data Mining Pipeline.
Executes all 47 tests across 4 test categories, generates reports, and displays validation results.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import time
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "outputs" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
XML_REPORT = EVIDENCE_DIR / "pytest_report.xml"
CSV_REPORT = EVIDENCE_DIR / "test_execution_report.csv"

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


CATEGORY_METADATA = [
    {
        "marker": "unit",
        "name": "Unit Tests (Tiền xử lý & Tiêu chuẩn dữ liệu)",
        "expected_count": 15,
        "note": "Kiểm tra schema 8 cột, lọc trùng lặp, xử lý đơn âm và cước phí."
    },
    {
        "marker": "leakage",
        "name": "Leakage Prevention Invariance Tests",
        "expected_count": 8,
        "note": "Khẳng định tuyệt đối không rò rỉ dữ liệu giữa Train và Test."
    },
    {
        "marker": "parity",
        "name": "Model Parity & Determinism Tests",
        "expected_count": 12,
        "note": "Kiểm tra tính nhất quán giữa file chạy console và Jupyter Notebook."
    },
    {
        "marker": "smoke",
        "name": "Integration & Pipeline Smoke Tests",
        "expected_count": 12,
        "note": "Kiểm tra chạy thông suốt từ đầu đến cuối pipeline và giao diện Web."
    }
]


class TestAuditPlugin:
    def __init__(self):
        self.stats = {
            cat["marker"]: {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
            for cat in CATEGORY_METADATA
        }
        self.all_passed = True

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            for cat in CATEGORY_METADATA:
                marker = cat["marker"]
                if marker in report.keywords:
                    self.stats[marker]["total"] += 1
                    if report.passed:
                        self.stats[marker]["passed"] += 1
                    elif report.failed:
                        self.stats[marker]["failed"] += 1
                        self.all_passed = False
                    elif report.skipped:
                        self.stats[marker]["skipped"] += 1


def run_test_suite():
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'=' * 85}")
    print("KHỞI CHẠY HỆ THỐNG KIỂM THỬ TỰ ĐỘNG (PYTEST TEST SUITE)")
    print(f"{'=' * 85}{Colors.ENDC}\n")

    plugin = TestAuditPlugin()
    pytest_args = [
        str(PROJECT_ROOT / "tests"),
        "-q",
        f"--junitxml={XML_REPORT}"
    ]

    start_time = time.time()
    exit_code = pytest.main(pytest_args, plugins=[plugin])
    elapsed = time.time() - start_time

    # Build report data
    rows = []
    total_tests = 0
    total_passed = 0

    for cat in CATEGORY_METADATA:
        marker = cat["marker"]
        data = plugin.stats[marker]
        total = data["total"]
        passed = data["passed"]
        total_tests += total
        total_passed += passed

        result_str = f"{passed} Passed"
        if data["failed"] > 0:
            result_str += f", {data['failed']} Failed"
        if data["skipped"] > 0:
            result_str += f", {data['skipped']} Skipped"

        rows.append({
            "Phân nhóm kiểm thử": cat["name"],
            "Số lượng ca kiểm thử": f"{total} tests",
            "Kết quả thực tế": result_str,
            "Ghi chú nghiệm thu": cat["note"],
            "status": "PASS" if data["failed"] == 0 and total == cat["expected_count"] else "FAIL"
        })

    # Summary row
    grand_summary = {
        "Phân nhóm kiểm thử": "TỔNG CỘNG HỆ THỐNG KIỂM THỬ",
        "Số lượng ca kiểm thử": f"{total_tests} tests",
        "Kết quả thực tế": f"{total_passed}/{total_tests} PASSED ({100.0 * total_passed / max(total_tests, 1):.0f}%)",
        "Ghi chú nghiệm thu": "Hệ thống đạt chuẩn tin cậy và khả năng tái lập cấp độ công nghiệp.",
        "status": "PASS" if total_passed == total_tests == 47 else "FAIL"
    }

    # Save to CSV
    export_df = pd.DataFrame(rows + [grand_summary])
    export_df.to_csv(CSV_REPORT, index=False, encoding="utf-8-sig")

    # Display Table
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'=' * 110}")
    print(f"{'BẢNG BÁO CÁO NGHIỆM THU HỆ THỐNG KIỂM THỬ (TEST ACCEPTANCE REPORT)':^110}")
    print(f"{'=' * 110}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'Phân nhóm kiểm thử':<46} | {'Số lượng':<10} | {'Kết quả thực tế':<16} | {'Ghi chú nghiệm thu'}{Colors.ENDC}")
    print("-" * 110)

    for r in rows:
        color = Colors.OKGREEN if r["status"] == "PASS" else Colors.FAIL
        print(f"{r['Phân nhóm kiểm thử']:<46} | {r['Số lượng ca kiểm thử']:<10} | {color}{r['Kết quả thực tế']:<16}{Colors.ENDC} | {r['Ghi chú nghiệm thu']}")

    print("=" * 110)
    summary_color = Colors.OKGREEN if grand_summary["status"] == "PASS" else Colors.FAIL
    print(f"{Colors.BOLD}{grand_summary['Phân nhóm kiểm thử']:<46} | {grand_summary['Số lượng ca kiểm thử']:<10} | {summary_color}{grand_summary['Kết quả thực tế']:<16}{Colors.ENDC} | {grand_summary['Ghi chú nghiệm thu']}{Colors.ENDC}")
    print(f"{'=' * 110}\n")

    print(f"{Colors.OKGREEN}[XÁC NHẬN] Toàn bộ {total_tests} ca kiểm thử đã hoàn tất trong {elapsed:.2f}s.{Colors.ENDC}")
    print(f"Báo cáo chi tiết đã lưu: {CSV_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"Báo cáo JUnit XML đã lưu: {XML_REPORT.relative_to(PROJECT_ROOT)}\n")

    if exit_code != 0 or total_passed != 47:
        print(f"{Colors.FAIL}[LỖI] Hệ thống kiểm thử không đạt 100% (passed: {total_passed}/47)!{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    run_test_suite()
