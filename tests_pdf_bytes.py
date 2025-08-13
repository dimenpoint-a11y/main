# tests_pdf_bytes.py
from app_trader_pro_v643 import to_pdf_bytes

def run_tests():
    data = to_pdf_bytes("Test", "Hello\nWorld")
    assert isinstance(data, (bytes, bytearray)), "PDF function must return bytes"
    assert data.startswith(b"%PDF"), "PDF bytes should start with %PDF"
    print("PDF byte generation test PASSED ✅")

if __name__ == "__main__":
    run_tests()