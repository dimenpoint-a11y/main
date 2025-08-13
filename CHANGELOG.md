# Changelog
## v6.4.3
- **FIX**: PDF export crash on Streamlit Cloud (FPDF now returns bytes; no BytesIO passed to `output()`).
- **CLEANUP**: Single `to_pdf_bytes` helper (removed duplicate definition).
- **UX**: Added clear "Meaning" captions beneath major charts.
- **SECURITY/LEGAL**: Kept disclosures, disabled webhooks, personal-use only.