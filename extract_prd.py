from pypdf import PdfReader
from pathlib import Path

pdf_path = Path("docs/prd/LLM_Gateway_PRD_V2.2.pdf")
output_path = Path("docs/prd/LLM_Gateway_PRD_V2.2.md")

reader = PdfReader(pdf_path)

with output_path.open("w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        f.write(f"# Page {i}\n\n")
        f.write(text)
        f.write("\n\n")

print(f"Extracted {len(reader.pages)} pages to {output_path}")