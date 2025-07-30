import os
import json
import re
from paddleocr import PPStructureV3
from groq import Groq


# ------------------------------
# 1. OCR Extraction
# ------------------------------
def run_ocr(image_path, output_dir="output"):
    print("📄 Running PaddleOCR on:", image_path)
    pipeline = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)

    file_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    output = pipeline.predict(input=image_path)

    for res in output:
        res.save_to_json(save_path=output_dir)

    ocr_json_path = f"{output_dir}/{file_name_no_ext}_res.json"
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    cleaned_ocr = []
    if isinstance(ocr_data, dict) and "parsing_res_list" in ocr_data:
        src_list = ocr_data["parsing_res_list"]
    elif isinstance(ocr_data, list) and ocr_data and "parsing_res_list" in ocr_data[0]:
        src_list = ocr_data[0]["parsing_res_list"]
    else:
        src_list = []

    for item in src_list:
        label = item.get("block_label", "").strip()
        content = item.get("block_content", "").strip()
        if content:
            cleaned_ocr.append({"label": label, "content": content})

    print("📝 Cleaned OCR Output:", json.dumps(cleaned_ocr, indent=2, ensure_ascii=False))
    return cleaned_ocr


# ------------------------------
# 2. OCR Noise Cleaning
# ------------------------------
def clean_ocr_noise(ocr_json):
    """
    Removes duplicates, numbers, headers/footers, and irrelevant noise
    from OCR JSON output before flattening.
    """
    seen = set()
    filtered = []

    for item in ocr_json:
        text = item["content"].strip()

        # Skip empty lines
        if not text:
            continue

        # Remove duplicates
        if text in seen:
            continue
        seen.add(text)

        # Skip pure numbers (page numbers, indexes)
        if text.isdigit():
            continue

        # Skip repeated "total amount" lines
        if re.search(r"total amount", text, re.I):
            continue

        # Skip placeholders or repeated headers
        if re.search(r"invoice insights|page \d+|thank you", text, re.I):
            continue

        filtered.append({"label": item["label"], "content": text})

    print("🧹 Cleaned OCR Noise Output:", json.dumps(filtered, indent=2, ensure_ascii=False))
    return filtered


# ------------------------------
# 3. Flatten OCR Text
# ------------------------------
def flatten_ocr_text(ocr_json):
    flat_text_lines = []
    for idx, item in enumerate(ocr_json, start=1):
        line = f"text{idx}: {item['content']}"
        flat_text_lines.append(line)
    flat_text = "\n".join(flat_text_lines)
    print("\n🧾 Structured Flattened OCR Text:\n", flat_text)
    return flat_text


# ------------------------------
# 4. Rule-Based Fallback
# ------------------------------
def rule_based_extraction(ocr_json):
    print("⚠️ Using Rule-Based Extraction Fallback...")
    flat_text = " ".join([item["content"] for item in ocr_json]).lower()

    result = {
        "vendor_name": "",
        "customer_name": "",
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "total_amount": "",
        "tax_amount": "",
        "payment_terms": "",
        "items_summary": "",
        "possible_risks": "",
        "recommendations": ""
    }

    # Simple regex-based extraction
    inv_match = re.search(r"(invoice\s*no\.?|no\.)\s*[:\-]?\s*([A-Za-z0-9\-]+)", flat_text)
    if inv_match:
        result["invoice_number"] = inv_match.group(2)

    date_match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", flat_text)
    if date_match:
        result["invoice_date"] = date_match.group(1)

    vendor_match = re.search(r"(inc\.|ltd|corp)", flat_text)
    if vendor_match:
        result["vendor_name"] = vendor_match.group(0)

    bank_match = re.search(r"bankname[:\s]+([A-Za-z\s]+)", flat_text)
    if bank_match:
        result["payment_terms"] = bank_match.group(1)

    return result


# ------------------------------
# 5. Call Groq API for LLM Extraction
# ------------------------------
def get_invoice_insights_groq(flat_text):
    client = Groq(api_key="")

    schema = {
        "vendor_name": "",
        "customer_name": "",
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "total_amount": "",
        "tax_amount": "",
        "payment_terms": "",
        "items_summary": "",
        "possible_risks": "",
        "recommendations": ""
    }

    prompt = f"""
You are an AI that extracts structured insights from OCR invoice data.

Given the following OCR text:
{flat_text}

Extract it into this JSON format:
{json.dumps(schema, indent=2)}

Rules:
- Fill in as much info as possible
- Leave blank if not found
- Only output valid JSON
"""

    print("\n📢 Sending prompt to Groq API...\n")

    chat_completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": "You are a precise JSON extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )

    response = chat_completion.choices[0].message.content
    print("\n🤖 Groq Raw Response:\n", response)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("❌ Groq returned invalid JSON — falling back to rules.")
        return None


# ------------------------------
# 6. Main Pipeline
# ------------------------------
if __name__ == "__main__":
    IMAGE_PATH = "image-62.png"
    OUTPUT_JSON = "output/invoice_insights.json"

    os.makedirs("output", exist_ok=True)

    # Step 1: OCR
    raw_ocr = run_ocr(IMAGE_PATH)

    # Step 2: Clean OCR noise
    cleaned_ocr = clean_ocr_noise(raw_ocr)

    # Step 3: Flatten OCR for LLM
    flat_text = flatten_ocr_text(cleaned_ocr)

    # Step 4: Groq LLM extraction
    insights = get_invoice_insights_groq(flat_text)

    # Step 5: Fallback to rules if LLM fails
    if insights is None:
        insights = rule_based_extraction(cleaned_ocr)

    # Step 6: Save results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Invoice insights saved to {OUTPUT_JSON}")
