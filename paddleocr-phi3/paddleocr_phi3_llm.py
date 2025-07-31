import os
import json
import re
import torch
from pdf2image import convert_from_path
from paddleocr import PPStructureV3
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------- OCR ----------------
def convert_pdf_to_images(pdf_path, output_dir="output/pdf_pages"):
    """Convert PDF to list of image paths."""
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        page.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths


def run_ocr(file_path, output_dir="output"):
    """Run OCR on image or PDF."""
    print("Running OCR on:", file_path)

    if file_path.lower().endswith(".pdf"):
        image_paths = convert_pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )

    all_text = []
    for img_path in image_paths:
        file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        output = pipeline.predict(input=img_path)

        for res in output:
            res.save_to_json(save_path=output_dir)

        ocr_json_path = f"{output_dir}/{file_name_no_ext}_res.json"
        if not os.path.exists(ocr_json_path):
            continue

        with open(ocr_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        if isinstance(ocr_data, dict) and "parsing_res_list" in ocr_data:
            src_list = ocr_data["parsing_res_list"]
        elif isinstance(ocr_data, list) and ocr_data and "parsing_res_list" in ocr_data[0]:
            src_list = ocr_data[0]["parsing_res_list"]
        else:
            src_list = []

        for item in src_list:
            content = item.get("block_content", "").strip()
            if content:
                all_text.append(content)

    return "\n".join(all_text)


# ---------------- TEXT CLEANING ----------------
def preprocess_ocr_text(ocr_text):
    """Remove duplicates, empty lines, and noise."""
    lines = ocr_text.split("\n")
    seen = set()
    filtered = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if line_clean in seen:
            continue
        seen.add(line_clean)

        if line_clean.isdigit():  # Skip numbers only
            continue

        filtered.append(line_clean)

    cleaned_text = "\n".join(filtered)
    print("\nCleaned OCR Text:\n", cleaned_text)
    return cleaned_text


# ---------------- LLM LOADING ----------------
def load_phi3_local(model_dir="microsoft/Phi-3-mini-4k-instruct"):
    """Load Phi-3 with 4-bit quantization."""
    print(f"Loading model: {model_dir} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=bnb_config
    )
    print("Model loaded successfully")
    return tokenizer, model


# ---------------- PROMPT ----------------
def build_prompt(ocr_text):
    """Builds minimal JSON-only extraction prompt."""
    return f"""
Extract structured invoice details from the following OCR text.

OCR TEXT:
{ocr_text}

Return only a valid JSON object. 
No explanations. No extra text. 
Do not repeat the question or OCR text.
Only output JSON.
""".strip()


# ---------------- OUTPUT CLEANING ----------------
def clean_model_output(response):
    """Strip everything before first { to keep only JSON."""
    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        return match.group(0)
    return "{}"


# ---------------- JSON PARSING ----------------
def extract_json_from_response(response):
    """Extract JSON object from model response."""
    cleaned = clean_model_output(response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


# ---------------- RULE-BASED FALLBACK ----------------
def rule_based_fallback(ocr_text):
    """Fallback regex extraction for invoices."""
    result = {
        "invoice_number": "",
        "invoice_date": "",
        "vendor_name": "",
        "amount_due": ""
    }

    inv_match = re.search(r"(invoice\s*(no\.?|number)?)\s*[:\-]?\s*([A-Z0-9\-]+)", ocr_text, re.I)
    if inv_match:
        result["invoice_number"] = inv_match.group(3)

    date_match = re.search(r"\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b", ocr_text)
    if date_match:
        result["invoice_date"] = date_match.group(0)

    vendor_match = re.search(r"([A-Z][A-Za-z\s]+(Ltd|Limited|Inc|Corporation|Corp))", ocr_text)
    if vendor_match:
        result["vendor_name"] = vendor_match.group(1)

    amount_match = re.search(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", ocr_text)
    if amount_match:
        result["amount_due"] = amount_match.group(0)

    return result


# ---------------- MAIN ----------------
if __name__ == "__main__":
    FILE_PATH = "MongoDB Atlas Invoice - 2025-07-01 - ConfirmU.pdf"
    OUTPUT_JSON = "output/MongoDB Atlas Invoice - 2025-07-01 - ConfirmU.json"

    os.makedirs("output", exist_ok=True)

    # Step 1: OCR
    raw_ocr_text = run_ocr(FILE_PATH)

    # Step 2: Preprocess OCR
    clean_ocr_text = preprocess_ocr_text(raw_ocr_text)

    # Step 3: Load Model
    tokenizer, model = load_phi3_local()

    # Step 4: Build Prompt
    prompt = build_prompt(clean_ocr_text)

    # Step 5: Model Inference
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,  # Large enough for nested JSON
        do_sample=False
    )

    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nRaw LLM Output:\n", raw_response)

    # Step 6: Extract JSON
    extracted_data = extract_json_from_response(raw_response)

    # Step 7: Fallback if LLM fails
    if not extracted_data or all(v == "" for v in extracted_data.values()):
        print("Model extraction failed. Using fallback...")
        extracted_data = rule_based_fallback(clean_ocr_text)

    # Step 8: Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Final Extracted JSON saved to {OUTPUT_JSON}")
