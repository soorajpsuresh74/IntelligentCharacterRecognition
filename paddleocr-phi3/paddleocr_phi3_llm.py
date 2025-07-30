import os
import json
import re
import torch
from paddleocr import PPStructureV3
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def run_ocr(image_path, output_dir="output"):
    print("Running PaddleOCR on:", image_path)
    pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )

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
            cleaned_ocr.append(content)

    return "\n".join(cleaned_ocr)


def preprocess_ocr_text(ocr_text):
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

        if line_clean.isdigit():
            continue

        filtered.append(line_clean)

    cleaned_text = "\n".join(filtered)
    print("\nCleaned OCR Text:\n", cleaned_text)
    return cleaned_text


def load_phi3_local(model_dir="microsoft/Phi-3-mini-4k-instruct"):
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


def build_prompt(ocr_text):
    return f"""
You are an Intelligent Character Recognition System that extracts structured documents fields from Optical Character Recognition text.

Return JSON in this format consisting of insights from the OCR
{{
  "Key1": "Value1",
  "Key2": "Value2",
  "Key3": "Value3",
  "Key4": "Value4",
}}

Fill the values from this text:
{ocr_text}

Only output valid JSON. No extra text.
""".strip()

def extract_json_from_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}



def rule_based_fallback(ocr_text):
    result = {
        "invoice_number": "",
        "invoice_date": "",
        "vendor_name": "",
        "total_amount": ""
    }

    inv_match = re.search(r"(invoice\s*(no\.?|number)?)\s*[:\-]?\s*([A-Z0-9\-]+)", ocr_text, re.I)
    if inv_match:
        result["invoice_number"] = inv_match.group(3)

    date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", ocr_text)
    if date_match:
        result["invoice_date"] = date_match.group(0)

    vendor_match = re.search(r"\b([A-Z][A-Za-z\s]+(Inc\.|Ltd|Corp))\b", ocr_text)
    if vendor_match:
        result["vendor_name"] = vendor_match.group(1)

    total_match = re.search(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", ocr_text)
    if total_match:
        result["total_amount"] = total_match.group(0)

    return result


if __name__ == "__main__":
    IMAGE_PATH = "image-62.png"
    OUTPUT_JSON = "output/invoice_extracted.json"

    os.makedirs("output", exist_ok=True)

    # Step 1: OCR
    raw_ocr_text = run_ocr(IMAGE_PATH)

    # Step 2: Preprocess OCR
    clean_ocr_text = preprocess_ocr_text(raw_ocr_text)

    # Step 3: Load Model
    tokenizer, model = load_phi3_local()

    # Step 4: Build Prompt
    prompt = build_prompt(clean_ocr_text)

    # Step 5: Model Inference
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nRaw LLM Output:\n", response)

    # Step 6: Parse JSON
    extracted_data = extract_json_from_response(response)

    # Step 7: If model fails, use regex fallback
    if not extracted_data or all(v == "" for v in extracted_data.values()):
        print("Model extraction failed. Using fallback...")
        extracted_data = rule_based_fallback(clean_ocr_text)

    # Step 8: Save Result => ICR
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"Final Extracted JSON saved to {OUTPUT_JSON}")
