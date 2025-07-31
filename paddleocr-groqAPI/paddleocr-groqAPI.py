import os
import re
import json
from pdf2image import convert_from_path
from paddleocr import PPStructureV3
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------------- PDF → Images ----------------
def convert_pdf_to_images(pdf_path, output_dir="output/pdf_pages"):
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, page in enumerate(pages):
        img_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        page.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths

# ---------------- OCR ----------------
def run_ocr(file_path, output_dir="output"):
    print(f"📄 Running OCR on: {file_path}")

    if file_path.lower().endswith(".pdf"):
        image_paths = convert_pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    pipeline = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)
    all_text = []

    for img_path in image_paths:
        file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        output = pipeline.predict(input=img_path)

        for res in output:
            res.save_to_json(save_path=output_dir)

        json_path = os.path.join(output_dir, f"{file_name_no_ext}_res.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
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

# ---------------- Clean OCR Text ----------------
def preprocess_ocr_text(ocr_text):
    lines = ocr_text.split("\n")
    seen = set()
    filtered = []
    for line in lines:
        clean_line = line.strip()
        if not clean_line or clean_line in seen or clean_line.isdigit():
            continue
        seen.add(clean_line)
        filtered.append(clean_line)
    return "\n".join(filtered)

# ---------------- Groq Call ----------------
def get_invoice_insights_groq(clean_text):
    client = Groq(api_key=os.getenv('groq_api_key'))

    prompt = f"""
Extract structured invoice details from the following OCR text.

OCR TEXT:
{clean_text}

Return only a valid JSON object.
No explanations. No extra text.
Do not repeat the question or OCR text.
Only output JSON.
""".strip()

    chat_completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": "You only output valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=2048
    )

    return chat_completion.choices[0].message.content.strip()

# ---------------- JSON Extraction ----------------
def extract_json_from_response(response, raw_path="output/raw_groq.json"):
    # Remove <think> blocks if present
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Find JSON part
    match = re.search(r"\{[\s\S]*\}", response)
    json_text = match.group(0).strip() if match else "{}"

    # Save raw JSON to debug
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(json_text)

    # Try parse
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        print("⚠️ JSON invalid, saving raw text.")
        return json_text  # Keep raw text if invalid

# ---------------- Main ----------------
if __name__ == "__main__":
    FILE_PATH = "MongoDB Atlas Invoice - 2025-07-01 - ConfirmU.pdf"
    OUTPUT_JSON = "output/invoice_insights.json"
    os.makedirs("output", exist_ok=True)

    # Step 1: OCR
    raw_text = run_ocr(FILE_PATH)

    # Step 2: Clean text
    clean_text = preprocess_ocr_text(raw_text)

    # Step 3: Groq extraction
    groq_output = get_invoice_insights_groq(clean_text)

    # Step 4: Extract & Save JSON
    extracted_data = extract_json_from_response(groq_output)

    # Save final structured insights
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        if isinstance(extracted_data, dict):
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        else:
            f.write(extracted_data)  # If raw string

    print(f"✅ Final structured invoice data saved to {OUTPUT_JSON}")
    print(f"📝 Raw Groq JSON saved to output/raw_groq.json")
