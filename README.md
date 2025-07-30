# Intelligent Character Recognition (ICR) Pipeline

An **AI-powered Intelligent Character Recognition (ICR)** pipeline that extracts **structured invoice data** from images and PDFs using:

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** for document layout and text extraction.
- **[Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** for AI-based document understanding and JSON extraction.
- **[Groq API](https://groq.com/)** for fast LLM-based structured data parsing.
- Rule-based extraction as a **fallback** when AI models cannot parse the document.

---

## Features

- OCR text extraction from **invoices, receipts, and forms**.
- **Document structure parsing** using PaddleOCR `PPStructureV3`.
- **AI JSON extraction** using Phi-3 or Groq LLMs.
- Fallback to **regex-based extraction** when LLM fails.
- **Merges multi-model outputs** for improved accuracy.
- Output stored as **JSON** for easy integration.

---



