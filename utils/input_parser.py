"""
FARSIX Input Parser — Extracts text from PDF, CSV, Image, or plain text inputs.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional, Tuple


def parse_pdf(file_bytes: bytes, max_pages: int = 30) -> Tuple[str, str]:
    """
    Extract text from a PDF file.

    Args:
        file_bytes: Raw PDF bytes.
        max_pages:  Maximum pages to extract.

    Returns:
        (extracted_text, metadata_summary)
    """
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            pages_to_extract = min(total_pages, max_pages)
            extracted_parts = []

            for i in range(pages_to_extract):
                page = pdf.pages[i]
                text = page.extract_text()
                if text and text.strip():
                    extracted_parts.append(
                        f"--- Page {i + 1} ---\n{text.strip()}"
                    )

            full_text = "\n\n".join(extracted_parts)

            if not full_text.strip():
                return (
                    "[PDF parsed but no extractable text found. "
                    "The document may contain only images or scanned content.]",
                    f"PDF: {total_pages} pages, no text"
                )

            meta = (
                f"PDF Document | {total_pages} total pages | "
                f"{pages_to_extract} pages extracted | "
                f"{len(full_text.split())} words"
            )
            return full_text, meta

    except ImportError:
        return (
            "[pdfplumber not installed. Install with: pip install pdfplumber]",
            "PDF: pdfplumber unavailable"
        )
    except Exception as exc:
        return (
            f"[PDF parsing error: {exc}]",
            f"PDF: error — {exc}"
        )


def parse_csv(file_bytes: bytes, filename: str = "data.csv") -> Tuple[str, str]:
    """
    Parse a CSV file and convert it to a structured text summary.

    Args:
        file_bytes: Raw CSV bytes.
        filename:   Original filename (for metadata).

    Returns:
        (structured_text_summary, metadata_summary)
    """
    try:
        import pandas as pd

        df = pd.read_csv(io.BytesIO(file_bytes))
        rows, cols = df.shape

        parts = [
            f"CSV Sensor Data: {filename}",
            f"Dimensions: {rows} rows × {cols} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            "",
            "=== DATA OVERVIEW ===",
        ]

        # Numeric summary
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            parts.append("\n--- Numeric Column Statistics ---")
            desc = df[numeric_cols].describe().round(4)
            parts.append(desc.to_string())

        # First 20 rows as readable table
        parts.append("\n--- Sample Data (first 20 rows) ---")
        sample = df.head(20).to_string(index=True)
        parts.append(sample)

        # Detect anomalies: values > 2 std from mean
        parts.append("\n--- Anomaly Detection (values > 2σ from mean) ---")
        anomaly_found = False
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                outliers = df[df[col].sub(mean_val).abs() > 2 * std_val]
                if not outliers.empty:
                    anomaly_found = True
                    parts.append(
                        f"  {col}: {len(outliers)} outlier(s) detected | "
                        f"Mean={mean_val:.2f}, Std={std_val:.2f}"
                    )
        if not anomaly_found:
            parts.append("  No statistical outliers detected.")

        # Missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if not missing_cols.empty:
            parts.append("\n--- Missing Values ---")
            parts.append(missing_cols.to_string())

        full_text = "\n".join(parts)
        meta = f"CSV: {rows} rows, {cols} columns, {len(numeric_cols)} numeric columns"
        return full_text, meta

    except ImportError:
        return (
            "[pandas not installed. Install with: pip install pandas]",
            "CSV: pandas unavailable"
        )
    except Exception as exc:
        return (
            f"[CSV parsing error: {exc}]",
            f"CSV: error — {exc}"
        )


def parse_image(file_bytes: bytes, filename: str = "image.jpg") -> Tuple[str, str]:
    """
    Process an image file — encode as base64 and generate a descriptive context string.

    Args:
        file_bytes: Raw image bytes.
        filename:   Original filename.

    Returns:
        (image_context_text, metadata_summary)
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(file_bytes))
        width, height = img.size
        mode = img.mode
        fmt = img.format or os.path.splitext(filename)[1].upper().lstrip(".")

        # Encode as base64 for potential multimodal use
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        b64_preview = b64[:100] + "..."  # Truncate for logging

        # Basic image analysis
        img_rgb = img.convert("RGB")
        avg_color = _get_average_color(img_rgb)

        context_text = (
            f"[IMAGE INPUT]\n"
            f"Filename: {filename}\n"
            f"Dimensions: {width} × {height} pixels\n"
            f"Format: {fmt} | Mode: {mode}\n"
            f"Average color (RGB): {avg_color}\n"
            f"Base64 encoded: {b64_preview}\n\n"
            f"The uploaded image shows a physical scene. "
            f"Analyse it based on the image metadata and any visual context "
            f"inferred from filename '{filename}'. "
            f"Apply physical AI reasoning to identify objects, states, risks, "
            f"and anomalies that would be visible in this type of scene."
        )

        meta = f"Image: {filename} | {width}×{height}px | {fmt}"
        return context_text, meta

    except ImportError:
        return (
            "[Pillow not installed. Install with: pip install pillow]",
            "Image: Pillow unavailable"
        )
    except Exception as exc:
        return (
            f"[Image parsing error: {exc}]",
            f"Image: error — {exc}"
        )


def parse_text(text: str) -> Tuple[str, str]:
    """
    Process plain text input — validate and return as-is.

    Returns:
        (cleaned_text, metadata_summary)
    """
    cleaned = text.strip()
    word_count = len(cleaned.split())
    meta = f"Text input: {word_count} words"

    if word_count < 5:
        return (
            f"[Input too short: '{cleaned}'. Please provide a detailed scene description.]",
            "Text: too short"
        )

    return cleaned, meta


def parse_input(
    input_type: str,
    text_content: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    filename: str = "upload",
) -> Tuple[str, str]:
    """
    Unified input parser dispatcher.

    Args:
        input_type:   "text" | "pdf" | "csv" | "image"
        text_content: Text content (for "text" mode)
        file_bytes:   Raw file bytes (for file modes)
        filename:     Original filename

    Returns:
        (extracted_text, metadata_string)
    """
    input_type = input_type.lower().strip()

    if input_type == "pdf":
        if not file_bytes:
            return "[No PDF file provided]", "PDF: no file"
        return parse_pdf(file_bytes, max_pages=30)

    elif input_type == "csv":
        if not file_bytes:
            return "[No CSV file provided]", "CSV: no file"
        return parse_csv(file_bytes, filename)

    elif input_type == "image":
        if not file_bytes:
            return "[No image file provided]", "Image: no file"
        return parse_image(file_bytes, filename)

    elif input_type == "text":
        if not text_content:
            return "[No text provided]", "Text: empty"
        return parse_text(text_content)

    else:
        return (
            f"[Unknown input type: '{input_type}']",
            f"Unknown: {input_type}"
        )


# ------------------------------------------------------------------ #
#  Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _get_average_color(img_rgb) -> str:
    """Compute average RGB color of an image."""
    try:
        import numpy as np
        arr = np.array(img_rgb)
        avg = arr.mean(axis=(0, 1))
        return f"({int(avg[0])}, {int(avg[1])}, {int(avg[2])})"
    except ImportError:
        # numpy not available — sample corner pixels
        try:
            px = img_rgb.getpixel((0, 0))
            return f"({px[0]}, {px[1]}, {px[2]})"
        except Exception:
            return "unknown"
