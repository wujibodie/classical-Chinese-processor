#!/usr/bin/env python3
"""
Classical Chinese Document Processor (Qwen-VL + DeepSeek)
OCR with Qwen-VL, cleaning with DeepSeek
Outputs single consolidated markdown file with coherent text
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import tempfile
import base64
import requests
import re
import time

# For DeepSeek (cleaning/translation)
from openai import OpenAI

# For PDF conversion
try:
    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  Warning: pdf2image not installed. PDF support disabled.")
    print("   Install with: pip install pdf2image")
    print("   Also requires: sudo dnf install poppler-utils")

# For image preprocessing (optional but recommended)
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  Warning: opencv-python not installed. Image preprocessing disabled.")
    print("   Install with: pip install opencv-python")

# Configuration
CONFIG = {
    'ocr_dpi': 200,
    'max_text_length': 1200,  # Reduced from 3000 for better reliability
    'api_timeout': 90,  # Increased from 60
    'chunk_overlap': 300,  # Increased from 200 for better context
    'retry_attempts': 3,
    'retry_delay': 2,
    'qwen_model': "qwen-vl-max",  # Best for classical Chinese
    'max_image_width': 1024,  # Resize large images for faster upload
}

# Model costs for estimation (USD per image)
MODEL_COSTS = {
    "qwen-vl-plus": 0.004,
    "qwen-vl-max": 0.008,
}

# Configure DeepSeek
deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Qwen-VL API endpoint (DashScope)
QWEN_VL_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation"

def get_pdf_page_count(pdf_path):
    """Get number of pages in PDF without loading all pages"""
    if not PDF_SUPPORT:
        raise Exception("PDF support not available")
    info = pdfinfo_from_path(pdf_path)
    return info["Pages"]

def convert_pdf_page_to_image(pdf_path, page_num):
    """Convert single PDF page to image (memory efficient)"""
    if not PDF_SUPPORT:
        raise Exception("PDF support not available. Install pdf2image and poppler-utils")

    try:
        images = convert_from_path(
            pdf_path,
            dpi=CONFIG['ocr_dpi'],
            first_page=page_num,
            last_page=page_num
        )

        if not images:
            return None

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        images[0].save(temp_file.name, 'PNG')
        return temp_file.name

    except Exception as e:
        print(f"Error converting page {page_num}: {e}")
        return None

def preprocess_image(image_path):
    """Optional: improve OCR quality by preprocessing image"""
    if not CV2_AVAILABLE:
        return image_path

    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize if too large (for faster upload and processing)
        h, w = gray.shape
        if w > CONFIG['max_image_width']:
            ratio = CONFIG['max_image_width'] / w
            new_h = int(h * ratio)
            gray = cv2.resize(gray, (CONFIG['max_image_width'], new_h), interpolation=cv2.INTER_AREA)
            print(f"    Resized image from {w}x{h} to {CONFIG['max_image_width']}x{new_h}")

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # Binarize (optional, useful for faded text)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save preprocessed image to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='_preprocessed.png', delete=False)
        cv2.imwrite(temp_file.name, binary)
        return temp_file.name

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return image_path

def ocr_with_qwen_vl(image_path):
    """Stage 1: OCR with Qwen-VL via DashScope API"""
    print(f"  [1/3] Running OCR with Qwen-VL...")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise Exception("DASHSCOPE_API_KEY environment variable not set")

    # Preprocess image (optional but recommended)
    processed_path = preprocess_image(image_path)
    temp_preprocessed = processed_path if processed_path != image_path else None

    try:
        # Read and encode image
        with open(processed_path, 'rb') as image_file:
            image_content = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare request with enhanced prompt
        payload = {
            "model": CONFIG['qwen_model'],
            "input": {
                "prompt": (
                    "你是一个古典文献OCR专家。请严格按以下要求提取图片中的全部中文文字：\n"
                    "1. 按传统书籍排版顺序：从右到左、从上到下逐列读取\n"
                    "2. 保留所有标点、空格和换行，不要合并或删减\n"
                    "3. 忽略页码、印章、水印、图书馆标记等元数据\n"
                    "4. 特别注意：这是明清时期手稿与刻本混合文献\n"
                    "5. 如果遇到模糊或破损文字，用【？】标注\n"
                    "6. 输出纯文本，不要添加任何说明或标题\n"
                    "\n请开始提取："
                ),
                "image": f"data:image/png;base64,{image_content}"
            },
            "parameters": {
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Make API request with retry
        for attempt in range(CONFIG['retry_attempts']):
            try:
                response = requests.post(
                    QWEN_VL_URL,
                    json=payload,
                    headers=headers,
                    timeout=CONFIG['api_timeout']
                )

                if response.status_code == 200:
                    result = response.json()
                    if "output" in result and "text" in result["output"]:
                        text = result["output"]["text"].strip()
                        print(f"  ✓ OCR success (attempt {attempt+1})")
                        return text
                    else:
                        print(f"  ❗ No text found in response (attempt {attempt+1})")
                else:
                    print(f"  ❌ Qwen-VL API error {response.status_code}: {response.text} (attempt {attempt+1})")

            except Exception as e:
                print(f"  ❌ Request failed (attempt {attempt+1}): {e}")

            if attempt < CONFIG['retry_attempts'] - 1:
                time.sleep(CONFIG['retry_delay'])

        return ""

    except Exception as e:
        print(f"  OCR error: {e}")
        return ""
    finally:
        # Clean up preprocessed temp file
        if temp_preprocessed and os.path.exists(temp_preprocessed):
            try:
                os.unlink(temp_preprocessed)
            except:
                pass

def remove_metadata_text(text):
    """Remove common institutional metadata from OCR text"""
    # Remove common institutional text patterns
    patterns = [
        r'国立公文書館',
        r'National Archives of Japan',
        r'內閣文庫',
        r'番號\s*漢?\s*\d+',
        r'冊數\s*\d+',
        r'品號\s*\d+',
        r'colorchecker',
        r'Kodak Gray Scale',
        r'x-rite',
        r'MSCCPPCC\d+',
        r'MSCCPPPE\d+',
        r'Kodak, 2007 TM: Kodak',
        r'【.*?】',  # Remove bracketed text that's likely metadata
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove extra blank lines and clean up
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and not any(keyword in line for keyword in ['綴心部', '文字為開不鮮明']):
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def clean_text_chunk(text_chunk, context=""):
    """Clean a single chunk of text with DeepSeek (with retry and exponential backoff)"""
    prompt = f"""请将以下OCR文本整理成连贯的文言文文献。这是明清时期的文集。

要求：
1. 修正明显的OCR识别错误（形近字误认等）
2. 添加适当的标点符号（句号、逗号等）使其可读
3. 保持原文的古汉语特征，不要现代化
4. 删除所有机构元数据（如页码、图书馆标记等）
5. 将文本整理成连贯的段落
6. 对于无法确定的字，用【？】标注

{f"上下文提示：{context}" if context else ""}

OCR文本：
{text_chunk}

请直接输出整理后的连贯文本，不要添加说明："""

    for attempt in range(CONFIG['retry_attempts']):
        try:
            response = deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                timeout=CONFIG['api_timeout'],
                max_tokens=2048
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"    DeepSeek chunk error (attempt {attempt+1}/{CONFIG['retry_attempts']}): {e}")
            if attempt < CONFIG['retry_attempts'] - 1:
                wait_time = CONFIG['retry_delay'] * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                print(f"    Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("    All retry attempts failed. Returning original text.")
                return text_chunk  # Fallback to original

    return text_chunk

def clean_combined_text(combined_text, context=""):
    """Clean the entire combined text using chunking"""
    print(f"  [2/3] Cleaning {len(combined_text)} characters in chunks...")

    # Split into manageable chunks
    chunk_size = CONFIG['max_text_length']
    overlap = CONFIG['chunk_overlap']

    chunks = []
    start = 0

    while start < len(combined_text):
        end = start + chunk_size

        # If not at end, try to break at paragraph/sentence boundary
        if end < len(combined_text):
            # Look for natural break points
            for break_point in ['\n\n', '。', '；', '！', '？', '\n']:
                last_break = combined_text.rfind(break_point, start + chunk_size//2, end)
                if last_break != -1:
                    end = last_break + len(break_point)
                    break

        chunk = combined_text[start:end]
        chunks.append(chunk)

        # Move start position, with overlap
        start = end - overlap if end < len(combined_text) else end

    print(f"  Split into {len(chunks)} chunks for cleaning...")

    # Clean each chunk
    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"    Cleaning chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        cleaned = clean_text_chunk(chunk, context)
        cleaned_chunks.append(cleaned)

    # Combine cleaned chunks
    full_cleaned = '\n\n'.join(cleaned_chunks)
    print(f"  All chunks cleaned successfully!")

    return full_cleaned

def polish_final_text(cleaned_text, context="", skip_polish=False):
    """Final polishing step to create a coherent, well-formatted document"""
    if skip_polish:
        print(f"  [Final] Skipping final polish (quick mode)")
        return cleaned_text

    print(f"  [Final] Polishing entire document...")

    prompt = f"""请将以下已经初步整理的文言文文本进行最终润色，形成一篇连贯优美的古典文献。这是明清时期的文集。

要求：
1. 添加完整的标点符号（句号、逗号、顿号、引号等）使其流畅可读
2. 确保文意连贯，段落分明
3. 修正任何剩余的OCR错误
4. 删除所有残留的元数据、页码标记和无关符号
5. 保持古典汉语的优雅风格
6. 如果有明显的章节结构，请适当分段

{f"上下文：{context}" if context else ""}

待润色文本：
{cleaned_text}

请输出最终润色后的完整文本："""

    for attempt in range(CONFIG['retry_attempts']):
        try:
            response = deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                timeout=CONFIG['api_timeout'],
                max_tokens=4096
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"  Final polish error (attempt {attempt+1}/{CONFIG['retry_attempts']}): {e}")
            if attempt < CONFIG['retry_attempts'] - 1:
                wait_time = CONFIG['retry_delay'] * (2 ** attempt)
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("  All retry attempts failed. Using chunked version.")
                return cleaned_text  # Fall back to chunked version

    return cleaned_text

def translate_if_needed(text, do_translate=False):
    """Stage 3: Optional translation"""
    if not do_translate:
        return None

    print(f"  [3/3] Translating...")

    try:
        response = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"请将以下明清文献翻译成学术英语，保持准确性：\n\n{text}"
            }],
            timeout=CONFIG['api_timeout']
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Translation error: {e}")
        return None

def process_page(image_path, page_num, total_pages):
    """Process a single page and return raw OCR results"""
    progress = f"({page_num}/{total_pages}, {page_num*100//total_pages}%)" if total_pages > 1 else ""
    print(f"\n--- Page {page_num} {progress} ---")

    try:
        # Stage 1: OCR only
        raw_ocr = ocr_with_qwen_vl(image_path)

        if not raw_ocr.strip():
            print("⚠️  Warning: No text detected")
            return None

        # Remove metadata from raw OCR
        cleaned_raw = remove_metadata_text(raw_ocr)

        return cleaned_raw

    except Exception as e:
        print(f"✗ Error processing page: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_consolidated_note(document_name, full_cleaned_text, translation, output_dir, context="", page_count=0):
    """Create a single consolidated markdown file with coherent text"""

    collection = Path(output_dir).name if output_dir != "./processed" else Path(document_name).stem

    # Build the document
    note_content = f"""---
type: primary-source
source: {document_name}
collection: {collection}
date-processed: {datetime.now().strftime('%Y-%m-%d')}
total-pages: {page_count}
status: needs-review
ocr-engine: Qwen-VL
tags:
  - primary-source
  - {collection}
  - Ming-Qing
---

# {document_name}

> [!info] Document Metadata
> - **Collection**: {collection}
> - **Total Pages**: {page_count}
> - **Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> - **OCR Engine**: Qwen-VL ({CONFIG['qwen_model']})
> - **Context**: {context if context else "None"}
> - **Status**: 🟡 Needs Review

---

## 全文整理

{full_cleaned_text}
"""

    # Add translation if provided
    if translation:
        note_content += f"\n---\n\n## English Translation\n\n{translation}\n"

    # Add research notes section
    note_content += f"""
---

## 研究笔记

### 关键术语
-

### 历史语境
-

### 相关文献
-

---

## 处理历史
- {datetime.now().strftime('%Y-%m-%d')}: OCR (Qwen-VL) and initial processing
- Pages processed: {page_count}
- Model used: {CONFIG['qwen_model']}
"""

    # Save file
    output_path = Path(output_dir) / f"{document_name}.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(note_content)

    print(f"\n✓ Created consolidated note: {output_path}")
    return output_path

def process_pdf(pdf_path, translate=False, context="", output_dir="./processed", quick_mode=False, max_pages=None):
    """Process a PDF by combining all pages first, then cleaning as one document"""
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    if quick_mode:
        print(f"Mode: QUICK (skipping final polish)")
    if max_pages:
        print(f"Limiting to first {max_pages} pages")

    try:
        # Get page count
        page_count = get_pdf_page_count(pdf_path)
        if max_pages:
            page_count = min(page_count, max_pages)
        print(f"Found {page_count} page(s)")
        print(f"{'='*60}")

        # Track timing
        start_time = time.time()

        # Step 1: OCR all pages first
        print("\n[1/3] Performing OCR on all pages...")
        all_raw_texts = []

        for page_num in range(1, page_count + 1):
            page_start = time.time()

            # Convert page
            img_path = None
            try:
                img_path = convert_pdf_page_to_image(pdf_path, page_num)

                if not img_path:
                    print(f"✗ Failed to convert page {page_num}")
                    continue

                # OCR the page
                raw_text = process_page(img_path, page_num, page_count)

                if raw_text and raw_text.strip():
                    all_raw_texts.append(raw_text.strip())
                    page_time = time.time() - page_start
                    print(f"✓ Page {page_num} OCR complete ({page_time:.1f}s)")
                else:
                    print(f"⚠️  Page {page_num} had no usable text")

            finally:
                # Always clean up temp file
                if img_path and os.path.exists(img_path):
                    try:
                        os.unlink(img_path)
                    except:
                        pass

        if not all_raw_texts:
            print("✗ No pages were successfully processed")
            return None

        # Step 2: Combine all pages and clean as one document
        print(f"\n[2/3] Combining {len(all_raw_texts)} pages and cleaning...")
        combined_raw = '\n\n'.join(all_raw_texts)

        # Clean the entire document using chunking
        chunked_text = clean_combined_text(combined_raw, context)

        # Final polishing step for coherence (skip if quick_mode)
        full_cleaned_text = polish_final_text(chunked_text, context, skip_polish=quick_mode)

        # Step 3: Create output directory and document
        os.makedirs(output_dir, exist_ok=True)
        document_name = Path(pdf_path).stem

        # Step 4: Optional translation
        translation = None
        if translate:
            print(f"\n[3/3] Translating entire document...")
            translation = translate_if_needed(full_cleaned_text, True)

        note_path = create_consolidated_note(
            document_name, full_cleaned_text, translation, output_dir, context, len(all_raw_texts)
        )

        successful = len(all_raw_texts)
        total_time = time.time() - start_time
        avg_time = total_time / successful if successful > 0 else 0

        print(f"\n{'='*60}")
        print(f"PDF processing complete: {successful}/{page_count} pages successful")
        print(f"Total time: {total_time:.1f}s ({avg_time:.1f}s per page)")

        # Cost estimation
        estimated_cost = successful * MODEL_COSTS.get(CONFIG['qwen_model'], 0.008)
        print(f"Estimated API cost: ${estimated_cost:.3f} USD")
        print(f"{'='*60}")

        return note_path

    except Exception as e:
        print(f"✗ Error processing PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_image(image_path, translate=False, context="", output_dir="./processed", quick_mode=False):
    """Process a single image file"""
    print(f"\nProcessing image: {image_path}")

    raw_text = process_page(image_path, 1, 1)

    if not raw_text:
        print("✗ No text detected in image")
        return None

    # Clean the text
    print(f"\n[2/3] Cleaning text...")
    cleaned_text = clean_combined_text(raw_text, context)
    polished_text = polish_final_text(cleaned_text, context, skip_polish=quick_mode)

    # Create document
    os.makedirs(output_dir, exist_ok=True)
    document_name = Path(image_path).stem

    translation = None
    if translate:
        translation = translate_if_needed(polished_text, True)

    note_path = create_consolidated_note(
        document_name, polished_text, translation, output_dir, context, 1
    )

    return note_path

def main():
    parser = argparse.ArgumentParser(
        description='Process classical Chinese documents with Qwen-VL and DeepSeek',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  %(prog)s scan.jpg --output ./processed

  # Process PDF with context
  %(prog)s document.pdf --context "明代文集，竖排无标点" --output ~/Obsidian/Primary-Sources/

  # Process with translation
  %(prog)s document.pdf --translate --output ./processed

  # Quick mode (skip final polish)
  %(prog)s document.pdf --quick --output ./processed

  # Use faster/cheaper model
  %(prog)s document.pdf --model qwen-vl-plus --output ./processed

  # Limit pages for testing
  %(prog)s document.pdf --max-pages 5 --output ./processed

  # Batch process directory
  %(prog)s ./scans/ --batch --context "泉州府志" --output ~/Obsidian/Primary-Sources/
        """
    )

    parser.add_argument('input', help='Image file, PDF, or directory')
    parser.add_argument('--translate', action='store_true',
                       help='Include English translation of full document')
    parser.add_argument('--context', default='',
                       help='Contextual information (e.g., "明代文集，竖排无标点")')
    parser.add_argument('--output', default='./processed',
                       help='Output directory for notes')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images/PDFs in directory')
    parser.add_argument('--dpi', type=int, default=200,
                       help='DPI for PDF conversion (default: 200)')
    parser.add_argument('--model', default='qwen-vl-max',
                       choices=['qwen-vl-plus', 'qwen-vl-max'],
                       help='Qwen model (default: qwen-vl-max, best for classical texts)')
    parser.add_argument('--quick', action='store_true',
                       help='Skip final polish step for faster processing')
    parser.add_argument('--max-pages', type=int,
                       help='Limit processing to first N pages (for testing)')

    args = parser.parse_args()

    # Update config with command-line arguments
    CONFIG['ocr_dpi'] = args.dpi
    CONFIG['qwen_model'] = args.model

    # Check API keys
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ Error: DASHSCOPE_API_KEY environment variable not set")
        print("Set it with: export DASHSCOPE_API_KEY='sk-your-key-here'")
        sys.exit(1)

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ Error: DEEPSEEK_API_KEY environment variable not set")
        print("Set it with: export DEEPSEEK_API_KEY='sk-your-key-here'")
        sys.exit(1)

    if args.batch:
        # Batch process directory
        input_path = Path(args.input)
        if not input_path.is_dir():
            print(f"❌ Error: {args.input} is not a directory")
            sys.exit(1)

        images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
        pdfs = list(input_path.glob('*.pdf'))

        if not images and not pdfs:
            print(f"❌ No images or PDFs found in {args.input}")
            sys.exit(1)

        print(f"Found {len(images)} image(s) and {len(pdfs)} PDF(s) to process\n")
        print("=" * 60)

        successful = 0
        failed = 0

        for img in images:
            result = process_image(str(img), args.translate, args.context, args.output, args.quick)
            if result:
                successful += 1
            else:
                failed += 1

        for pdf in pdfs:
            result = process_pdf(str(pdf), args.translate, args.context, args.output, args.quick, args.max_pages)
            if result:
                successful += 1
            else:
                failed += 1

        print("=" * 60)
        print(f"\n📊 Processing complete:")
        print(f"   ✓ Successful: {successful}")
        print(f"   ✗ Failed: {failed}")

    else:
        # Single file
        if not Path(args.input).exists():
            print(f"❌ Error: File {args.input} not found")
            sys.exit(1)

        file_ext = Path(args.input).suffix.lower()
        if file_ext == '.pdf':
            if not PDF_SUPPORT:
                print("❌ Error: PDF support not available")
                print("   Install with: pip install pdf2image")
                print("   Also requires: sudo dnf install poppler-utils")
                sys.exit(1)
            process_pdf(args.input, args.translate, args.context, args.output, args.quick, args.max_pages)
        else:
            process_image(args.input, args.translate, args.context, args.output, args.quick)

        print("\n✓ Processing complete!")

if __name__ == "__main__":
    main()
