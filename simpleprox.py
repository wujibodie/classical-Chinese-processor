#!/usr/bin/env python3
"""
Modern Classical Chinese PDF to Markdown Converter
A streamlined tool for converting modern editions of classical Chinese texts to markdown.
Revised for memory safety and robust error handling.
"""

import sys
import os
import re
import time
import tempfile
import base64
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Third-party imports
import requests
from openai import OpenAI

# Optional dependencies
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  Warning: pdf2image not installed. PDF support disabled.")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  Warning: opencv-python not installed. Image preprocessing disabled.")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Simplified configuration for modern text processing"""

    # OCR Settings
    ocr_dpi: int = 300  # Higher DPI for modern printed text
    qwen_model: str = "qwen-vl-max"
    max_image_width: int = 1536  # Higher resolution for modern text

    # Text Processing
    api_timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 2

    # Loop detection thresholds
    ocr_soft_cap: int = 3000  # Warning threshold
    ocr_hard_cap: int = 4096  # Absolute maximum

# ============================================================================
# CLIENT MANAGEMENT
# ============================================================================

class APIClient:
    """Manages API client for Qwen-VL"""

    def __init__(self):
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("❌ Error: DASHSCOPE_API_KEY environment variable not set")
            sys.exit(1)

        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class ImageProcessor:
    """Handles image preprocessing for modern text"""

    @staticmethod
    def preprocess_image(image_path: str, config: Config) -> str:
        """Preprocess image for better OCR on modern text"""
        if not CV2_AVAILABLE:
            return image_path

        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize if too large
            h, w = gray.shape
            if w > config.max_image_width:
                ratio = config.max_image_width / w
                new_h = int(h * ratio)
                gray = cv2.resize(gray, (config.max_image_width, new_h), interpolation=cv2.INTER_AREA)

            # Enhance contrast for modern printed text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Save preprocessed image
            temp_file = tempfile.NamedTemporaryFile(suffix='_preprocessed.png', delete=False)
            cv2.imwrite(temp_file.name, enhanced)
            return temp_file.name

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return image_path

# ============================================================================
# TEXT PROCESSING
# ============================================================================

class TextProcessor:
    """Handles text processing and loop detection"""

    @staticmethod
    def detect_ocr_loops_simple(text: str, sample_size: int = 200, max_repeats: int = 3) -> bool:
        """Quick check for obvious OCR repetition artifacts"""
        if len(text) < 1000:  # Only check longer texts
            return False
        # Sample a few spots for repetition
        check_length = min(len(text), 2000)
        for i in range(0, check_length - 100, sample_size):
            sample = text[i:i+100]
            if len(sample) < 100:
                continue
            if text.count(sample) > max_repeats:
                return True
        return False

    @staticmethod
    def truncate_at_loop(text: str, max_length: int = 1500) -> str:
        """
        Log warning if loop detected, but DO NOT delete text.
        Modern editions rarely have fatal OCR loops.
        """
        for i in range(0, min(len(text), max_length), 50):
            sample = text[i:i+100]
            if len(sample) < 100:
                continue
            second_occurrence = text.find(sample, i+100)
            if second_occurrence != -1:
                print(f"    ⚠️  Possible loop detected around char {i}. Preserving full text.")
                # Return full text to be safe
                return text
        return text

# ============================================================================
# PDF PROCESSING
# ============================================================================

class PDFProcessor:
    """Handles PDF conversion to images"""

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get number of pages in PDF"""
        if not PDF_SUPPORT:
            raise Exception("PDF support not available")
        info = pdfinfo_from_path(pdf_path)
        return info["Pages"]

    @staticmethod
    def convert_single_page(pdf_path: str, page_num: int, config: Config) -> Optional[str]:
        """Convert a single PDF page to an image"""
        if not PDF_SUPPORT:
            raise Exception("PDF support not available")

        try:
            images = convert_from_path(
                pdf_path,
                dpi=config.ocr_dpi,
                first_page=page_num,
                last_page=page_num
            )

            if images:
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                images[0].save(temp_file.name, 'PNG')
                return temp_file.name
            return None
        except Exception as e:
            print(f"Error converting page {page_num}: {e}")
            return None

# ============================================================================
# OCR ENGINE
# ============================================================================

class OCREngine:
    """Handles OCR for modern classical Chinese texts"""

    def __init__(self, client: APIClient, config: Config):
        self.client = client
        self.config = config
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()

    def _api_call_with_retry(self, func, *args, **kwargs):
        """Execute API call with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                print(f"API call failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}")
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

    def process_image(self, image_path: str, page_num: int) -> Optional[str]:
        """Process a single image and return extracted text"""
        print(f"Processing page {page_num}...", end=' ', flush=True)

        try:
            # Preprocess image
            processed_path = self.image_processor.preprocess_image(image_path, self.config)
            temp_preprocessed = processed_path if processed_path != image_path else None

            try:
                # Read and encode image
                with open(processed_path, 'rb') as image_file:
                    image_content = base64.b64encode(image_file.read()).decode('utf-8')

                # Send to Qwen-VL API for OCR
                response = self._api_call_with_retry(
                    self.client.client.chat.completions.create,
                    model=self.config.qwen_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "请提取图片中的所有文言文文字。图片格式为竖排、从右至左的现代印刷版本，已有标点符号。要求：\n"
                                    "1. 按照从右至左、从上至下的顺序准确识别所有文字\n"
                                    "2. 保持原文的段落结构和换行\n"
                                    "3. 保留所有标点符号（句读、引号等）\n"
                                    "4. 不要添加任何解释、注释或说明\n"
                                    "5. 不要修改或润色原文\n"
                                    "6. 如果页面没有文字内容，输出'[空页]'\n"
                                    "7. 如果页面主要是图、表、地图、图表等非文字内容，简单描述后输出'[图: 简短描述]'或'[表: 简短描述]'\n"
                                    "8. 如果遇到难以识别的内容，不要重复输出，标注'[难以识别]'即可\n"
                                    "9. 直接输出文字内容，不要添加任何标题或说明\n\n"
                                    "请开始提取："
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_content}"}
                            }
                        ]
                    }],
                    max_tokens=4096,
                    temperature=0.1
                )

                text = response.choices[0].message.content.strip()
                char_count = len(text)

                # Check for excessive length
                if char_count > self.config.ocr_soft_cap:
                    print(f"⚠️  Long page: {char_count} chars", end=' ')

                if char_count > self.config.ocr_hard_cap:
                    print(f"🔴 Truncating to {self.config.ocr_hard_cap}", end=' ')
                    text = text[:self.config.ocr_hard_cap]

                # Detect loops (Logging only, non-destructive)
                has_loops = self.text_processor.detect_ocr_loops_simple(text)
                if has_loops:
                    text = self.text_processor.truncate_at_loop(text, max_length=1500)

                print(f"✓ ({len(text)} chars)")
                return text

            finally:
                # Clean up temporary preprocessed image
                if temp_preprocessed and os.path.exists(temp_preprocessed):
                    try:
                        os.unlink(temp_preprocessed)
                    except:
                        pass

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

# ============================================================================
# ZOTERO INTEGRATION
# ============================================================================

class ZoteroExporter:
    """Handles exporting to Zotero"""

    @staticmethod
    def get_collection_key(collection_name: str) -> Optional[str]:
        """Look up Zotero collection key by name"""
        api_key = os.getenv("ZOTERO_API_KEY")
        user_id = os.getenv("ZOTERO_USER_ID")

        if not api_key or not user_id:
            return None

        try:
            headers = {"Zotero-API-Key": api_key}
            # FIX: Added limit=100 to catch more collections
            response = requests.get(
                f"https://api.zotero.org/users/{user_id}/collections",
                headers=headers,
                params={"limit": 100}
            )

            if response.status_code != 200:
                return None

            collections = response.json()
            for collection in collections:
                if collection['data']['name'] == collection_name:
                    return collection['data']['key']

            print(f"⚠️  Collection '{collection_name}' not found (checked first 100)")
            return None

        except Exception as e:
            print(f"❌ Error looking up collection: {e}")
            return None

    @staticmethod
    def export_to_zotero(markdown_path: Path, pdf_path: str,
                        start_page: int, end_page: int,
                        qwen_model: str,
                        title: Optional[str] = None,
                        collection_key: Optional[str] = None) -> Optional[str]:
        """
        Export processed text to Zotero library with metadata and markdown attachment
        """
        api_key = os.getenv("ZOTERO_API_KEY")
        user_id = os.getenv("ZOTERO_USER_ID")

        if not api_key or not user_id:
            print("⚠️  Warning: ZOTERO_API_KEY or ZOTERO_USER_ID not set. Skipping Zotero export.")
            return None

        try:
            # Prepare metadata
            pdf_name = Path(pdf_path).stem
            display_title = title if title else pdf_name
            process_date = datetime.now().strftime('%Y-%m-%d')

            # Create Zotero item
            item_data = {
                "itemType": "book",
                "title": display_title,
                "abstractNote": f"OCR-processed classical Chinese text from {Path(pdf_path).name}",
                "date": process_date,
                "language": "zh",
                "extra": (
                    f"OCR Engine: Qwen-VL ({qwen_model})\n"
                    f"Source File: {Path(pdf_path).name}\n"
                    f"Pages Processed: {start_page}-{end_page}\n"
                    f"Processing Date: {process_date}"
                )
            }

            # Add to collection if specified
            if collection_key:
                item_data["collections"] = [collection_key]

            # Create item in Zotero
            headers = {
                "Zotero-API-Key": api_key,
                "Content-Type": "application/json"
            }

            response = requests.post(
                f"https://api.zotero.org/users/{user_id}/items",
                headers=headers,
                json=[item_data]
            )

            if response.status_code != 200:
                print(f"❌ Failed to create Zotero item: {response.status_code}")
                print(response.text)
                return None

            # Get the created item key
            item_key = response.json()['successful']['0']['key']
            print(f"✓ Created Zotero item: {item_key}")

            # Attach markdown file
            with open(markdown_path, 'rb') as f:
                file_content = f.read()

            attachment_response = requests.post(
                f"https://api.zotero.org/users/{user_id}/items/{item_key}/file",
                headers={
                    "Zotero-API-Key": api_key,
                    "Content-Type": "text/markdown",
                    "If-None-Match": "*"
                },
                data=file_content
            )

            if attachment_response.status_code == 200:
                print(f"✓ Attached markdown file to Zotero item")
            else:
                print(f"⚠️  Warning: Failed to attach file: {attachment_response.status_code}")

            return item_key

        except Exception as e:
            print(f"❌ Error exporting to Zotero: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class ModernTextProcessor:
    """Main processor for modern classical Chinese texts"""

    def __init__(self):
        self.config = Config()
        self.client = APIClient()
        self.ocr_engine = OCREngine(self.client, self.config)
        self.pdf_processor = PDFProcessor()
        self.zotero = ZoteroExporter()

    def process_pdf(self, pdf_path: str, output_dir: str = "./output",
                   start_page: int = 1, end_page: Optional[int] = None,
                   export_zotero: bool = False,
                   zotero_title: Optional[str] = None,
                   zotero_collection: Optional[str] = None) -> Optional[Path]:
        """Process PDF and create markdown file"""
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        try:
            # Get page count
            total_pages = self.pdf_processor.get_page_count(pdf_path)
            actual_end_page = end_page if end_page else total_pages
            page_count = actual_end_page - start_page + 1

            print(f"Total pages: {total_pages}, processing: {page_count}")
            print("Note: Images are converted and deleted on-the-fly to save disk space.")

            all_texts = []
            start_time = time.time()

            # FIX: Iterate strictly page by page (Convert -> OCR -> Delete)
            for page_num in range(start_page, actual_end_page + 1):
                
                # 1. Convert single page
                img_path = self.pdf_processor.convert_single_page(pdf_path, page_num, self.config)
                
                if not img_path:
                    print(f"⚠️  Skipping page {page_num} (conversion failed)")
                    continue

                try:
                    # 2. OCR single page
                    text = self.ocr_engine.process_image(img_path, page_num)
                    if text:
                        page_marker = f"<!-- Page {page_num} -->\n\n"
                        all_texts.append((page_num, page_marker + text))
                finally:
                    # 3. Immediate cleanup
                    if os.path.exists(img_path):
                        try:
                            os.unlink(img_path)
                        except:
                            pass

            if not all_texts:
                print("❌ No text extracted")
                return None

            # Sort by page number and combine texts
            all_texts.sort(key=lambda x: x[0])
            combined_text = '\n\n'.join([text for _, text in all_texts])

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate output filename
            pdf_name = Path(pdf_path).stem
            if start_page > 1 or end_page:
                pdf_name = f"{pdf_name}_p{start_page}-{actual_end_page}"
            output_path = Path(output_dir) / f"{pdf_name}.md"

            # Create markdown content
            markdown_content = self._create_markdown(pdf_path, combined_text, start_page, actual_end_page)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            processing_time = time.time() - start_time
            print(f"\n✓ Processing complete in {processing_time:.1f} seconds")
            print(f"✓ Output saved to: {output_path}")

            # Export to Zotero if requested
            if export_zotero:
                print(f"\n📚 Exporting to Zotero...")
                collection_key = None
                if zotero_collection:
                    collection_key = self.zotero.get_collection_key(zotero_collection)

                self.zotero.export_to_zotero(
                    output_path,
                    pdf_path,
                    start_page,
                    actual_end_page,
                    self.config.qwen_model,
                    title=zotero_title,
                    collection_key=collection_key
                )

            return output_path

        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_markdown(self, pdf_path: str, text: str, start_page: int, end_page: int) -> str:
        """Create markdown content from extracted text"""
        pdf_name = Path(pdf_path).stem
        current_date = datetime.now().strftime('%Y-%m-%d')

        return f"""---
title: {pdf_name}
source: {Path(pdf_path).name}
processed: {current_date}
pages: {start_page}-{end_page}
---

# {pdf_name}

{text}

---

## Processing Information

- Source: {Path(pdf_path).name}
- Pages: {start_page}-{end_page}
- Processing Date: {current_date}
- OCR Engine: Qwen-VL ({self.config.qwen_model})
"""

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Convert modern classical Chinese PDFs to markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire PDF
  %(prog)s document.pdf

  # Process specific pages
  %(prog)s document.pdf --start-page 10 --end-page 20

  # Specify output directory
  %(prog)s document.pdf --output ./texts/

  # Export to Zotero
  %(prog)s document.pdf --zotero --zotero-collection "Primary Sources"

  # Export to Zotero with custom title
  %(prog)s document.pdf --zotero --zotero-title "明代福建地方志" --zotero-collection "Gazetteers"
"""
    )

    parser.add_argument('pdf', help='PDF file to process')
    parser.add_argument('--output', default='./output', help='Output directory (default: ./output)')
    parser.add_argument('--start-page', type=int, default=1, help='Start page (default: 1)')
    parser.add_argument('--end-page', type=int, help='End page (default: last page)')
    parser.add_argument('--zotero', action='store_true', help='Export to Zotero after processing')
    parser.add_argument('--zotero-title', help='Custom title for Zotero item (defaults to filename)')
    parser.add_argument('--zotero-collection', help='Zotero collection name to file the item in')

    args = parser.parse_args()

    # Validate PDF file
    if not Path(args.pdf).exists():
        print(f"❌ Error: PDF file {args.pdf} not found")
        sys.exit(1)

    if not PDF_SUPPORT:
        print("❌ Error: PDF support not available")
        print("   Install with: pip install pdf2image")
        sys.exit(1)

    # Process PDF
    processor = ModernTextProcessor()
    result = processor.process_pdf(
        args.pdf,
        args.output,
        args.start_page,
        args.end_page,
        export_zotero=args.zotero,
        zotero_title=args.zotero_title,
        zotero_collection=args.zotero_collection
    )

    if result:
        print("\n✓ Success!")
    else:
        print("\n✗ Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
