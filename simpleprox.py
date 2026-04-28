#!/usr/bin/env python3
"""
Modern Classical Chinese PDF to Markdown Converter
A streamlined tool for converting modern editions of classical Chinese texts to markdown.
Revised for memory safety, robust error handling, and Electron app compatibility.
"""

import sys
import os
import re
import time
import tempfile
import shutil
import base64
import argparse
import concurrent.futures
import threading
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
    print("⚠️  Warning: pdf2image not installed. PDF support disabled.", flush=True)

# Bundled Poppler path (set by Electron app for packaged builds)
_poppler_path = None

try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print(
        "⚠️  Warning: opencv-python not installed. Image preprocessing disabled.",
        flush=True,
    )

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Simplified configuration for modern text processing"""

    # OCR Settings
    ocr_dpi: int = 300  # Optimal DPI for character detail
    qwen_model: str = "qwen3-vl-plus"
    max_image_width: int = 3072  # 300 DPI at A4 width with margins

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
            print(
                "❌ Error: DASHSCOPE_API_KEY environment variable not set", flush=True
            )
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
                gray = cv2.resize(
                    gray, (config.max_image_width, new_h), interpolation=cv2.INTER_AREA
                )

            # Enhance contrast for modern printed text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Save preprocessed image
            temp_file = tempfile.NamedTemporaryFile(
                suffix="_preprocessed.png", delete=False
            )
            cv2.imwrite(temp_file.name, enhanced)
            return temp_file.name

        except Exception as e:
            print(f"Preprocessing error: {e}", flush=True)
            return image_path


# ============================================================================
# TEXT PROCESSING
# ============================================================================


class TextProcessor:
    """Handles text processing and loop detection"""

    @staticmethod
    def detect_ocr_loops_simple(
        text: str, sample_size: int = 200, max_repeats: int = 3
    ) -> bool:
        """Quick check for obvious OCR repetition artifacts"""
        if len(text) < 1000:  # Only check longer texts
            return False
        # Sample a few spots for repetition
        check_length = min(len(text), 2000)
        for i in range(0, check_length - 100, sample_size):
            sample = text[i : i + 100]
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
            sample = text[i : i + 100]
            if len(sample) < 100:
                continue
            second_occurrence = text.find(sample, i + 100)
            if second_occurrence != -1:
                print(
                    f"    ⚠️  Possible loop detected around char {i}. Preserving full text.",
                    flush=True,
                )
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
        info = pdfinfo_from_path(pdf_path, poppler_path=_poppler_path) if _poppler_path else pdfinfo_from_path(pdf_path)
        return info["Pages"]

    @staticmethod
    def convert_single_page(
        pdf_path: str, page_num: int, config: Config
    ) -> Optional[str]:
        """Convert a single PDF page to an image"""
        if not PDF_SUPPORT:
            raise Exception("PDF support not available")

        try:
            _convert_kwargs = dict(dpi=config.ocr_dpi, first_page=page_num, last_page=page_num)
            if _poppler_path:
                _convert_kwargs['poppler_path'] = _poppler_path
            images = convert_from_path(
                pdf_path, **_convert_kwargs
            )

            if images:
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                images[0].save(temp_file.name, "PNG")
                return temp_file.name
            return None
        except Exception as e:
            print(f"Error converting page {page_num}: {e}", flush=True)
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
                print(
                    f"API call failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}",
                    flush=True,
                )
                time.sleep(
                    self.config.retry_delay * (2**attempt)
                )  # Exponential backoff

    def process_image(
        self,
        image_path: str,
        page_num: int,
        total_pages: int = 0,
        context: str = "",
        current_index: int = None,
    ) -> Optional[str]:
        """Process a single image and return extracted text"""
        # Format progress string to match cc-prox.py format for Electron app compatibility
        # Use current_index for progress calculation if provided, otherwise fall back to page_num
        progress_index = current_index if current_index is not None else page_num
        if total_pages > 0:
            pct = progress_index * 100 // total_pages
            print(
                f"\n--- Page {page_num} ({progress_index}/{total_pages}, {pct}%) ---",
                flush=True,
            )
        else:
            print(f"\n--- Page {page_num} ---", flush=True)

        try:
            # Preprocess image
            processed_path = self.image_processor.preprocess_image(
                image_path, self.config
            )
            temp_preprocessed = processed_path if processed_path != image_path else None

            try:
                # Read and encode image
                with open(processed_path, "rb") as image_file:
                    image_content = base64.b64encode(image_file.read()).decode("utf-8")

                # Build prompt - include context if provided
                prompt_text = (
                    "请提取图片中的所有文言文文字。图片格式为竖排、从右至左的现代印刷版本，已有标点符号。要求：\n"
                    "1. 按照从右至左、从上至下的顺序准确识别所有文字\n"
                    "2. **绝对必须保持原文的繁简体字形式**。严禁擅自转换字形\n"
                    "3. 保持原文的段落结构和换行\n"
                    "4. 保留所有标点符号（句读、引号等）\n"
                    "5. 不要添加任何解释、注释或说明\n"
                    "6. 不要修改或润色原文\n"
                    "7. 如果页面没有文字内容，输出'[空页]'\n"
                    "8. 如果页面主要是图、表、地图、图表等非文字内容，简单描述后输出'[图: 简短描述]'或'[表: 简短描述]'\n"
                    "9. 如果遇到难以识别的内容，不要重复输出，标注'[难以识别]'即可\n"
                    "10. 直接输出文字内容，不要添加任何标题或说明\n"
                )

                if context:
                    prompt_text += f"\n上下文提示：{context}\n"

                prompt_text += "\n请开始提取："

                # Send to Qwen-VL API for OCR
                response = self._api_call_with_retry(
                    self.client.client.chat.completions.create,
                    model=self.config.qwen_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_content}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                )

                text = response.choices[0].message.content.strip()
                char_count = len(text)

                # Check for excessive length
                if char_count > self.config.ocr_soft_cap:
                    print(f"  ⚠️  Long page: {char_count} chars", flush=True)

                if char_count > self.config.ocr_hard_cap:
                    print(f"  🔴 Truncating to {self.config.ocr_hard_cap}", flush=True)
                    text = text[: self.config.ocr_hard_cap]

                # Detect loops (Logging only, non-destructive)
                has_loops = self.text_processor.detect_ocr_loops_simple(text)
                if has_loops:
                    text = self.text_processor.truncate_at_loop(text, max_length=1500)

                print(f"  ✓ OCR success ({len(text)} chars)", flush=True)
                return text

            finally:
                # Clean up temporary preprocessed image
                if temp_preprocessed and os.path.exists(temp_preprocessed):
                    try:
                        os.unlink(temp_preprocessed)
                    except Exception:
                        pass

        except Exception as e:
            print(f"  ❌ Error: {e}", flush=True)
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
            response = requests.get(
                f"https://api.zotero.org/users/{user_id}/collections",
                headers=headers,
                params={"limit": 100},
            )

            if response.status_code != 200:
                return None

            try:
                collections = response.json()
                if not isinstance(collections, list):
                    raise ValueError("Expected list of collections")
            except (ValueError, KeyError, AttributeError) as e:
                print(f"❌ Unexpected API response structure: {e}", flush=True)
                return None
            for collection in collections:
                try:
                    if collection["data"]["name"] == collection_name:
                        return collection["data"]["key"]
                except (KeyError, TypeError) as e:
                    print(f"⚠️  Invalid collection structure: {e}", flush=True)
                    continue

            print(
                f"⚠️  Collection '{collection_name}' not found (checked first 100)",
                flush=True,
            )
            return None

        except Exception as e:
            print(f"❌ Error looking up collection: {e}", flush=True)
            return None

    @staticmethod
    def export_to_zotero(
        markdown_path: Path,
        pdf_path: str,
        start_page: int,
        end_page: int,
        qwen_model: str,
        title: Optional[str] = None,
        collection_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export processed text to Zotero library with metadata and markdown attachment
        """
        api_key = os.getenv("ZOTERO_API_KEY")
        user_id = os.getenv("ZOTERO_USER_ID")

        if not api_key or not user_id:
            print(
                "⚠️  Warning: ZOTERO_API_KEY or ZOTERO_USER_ID not set. Skipping Zotero export.",
                flush=True,
            )
            return None

        try:
            # Prepare metadata
            pdf_name = Path(pdf_path).stem
            display_title = title if title else pdf_name
            process_date = datetime.now().strftime("%Y-%m-%d")

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
                ),
            }

            # Add to collection if specified
            if collection_key:
                item_data["collections"] = [collection_key]

            # Create item in Zotero
            headers = {"Zotero-API-Key": api_key, "Content-Type": "application/json"}

            response = requests.post(
                f"https://api.zotero.org/users/{user_id}/items",
                headers=headers,
                json=[item_data],
            )

            if response.status_code != 200:
                print(
                    f"❌ Failed to create Zotero item: {response.status_code}",
                    flush=True,
                )
                print(response.text, flush=True)
                return None

            # Get the created item key
            try:
                response_data = response.json()
                item_key = response_data["successful"]["0"]["key"]
                if not item_key:
                    raise ValueError("Empty item key in response")
            except (KeyError, IndexError, AttributeError, ValueError) as e:
                print(f"❌ Unexpected API response structure: {e}", flush=True)
                return None
            print(f"✓ Created Zotero item: {item_key}", flush=True)

            # Attach markdown file
            with open(markdown_path, "rb") as f:
                file_content = f.read()

            attachment_response = requests.post(
                f"https://api.zotero.org/users/{user_id}/items/{item_key}/file",
                headers={
                    "Zotero-API-Key": api_key,
                    "Content-Type": "text/markdown",
                    "If-None-Match": "*",
                },
                data=file_content,
            )

            if attachment_response.status_code == 200:
                print(f"✓ Attached markdown file to Zotero item", flush=True)
            else:
                print(
                    f"⚠️  Warning: Failed to attach file: {attachment_response.status_code}",
                    flush=True,
                )

            return item_key

        except Exception as e:
            print(f"❌ Error exporting to Zotero: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return None


# ============================================================================
# MAIN PROCESSOR
# ============================================================================


class ModernTextProcessor:
    """Main processor for modern classical Chinese texts"""

    # Default OCR concurrency settings
    DEFAULT_OCR_WORKERS = 15  # Conservative default to avoid memory issues

    def __init__(self, model: str = "qwen3-vl-plus"):
        self.config = Config()
        self.config.qwen_model = model
        self.client = APIClient()
        self.ocr_engine = OCREngine(self.client, self.config)
        self.pdf_processor = PDFProcessor()
        self.zotero = ZoteroExporter()

        # Parallel OCR settings
        self.parallel_ocr = True  # Can be disabled via --sequential flag
        self.max_ocr_workers = self.DEFAULT_OCR_WORKERS

        # Thread-safe progress tracking for parallel OCR
        self._ocr_progress_lock = threading.Lock()
        self._ocr_completed_count = 0

    def _extract_all_images(
        self, pdf_path: str, start_page: int, end_page: int, temp_dir: str
    ) -> List[Tuple[int, str]]:
        """
        Phase 1: Extract all PDF pages to images sequentially.
        Returns list of (page_num, image_path) tuples.
        """
        extracted = []
        total_pages = end_page - start_page + 1

        print(f"\n📄 Extracting {total_pages} pages to temporary images...", flush=True)

        for page_num in range(start_page, end_page + 1):
            img_path = self.pdf_processor.convert_single_page(
                pdf_path, page_num, self.config
            )
            if img_path:
                # Move to temp_dir with consistent naming
                new_path = os.path.join(temp_dir, f"page_{page_num:04d}.png")
                shutil.move(img_path, new_path)
                extracted.append((page_num, new_path))
            else:
                print(f"  ⚠️  Page {page_num}: conversion failed", flush=True)

        print(f"  ✓ Extracted {len(extracted)}/{total_pages} pages", flush=True)
        return extracted

    def _ocr_page_worker(
        self, page_num: int, image_path: str, total_pages: int, context: str
    ) -> Tuple[int, Optional[str]]:
        """
        Worker function for parallel OCR. Processes a single page.
        Returns: (page_num, text)
        """
        try:
            # Process the image
            text = self.ocr_engine.process_image(
                image_path, page_num, total_pages, context, current_index=page_num
            )

            # Update progress (thread-safe)
            with self._ocr_progress_lock:
                self._ocr_completed_count += 1
                completed = self._ocr_completed_count
            print(f"  📊 OCR progress: {completed}/{total_pages} pages", flush=True)

            # Web service progress callback
            if hasattr(self, "_web_progress_callback") and self._web_progress_callback:
                percent = 10 + int(
                    (completed / total_pages) * 85
                )  # OCR is 10-95% (no cleanup stage)
                self._web_progress_callback("ocr", page_num, percent)

            if text:
                page_marker = f"<!-- Page {page_num} -->\n\n"
                return (page_num, page_marker + text)
            else:
                return (page_num, None)

        except Exception as e:
            print(f"  ❌ Page {page_num} OCR error: {e}", flush=True)
            return (page_num, None)
        finally:
            # Clean up the temp image immediately after processing
            try:
                if os.path.exists(image_path):
                    os.unlink(image_path)
            except Exception:
                pass

    def _parallel_ocr(
        self, extracted_pages: List[Tuple[int, str]], total_pages: int, context: str
    ) -> List[Tuple[int, str]]:
        """
        Phase 2: Process extracted images with parallel OCR.
        Returns: list of (page_num, text) tuples
        """
        # Reset progress counter
        self._ocr_completed_count = 0

        # Determine worker count
        num_workers = min(self.max_ocr_workers, len(extracted_pages))
        print(f"\n🚀 Starting parallel OCR with {num_workers} workers...", flush=True)

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(
                    self._ocr_page_worker, page_num, image_path, total_pages, context
                ): page_num
                for page_num, image_path in extracted_pages
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result_page_num, text = future.result()
                    if text:
                        results.append((result_page_num, text))
                except Exception as e:
                    print(f"  ❌ Page {page_num} failed: {e}", flush=True)

        # Sort by page number
        results.sort(key=lambda x: x[0])
        print(f"\n🔗 Assembled {len(results)} pages in order", flush=True)

        return results

    def _sequential_ocr(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
        page_count: int,
        context: str,
    ) -> List[Tuple[int, str]]:
        """
        Original sequential OCR flow (convert-OCR-delete one page at a time).
        """
        all_texts = []

        current_page_index = 0
        for page_num in range(start_page, end_page + 1):
            current_page_index += 1

            # Convert single page
            img_path = self.pdf_processor.convert_single_page(
                pdf_path, page_num, self.config
            )

            if not img_path:
                print(f"⚠️  Skipping page {page_num} (conversion failed)", flush=True)
                continue

            try:
                # OCR single page
                text = self.ocr_engine.process_image(
                    img_path, page_num, page_count, context, current_page_index
                )
                if text:
                    page_marker = f"<!-- Page {page_num} -->\n\n"
                    all_texts.append((page_num, page_marker + text))

                # Web service progress callback
                if hasattr(self, "_web_progress_callback") and self._web_progress_callback:
                    percent = 10 + int(
                        (current_page_index / page_count) * 85
                    )  # OCR is 10-95%
                    self._web_progress_callback("ocr", page_num, percent)
            finally:
                # Immediate cleanup
                if os.path.exists(img_path):
                    try:
                        os.unlink(img_path)
                    except Exception:
                        pass

        return all_texts

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str = "./output",
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        context: str = "",
        review_mode: bool = False,  # Accepted but not used (simple mode has no review)
        export_zotero: bool = False,
        zotero_title: Optional[str] = None,
        zotero_collection: Optional[str] = None,
    ) -> Optional[Path]:
        """Process PDF and create markdown file"""
        print(f"\n{'=' * 60}", flush=True)
        print(f"Processing PDF: {pdf_path}", flush=True)
        print(f"Output directory: {output_dir}", flush=True)
        print(f"Model: {self.config.qwen_model}", flush=True)
        if context:
            print(f"Context: {context}", flush=True)
        print(f"{'=' * 60}", flush=True)

        try:
            # Get page count
            total_pages = self.pdf_processor.get_page_count(pdf_path)

            # Calculate actual end page from various inputs
            # Priority: end_page > max_pages > total_pages
            if end_page:
                actual_end_page = min(end_page, total_pages)
            elif max_pages:
                actual_end_page = min(start_page + max_pages - 1, total_pages)
            else:
                actual_end_page = total_pages

            page_count = actual_end_page - start_page + 1

            print(f"Total pages in PDF: {total_pages}", flush=True)
            print(
                f"Processing pages {start_page} to {actual_end_page} ({page_count} pages)",
                flush=True,
            )

            start_time = time.time()

            # Choose parallel or sequential based on settings and page count
            use_parallel = self.parallel_ocr and page_count > 3

            if use_parallel:
                print(f"Mode: PARALLEL ({self.max_ocr_workers} workers)", flush=True)

                # Create temp directory for images
                temp_dir = tempfile.mkdtemp(prefix="zhiliu_simple_ocr_")

                try:
                    # Phase 1: Extract all images sequentially
                    extracted_pages = self._extract_all_images(
                        pdf_path, start_page, actual_end_page, temp_dir
                    )

                    if not extracted_pages:
                        print("❌ No pages extracted", flush=True)
                        return None

                    # Phase 2 & 3: Parallel OCR and assembly
                    all_texts = self._parallel_ocr(extracted_pages, page_count, context)

                finally:
                    # Clean up temp directory (any remaining files)
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass

            else:
                print(f"Mode: SEQUENTIAL", flush=True)
                all_texts = self._sequential_ocr(
                    pdf_path, start_page, actual_end_page, page_count, context
                )

            if not all_texts:
                print("❌ No text extracted", flush=True)
                return None

            # Combine texts (already sorted)
            combined_text = "\n\n".join([text for _, text in all_texts])

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate output filename
            pdf_name = Path(pdf_path).stem
            if start_page > 1 or (end_page or max_pages):
                pdf_name = f"{pdf_name}_p{start_page}-{actual_end_page}"
            output_path = Path(output_dir) / f"{pdf_name}.md"

            # Create markdown content
            original_filename = getattr(self, "_original_filename", None)
            markdown_content = self._create_markdown(
                pdf_path, combined_text, start_page, actual_end_page, context,
                original_filename=original_filename,
            )

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            processing_time = time.time() - start_time
            print(
                f"\n✓ Processing complete in {processing_time:.1f} seconds", flush=True
            )
            print(f"✓ Output saved to: {output_path}", flush=True)

            # Export to Zotero if requested
            if export_zotero:
                print(f"\n📚 Exporting to Zotero...", flush=True)
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
                    collection_key=collection_key,
                )

            return output_path

        except Exception as e:
            print(f"❌ Error processing PDF: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return None

    def _create_markdown(
        self,
        pdf_path: str,
        text: str,
        start_page: int,
        end_page: int,
        context: str = "",
        original_filename: str = None,
    ) -> str:
        """Create markdown content from extracted text"""
        pdf_name = (
            Path(original_filename).stem
            if original_filename
            else Path(pdf_path).stem
        )
        source_name = (
            original_filename if original_filename else Path(pdf_path).name
        )
        current_date = datetime.now().strftime("%Y-%m-%d")

        return f"""---
title: {pdf_name}
source: {source_name}
processed: {current_date}
pages: {start_page}-{end_page}
ocr-engine: Qwen-VL ({self.config.qwen_model})
context: {context if context else "None"}
---

# {pdf_name}

{text}

---

## Processing Information

- Source: {source_name}
- Pages: {start_page}-{end_page}
- Processing Date: {current_date}
- OCR Engine: Qwen-VL ({self.config.qwen_model})
- Context: {context if context else "None"}
"""


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert modern classical Chinese PDFs to markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire PDF
  %(prog)s document.pdf

  # Process specific pages
  %(prog)s document.pdf --start-page 10 --end-page 20

  # Process first 5 pages only
  %(prog)s document.pdf --max-pages 5

  # Specify output directory
  %(prog)s document.pdf --output ./texts/

  # Use higher quality alternative model
  %(prog)s document.pdf --model qwen-vl-max

  # Add context for better OCR
  %(prog)s document.pdf --context "明代福建地方志"

  # Export to Zotero
  %(prog)s document.pdf --zotero --zotero-collection "Primary Sources"

  # Export to Zotero with custom title
  %(prog)s document.pdf --zotero --zotero-title "明代福建地方志" --zotero-collection "Gazetteers"
""",
    )

    parser.add_argument("pdf", help="PDF file to process")
    parser.add_argument(
        "--output", default="./output", help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--start-page", type=int, default=1, help="Start page (default: 1)"
    )
    parser.add_argument("--end-page", type=int, help="End page (default: last page)")
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to process (alternative to --end-page)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-plus",
        choices=["qwen3-vl-plus", "qwen-vl-max"],
        help="Qwen model to use (default: qwen3-vl-plus)",
    )
    parser.add_argument(
        "--context",
        default="",
        help='Context information to help OCR (e.g., "明代地方志")',
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Review mode (accepted for compatibility, no effect in simple mode)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential processing (disables parallel OCR)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent OCR requests (default: 15)",
    )
    parser.add_argument(
        "--zotero", action="store_true", help="Export to Zotero after processing"
    )
    parser.add_argument(
        "--zotero-title", help="Custom title for Zotero item (defaults to filename)"
    )
    parser.add_argument(
        "--zotero-collection", help="Zotero collection name to file the item in"
    )
    parser.add_argument(
        "--poppler-path", default=None,
        help="Path to Poppler binaries directory (for bundled Poppler)"
    )

    args = parser.parse_args()

    # Validate PDF file
    if not Path(args.pdf).exists():
        print(f"❌ Error: PDF file {args.pdf} not found", flush=True)
        sys.exit(1)

    if not PDF_SUPPORT:
        print("❌ Error: PDF support not available", flush=True)
        print("   Install with: pip install pdf2image", flush=True)
        sys.exit(1)

    # Set bundled Poppler path if provided
    global _poppler_path
    _poppler_path = args.poppler_path
    if _poppler_path:
        print(f"⚙️  Using bundled Poppler from: {_poppler_path}", flush=True)

    # Process PDF
    processor = ModernTextProcessor(model=args.model)

    # Handle sequential mode and concurrency settings
    if args.sequential:
        print("⚠️  Sequential mode forced (--sequential flag)", flush=True)
        processor.parallel_ocr = False
    elif args.max_concurrent:
        processor.max_ocr_workers = args.max_concurrent
        print(f"⚙️  Concurrency limit set to {args.max_concurrent} workers", flush=True)

    result = processor.process_pdf(
        args.pdf,
        args.output,
        args.start_page,
        args.end_page,
        max_pages=args.max_pages,
        context=args.context,
        review_mode=args.review,
        export_zotero=args.zotero,
        zotero_title=args.zotero_title,
        zotero_collection=args.zotero_collection,
    )

    if result:
        print("\n✓ Success!", flush=True)
    else:
        print("\n✗ Processing failed", flush=True)
        sys.exit(1)


def process_pdf_simple(
    input_path: str,
    output_dir: str,
    start_page: int = 1,
    end_page: int = None,
    progress_callback=None,
    original_filename: str = None,
) -> str:
    """
    Web service wrapper for simple OCR processing (no cleanup stage).

    Args:
        input_path: Path to input PDF file
        output_dir: Directory to save output files
        start_page: First page to process (1-indexed)
        end_page: Last page to process (None = all pages)
        progress_callback: Called as progress_callback(stage, page, percent)
                          Stages: "preprocessing", "ocr", "postprocessing"
        original_filename: Original upload filename (used for markdown title)

    Returns:
        Path to output markdown file

    Raises:
        Exception on processing failure
    """
    processor = ModernTextProcessor()
    processor._web_progress_callback = progress_callback
    processor._original_filename = original_filename
    output_path = processor.process_pdf(
        pdf_path=input_path,
        output_dir=output_dir,
        start_page=start_page,
        end_page=end_page,
    )
    if output_path is None:
        raise Exception("OCR processing failed - no output generated")
    return str(output_path)


if __name__ == "__main__":
    main()
