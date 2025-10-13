#!/usr/bin/env python3
"""
Classical Chinese Document Processor (Qwen-VL + Kimi)
OCR with Qwen-VL, cleaning with Kimi
Outputs single consolidated markdown file with coherent text
"""

import sys
import os
import json
import re
import time
import tempfile
import base64
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

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
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class ProcessingConfig:
    """Centralized configuration for document processing"""

    # OCR Settings
    ocr_dpi: int = 200
    qwen_model: str = "qwen-vl-max"
    max_image_width: int = 1024

    # OCR Quality Thresholds
    expected_chars_normal: int = 500   # Flag if above this
    ocr_soft_cap: int = 800           # Warn if above this
    ocr_hard_cap: int = 1000          # Truncate if above this

    # Text Processing Settings
    max_text_length: int = 2000
    chunk_overlap: int = 400
    api_timeout: int = 300
    retry_attempts: int = 3
    retry_delay: int = 2

    # Model Costs
    model_costs: Dict[str, float] = None

    def __post_init__(self):
        if self.model_costs is None:
            self.model_costs = {
                "qwen-vl-plus": 0.004,
                "qwen-vl-max": 0.008,
            }

    def update_from_args(self, args):
        """Update config from command line arguments"""
        if hasattr(args, 'dpi'):
            self.ocr_dpi = args.dpi
        if hasattr(args, 'model'):
            self.qwen_model = args.model

# ============================================================================
# CLIENT MANAGEMENT
# ============================================================================

class APIClients:
    """Manages API clients for different services"""

    def __init__(self):
        if not self.validate_keys():
            raise Exception("API keys not configured")

        self.kimi = OpenAI(
            api_key=os.getenv("KIMI_API_KEY"),
            base_url="https://api.moonshot.cn/v1"
        )

        self.qwen = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    @staticmethod
    def validate_keys():
        """Validate that required API keys are set"""
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("❌ Error: DASHSCOPE_API_KEY environment variable not set")
            return False
        if not os.getenv("KIMI_API_KEY"):
            print("❌ Error: KIMI_API_KEY environment variable not set")
            return False
        return True

# ============================================================================
# TEXT PROCESSING UTILITIES
# ============================================================================

class TextProcessor:
    """Handles text validation and cleaning operations"""

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
    def remove_metadata_text(text: str) -> str:
        """Remove common institutional metadata from OCR text"""
        patterns = [
            r'国立公文書館', r'National Archives of Japan', r'內閣文庫',
            r'番號\s*漢?\s*\d+', r'冊數\s*\d+', r'号號\s*\d+',
            r'colorchecker', r'Kodak Gray Scale', r'x-rite',
            r'MSCCPPCC\d+', r'MSCCPPPE\d+', r'Kodak, 2007 TM: Kodak',
            r'〔.*?〕',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(keyword in line for keyword in ['綴心部', '文字為開不鮮明']):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @staticmethod
    def remove_explanatory_text(text: str) -> str:
        """Remove any explanatory notes added by AI"""
        patterns = [
            r'---\s*\n\*\*改写说明\*\*.*?(?=---|\Z)',
            r'〔改写说明〕.*?〔/改写说明〕',
            r'〔润色说明〕.*?〔/润色说明〕',
            r'注：.*', r'说明：.*',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)

        # Remove standalone explanatory lines
        lines = text.split('\n')
        cleaned_lines = []
        skip_next = False

        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['改写说明', '润色说明', '调整说明']):
                skip_next = True
                continue
            if skip_next and line.startswith('---'):
                skip_next = False
                continue
            if not skip_next and line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class ImageProcessor:
    """Handles image preprocessing operations"""

    @staticmethod
    def preprocess_image(image_path: str, config: ProcessingConfig) -> str:
        """Improve OCR quality by preprocessing image"""
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
                print(f"    Resized image from {w}x{h} to {config.max_image_width}x{new_h}")

            # Denoise and binarize
            denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Save preprocessed image
            temp_file = tempfile.NamedTemporaryFile(suffix='_preprocessed.png', delete=False)
            cv2.imwrite(temp_file.name, binary)
            return temp_file.name

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return image_path

    @staticmethod
    def truncate_at_loop(text: str, max_length: int = 500) -> str:
        """Truncate text at first major repetition pattern"""
        for i in range(0, min(len(text), max_length), 50):
            sample = text[i:i+100]
            if len(sample) < 100:
                continue
            # If this 100-char chunk appears again, cut before second occurrence
            second_occurrence = text.find(sample, i+100)
            if second_occurrence != -1:
                print(f"    ✂️ Cutting at char {second_occurrence} (loop detected)")
                return text[:second_occurrence]
        return text[:max_length]

# ============================================================================
# PDF PROCESSING
# ============================================================================

class PDFProcessor:
    """Handles PDF conversion and page management"""

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get number of pages in PDF"""
        if not PDF_SUPPORT:
            raise Exception("PDF support not available")
        info = pdfinfo_from_path(pdf_path)
        return info["Pages"]

    @staticmethod
    def convert_to_images_batch(pdf_path: str, config: ProcessingConfig,
                               start_page: int = 1, end_page: Optional[int] = None,
                               batch_size: int = 5) -> List[Tuple[int, str]]:
        """Convert PDF pages to images in batches"""
        if not PDF_SUPPORT:
            raise Exception("PDF support not available")

        info = pdfinfo_from_path(pdf_path)
        total_pages_in_pdf = info["Pages"]
        actual_end_page = end_page if end_page else total_pages_in_pdf
        
        print(f"🔍 PDF has {total_pages_in_pdf} total pages")
        print(f"🔍 Converting pages {start_page} to {actual_end_page}")
        temp_files = []

        for batch_start in range(start_page, actual_end_page + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, actual_end_page)
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=config.ocr_dpi,
                    first_page=batch_start,
                    last_page=batch_end
                )
                
                for i, img in enumerate(images):
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    img.save(temp_file.name, 'PNG')
                    actual_page_num = batch_start + i
                    temp_files.append((actual_page_num, temp_file.name))
            except Exception as e:
                print(f"Error converting batch {batch_start}-{batch_end}: {e}")
                import traceback
                traceback.print_exc()

        return temp_files

# ============================================================================
# OCR ENGINE
# ============================================================================

class OCREngine:
    """Handles OCR operations using Qwen-VL"""

    def __init__(self, clients: APIClients, config: ProcessingConfig):
        self.clients = clients
        self.config = config
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()

    def process_image(self, image_path: str, page_num: int, total_pages: int) -> Optional[Tuple[str, bool]]:
        """Process a single image and return text with loop detection flag"""
        progress = f"({page_num}/{total_pages}, {page_num*100//total_pages}%)" if total_pages > 1 else ""
        print(f"\n--- Page {page_num} {progress} ---")

        try:
            # Preprocess image
            processed_path = self.image_processor.preprocess_image(image_path, self.config)
            temp_preprocessed = processed_path if processed_path != image_path else None

            try:
                # Read and encode image as base64
                with open(processed_path, 'rb') as image_file:
                    image_content = base64.b64encode(image_file.read()).decode('utf-8')

                # Send to Qwen-VL API for OCR
                response = self.clients.qwen.chat.completions.create(
                    model=self.config.qwen_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                   "你是一个古典文献OCR专家。请严格按以下要求提取图片中的全部中文文字:\n"
                                    "1. 版面格式:传统竖排版式,从右到左、从上到下阅读\n"
                                    "2. 每列约1-25字,完整提取每一列\n"
                                    "3. 保留所有标点、空格和换行\n"
                                    "4. 忽略页码、印章、水印、图书馆标记等元数据\n"
                                    "5. 如果遇到模糊或破损文字,用【?】标注\n"
                                    "6. 只提取页面主体文字区域,忽略装订边和页边距\n"
                                    "7. 输出纯文本,不要添加任何说明或标题\n"
                                    "\n请开始提取:"
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

                # Progressive quality checks with our agreed thresholds
                if char_count > self.config.expected_chars_normal:
                    print(f"  ⚠️  Long page: {char_count} chars (expected <{self.config.expected_chars_normal})")

                if char_count > self.config.ocr_soft_cap:
                    print(f"  ⚠️  Very long page: {char_count} chars (soft cap: {self.config.ocr_soft_cap})")

                if char_count > self.config.ocr_hard_cap:
                    print(f"  🔴 Excessive length: {char_count} chars - truncating to {self.config.ocr_hard_cap}")
                    text = text[:self.config.ocr_hard_cap]

                # Detect loops and first pass truncation
                has_loops = self.text_processor.detect_ocr_loops_simple(text)

                if has_loops:
                    print(f"  🔄 Potential OCR loops detected - truncating aggressively")
                    # Find first major repetition and cut there
                    text = ImageProcessor.truncate_at_loop(text, max_length=800)

                print(f"  ✓ OCR success ({len(text)} chars{', has loops' if has_loops else ''})")

                # Remove library metadata
                cleaned_text = self.text_processor.remove_metadata_text(text)

                return cleaned_text, has_loops

            finally:
                # Clean up temporary preprocessed image
                if temp_preprocessed and os.path.exists(temp_preprocessed):
                    try:
                        os.unlink(temp_preprocessed)
                    except:
                        pass

        except Exception as e:
            print(f"  ❌ OCR error: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# TEXT CLEANING ENGINE
# ============================================================================

class TextCleaner:
    """Handles text cleaning using Kimi"""

    def __init__(self, clients: APIClients, config: ProcessingConfig):
        self.clients = clients
        self.config = config
        self.text_processor = TextProcessor()

    def clean_chunk(self, text_chunk: str, context: str = "", has_ocr_loops: bool = False) -> str:
        """Clean a single chunk of text with retry logic"""
        start_time = time.time()

        loop_warning = ""
        if has_ocr_loops:
            loop_warning = "\n\n**重要提示**：此文本可能包含OCR识别循环导致的重复内容，请仔细检查并删除所有重复部分。"

        prompt = f"""请将以下OCR文本整理成连贯的文言文文献。这是明清时期的文集。

要求：
1. 修正明显的OCR识别错误（形近字误认等）
2. 添加适当的标点符号（句号、逗号等）使其可读
3. 保持原文的古汉语特征，不要现代化
4. 删除所有机构元数据（如页码、图书馆标记等）
5. 将文本整理成连贯的段落
6. 对于无法确定的字，用【？】标注
7. **特别注意：如果发现重复出现的文本块（OCR识别循环错误），请删除这些重复部分**
8. **如果文本明显不完整或被截断，保持原样即可**{loop_warning}

{f"上下文提示：{context}" if context else ""}

OCR文本：
{text_chunk}

请直接返回整理后的文本，不要包含任何额外的说明、列表或标题，立即开始正文："""

        for attempt in range(self.config.retry_attempts):
            try:
                response = self.clients.kimi.chat.completions.create(
                    model="kimi-k2-0905-preview",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.config.api_timeout,
                    max_tokens=2048
                )
                processing_time = time.time() - start_time
                print(f"({processing_time:.1f}s)")
                return response.choices[0].message.content

            except Exception as e:
                print(f"✗ Error (attempt {attempt+1}/{self.config.retry_attempts}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("    All retry attempts failed. Returning original text.")
                    return text_chunk

        return text_chunk

    def clean_document_sequential(self, combined_text: str, context: str = "", has_ocr_loops: bool = False) -> str:
        """Clean entire document by processing chunks sequentially"""
        print(f"  [2/3] Cleaning {len(combined_text)} characters sequentially...")

        # Split into chunks
        chunks = self._split_into_chunks(combined_text)
        print(f"  Split into {len(chunks)} chunks for sequential cleaning...")

        # Process chunks
        cleaned_chunks = []
        total_start_time = time.time()

        for i, chunk in enumerate(chunks, 1):
            chunk_start = time.time()
            print(f"    Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)... ", end='')

            try:
                cleaned = self.clean_chunk(chunk, context, has_ocr_loops)
                cleaned = self.text_processor.remove_explanatory_text(cleaned)
                cleaned_chunks.append(cleaned)

                chunk_time = time.time() - chunk_start
                print(f"✓ ({chunk_time:.1f}s)")

                # Progress update
                if i % 10 == 0:
                    elapsed = time.time() - total_start_time
                    avg_time = elapsed / i
                    remaining = (len(chunks) - i) * avg_time
                    eta_mins = remaining / 60
                    print(f"    📊 Progress: {i}/{len(chunks)} ({i*100//len(chunks)}%) - ETA: {eta_mins:.1f} minutes")

                # Rate limiting
                if i < len(chunks):
                    time.sleep(1)

            except Exception as e:
                print(f"✗ Failed: {e}")
                cleaned_chunks.append(chunk)
                time.sleep(2)

        total_time = time.time() - total_start_time
        print(f"  ✓ All {len(chunks)} chunks cleaned in {total_time/60:.1f} minutes!")

        return '\n\n'.join(cleaned_chunks)

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks with overlap"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.max_text_length

            # Try to break at natural boundaries
            if end < len(text):
                for break_point in ['\n\n', '。', '；', '！', '？', '\n']:
                    last_break = text.rfind(break_point, start + self.config.max_text_length//2, end)
                    if last_break != -1:
                        end = last_break + len(break_point)
                        break

            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.config.chunk_overlap if end < len(text) else end

        return chunks

# ============================================================================
# BACKUP MANAGEMENT
# ============================================================================

class BackupManager:
    """Handles OCR backup creation and validation"""

    @staticmethod
    def save_raw_ocr(document_name: str, all_raw_texts: List[str],
                    pages_with_loops: List[int], output_dir: str) -> Path:
        """Save raw OCR results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ocr_backup_path = Path(output_dir) / f"{document_name}_raw_ocr_{timestamp}.json"
        latest_path = Path(output_dir) / f"{document_name}_raw_ocr_latest.json"

        backup_data = {
            "document_name": document_name,
            "total_pages": len(all_raw_texts),
            "total_characters": sum(len(text) for text in all_raw_texts),
            "pages_with_loops": pages_with_loops,
            "timestamp": datetime.now().isoformat(),
            "pages": all_raw_texts
        }

        # Save both versions
        for path in [ocr_backup_path, latest_path]:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Raw OCR backup saved: {ocr_backup_path}")
        return ocr_backup_path

    @staticmethod
    def validate_backup(backup_path: Path) -> bool:
        """Validate OCR backup file"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            required_fields = ["document_name", "pages", "total_pages", "timestamp"]
            missing = [field for field in required_fields if field not in data]

            if missing:
                print(f"❌ Invalid backup file: missing {missing}")
                return False

            if not data["pages"]:
                print("❌ Backup file contains no page data")
                return False

            return True
        except Exception as e:
            print(f"❌ Error validating backup file: {e}")
            return False

# ============================================================================
# DOCUMENT PROCESSOR (Main Orchestrator)
# ============================================================================

class DocumentProcessor:
    """Main document processing orchestrator"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.clients = APIClients()
        self.ocr_engine = OCREngine(self.clients, config)
        self.text_cleaner = TextCleaner(self.clients, config)
        self.pdf_processor = PDFProcessor()
        self.backup_manager = BackupManager()

    def process_pdf(self, pdf_path: str, translate: bool = False, context: str = "",
               output_dir: str = "./processed", quick_mode: bool = False,
               max_pages: Optional[int] = None, start_page: int = 1) -> Optional[Path]:
        """Process a PDF document"""
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_path}")
        if quick_mode:
            print(f"Mode: QUICK (skipping final polish)")
        if max_pages:
            print(f"Limiting to {max_pages} pages starting from page {start_page}")

        try:
            # Get page count
            total_pages = self.pdf_processor.get_page_count(pdf_path)
            
            # Calculate actual page range
            if max_pages:
                # Process max_pages starting from start_page
                end_page = min(start_page + max_pages - 1, total_pages)
            else:
                end_page = total_pages

            page_count = end_page - start_page + 1

            if page_count <= 0:
                print(f"❌ Error: Invalid page range")
                return None

            print(f"Processing pages {start_page}-{end_page} ({page_count} page(s))")
            print(f"{'='*60}")

            start_time = time.time()

            # Step 1: OCR all pages
            print("\n[1/3] Performing OCR on all pages...")
            all_raw_texts = []
            pages_with_loops = []

            # FIXED: Pass end_page instead of max_pages
            temp_files = self.pdf_processor.convert_to_images_batch(
                pdf_path, self.config, start_page=start_page, end_page=end_page, batch_size=5
            )

            if not temp_files:
                print("❌ No images were converted from PDF")
                return None

            for page_num, img_path in temp_files:
                page_start = time.time()
                try:
                    result = self.ocr_engine.process_image(img_path, page_num, page_count)
                    if result:
                        text, has_loops = result
                        if text and text.strip():
                            all_raw_texts.append(text.strip())
                            if has_loops:
                                pages_with_loops.append(page_num)
                            page_time = time.time() - page_start
                            print(f"✓ Page {page_num} OCR complete ({page_time:.1f}s)")
                        else:
                            print(f"⚠️  Page {page_num} had no usable text")
                    else:
                        print(f"⚠️  Page {page_num} processing failed")
                finally:
                    if img_path and os.path.exists(img_path):
                        try:
                            os.unlink(img_path)
                        except:
                            pass

            # Save backup
            os.makedirs(output_dir, exist_ok=True)
            document_name = Path(pdf_path).stem
            if start_page > 1 or end_page < total_pages:
                document_name = f"{document_name}_p{start_page}-{end_page}"

            backup_path = self.backup_manager.save_raw_ocr(
                document_name, all_raw_texts, pages_with_loops, output_dir
            )

            # OCR Summary
            print(f"\n📊 OCR SUMMARY:")
            print(f"  Total pages processed: {len(all_raw_texts)}")
            print(f"  Total characters: {sum(len(text) for text in all_raw_texts)}")
            if pages_with_loops:
                print(f"  🔄 Pages with potential loops: {len(pages_with_loops)}")
                print(f"     {pages_with_loops[:10]}")
                if len(pages_with_loops) > 10:
                    print(f"     ...and {len(pages_with_loops)-10} more")

            if not all_raw_texts:
                print("✗ No pages were successfully processed")
                return None

            # Step 2: Clean document
            print(f"\n[2/3] Combining {len(all_raw_texts)} pages and cleaning...")
            combined_raw = '\n\n'.join(all_raw_texts)

            # Pass loop flag to cleaning
            has_any_loops = len(pages_with_loops) > 0
            cleaned_text = self.text_cleaner.clean_document_sequential(combined_raw, context, has_any_loops)
            final_text = cleaned_text

            # Step 3: Optional translation
            translation = None
            if translate:
                print(f"\n[3/3] Translating entire document...")
                translation = self._translate_document(final_text)

            # Create output
            note_path = self._create_consolidated_note(
                document_name, final_text, translation, pages_with_loops,
                output_dir, context, len(all_raw_texts)
            )

            # Final summary
            successful = len(all_raw_texts)
            total_time = time.time() - start_time
            avg_time = total_time / successful if successful > 0 else 0

            print(f"\n{'='*60}")
            print(f"PDF processing complete: {successful}/{page_count} pages successful")
            print(f"Total time: {total_time:.1f}s ({avg_time:.1f}s per page)")

            estimated_cost = successful * self.config.model_costs.get(self.config.qwen_model, 0.008)
            print(f"Estimated API cost: ${estimated_cost:.3f} USD")
            print(f"{'='*60}")

            return note_path

        except Exception as e:
            print(f"✗ Error processing PDF {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def resume_from_backup(self, backup_path: str, translate: bool = False,
                          context: str = "", output_dir: str = "./processed",
                          quick_mode: bool = False) -> Optional[Path]:
        """Resume processing from saved OCR backup"""
        print(f"\n{'='*60}")
        print(f"Resuming from OCR backup: {backup_path}")

        backup_path = Path(backup_path)
        if not self.backup_manager.validate_backup(backup_path):
            return None

        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            document_name = backup_data["document_name"]
            all_raw_texts = backup_data["pages"]
            pages_with_loops = backup_data.get("pages_with_loops", [])
            total_chars = backup_data['total_characters']

            print(f"Loaded {len(all_raw_texts)} pages from backup")
            print(f"Total characters: {total_chars}")
            print(f"Original document: {document_name}")
            if pages_with_loops:
                print(f"Pages with loops: {len(pages_with_loops)}")

            # Show estimated processing time
            estimated_chunks = (total_chars // self.config.max_text_length) + 1
            print(f"Estimated cleaning time: {estimated_chunks * 30 / 60:.1f} minutes")
            print(f"{'='*60}")

            # Continue with cleaning
            print(f"\n[2/3] Combining {len(all_raw_texts)} pages and cleaning...")
            combined_raw = '\n\n'.join(all_raw_texts)

            # Use loop flag from backup
            has_any_loops = len(pages_with_loops) > 0
            cleaned_text = self.text_cleaner.clean_document_sequential(combined_raw, context, has_any_loops)
            final_text = cleaned_text

            # Handle translation
            translation = None
            if translate:
                print(f"\n[3/3] Translating entire document...")
                translation = self._translate_document(final_text)

            note_path = self._create_consolidated_note(
                document_name, final_text, translation, pages_with_loops,
                output_dir, context, len(all_raw_texts)
            )

            print(f"\n✓ Successfully processed from OCR backup!")
            return note_path

        except Exception as e:
            print(f"✗ Error processing from OCR backup: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_image(self, image_path: str, translate: bool = False, context: str = "",
                 output_dir: str = "./processed", quick_mode: bool = False) -> Optional[Path]:
        """Process a single image file"""
        print(f"\nProcessing image: {image_path}")
        result = self.ocr_engine.process_image(image_path, 1, 1)
        if not result:
            print("✗ No text detected in image")
            return None
        text, has_loops = result

        # Clean the text
        print(f"\n[2/2] Cleaning text...")
        final_text = self.text_cleaner.clean_document_sequential(text, context, has_loops)

        # Create document
        os.makedirs(output_dir, exist_ok=True)
        document_name = Path(image_path).stem
        translation = None
        if translate:
            translation = self._translate_document(final_text)
        pages_with_loops = [1] if has_loops else []
        note_path = self._create_consolidated_note(
            document_name, final_text, translation, pages_with_loops,
            output_dir, context, 1
        )
        return note_path

    def _translate_document(self, text: str) -> Optional[str]:
        """Translate document to English"""
        try:
            response = self.clients.kimi.chat.completions.create(
                model="kimi-k2-0905-preview",
                messages=[{
                    "role": "user",
                    "content": f"请将以下明清文献翻译成学术英语，保持准确性：\n\n{text}"
                }],
                timeout=self.config.api_timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Translation error: {e}")
            return None

    def _create_consolidated_note(self, document_name: str, full_cleaned_text: str,
                                translation: Optional[str], pages_with_loops: List[int],
                                output_dir: str, context: str, page_count: int) -> Path:
        """Create consolidated markdown note"""
        collection = Path(output_dir).name if output_dir != "./processed" else Path(document_name).stem

        # Generate quality section if there were problematic pages
        quality_section = ""
        if pages_with_loops:
            quality_section = f"""
    ## ⚠️ OCR质量提示

    检测到 {len(pages_with_loops)} 页可能包含OCR循环错误，已标记供Kimi清理：

    **需要检查的页面**: {', '.join(map(str, pages_with_loops[:20]))}
    {f"...及其他 {len(pages_with_loops)-20} 页" if len(pages_with_loops) > 20 else ""}

    这些页面的重复内容应该已被自动清理，但建议人工复核。

    ---
    """

        # Build the document
        note_content = f"""---
    type: primary-source
    source: {document_name}
    collection: {collection}
    date-processed: {datetime.now().strftime('%Y-%m-%d')}
    total-pages: {page_count}
    flagged-pages: {len(pages_with_loops)}
    status: {"needs-review" if pages_with_loops else "clean"}
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
    > - **Flagged Pages**: {len(pages_with_loops)}/{page_count}
    > - **Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    > - **OCR Engine**: Qwen-VL ({self.config.qwen_model})
    > - **Context**: {context if context else "None"}
    > - **Status**: {"🟡 Needs Review" if pages_with_loops else "🟢 Clean"}

    ---

    {quality_section}

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
    - Model used: {self.config.qwen_model}
    - Pages with loops: {len(pages_with_loops)}
    """

        # Save file
        output_path = Path(output_dir) / f"{document_name}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(note_content)

        print(f"\n✓ Created consolidated note: {output_path}")
        return output_path

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Process classical Chinese documents with Qwen-VL and Kimi',
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

  # Process pages 50-100 only
  %(prog)s document.pdf --start-page 50 --max-pages 51 --output ./processed

  # Process from page 100 to end
  %(prog)s document.pdf --start-page 100 --output ./processed

  # Limit pages for testing
  %(prog)s document.pdf --max-pages 5 --output ./processed

  # Resume from OCR backup
  %(prog)s --resume-from ./processed/document_raw_ocr_latest.json --output ./processed

  # Batch process directory
  %(prog)s ./scans/ --batch --context "泉州府志" --output ~/Obsidian/Primary-Sources/
"""
    )

    parser.add_argument('input', nargs='?', help='Image file, PDF, or directory (not required with --resume-from)')
    parser.add_argument('--translate', action='store_true', help='Include English translation')
    parser.add_argument('--context', default='', help='Contextual information')
    parser.add_argument('--output', default='./processed', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Process all files in directory')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for PDF conversion')
    parser.add_argument('--model', default='qwen-vl-max', choices=['qwen-vl-plus', 'qwen-vl-max'], help='Qwen model')
    parser.add_argument('--quick', action='store_true', help='Skip final polish step')
    parser.add_argument('--max-pages', type=int, help='Limit processing to first N pages')
    parser.add_argument('--start-page', type=int, default=1, help='Start processing from page N (default: 1)')
    parser.add_argument('--resume-from', help='Resume from raw OCR JSON file')

    return parser

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate API keys
    if not APIClients.validate_keys():
        sys.exit(1)

    # Initialize configuration
    config = ProcessingConfig()
    config.update_from_args(args)

    # Initialize processor
    processor = DocumentProcessor(config)

    # Handle resume case
    if args.resume_from:
        if not Path(args.resume_from).exists():
            print(f"❌ Error: Backup file {args.resume_from} not found")
            sys.exit(1)
        processor.resume_from_backup(
            args.resume_from, args.translate, args.context, args.output, args.quick
        )
        return

    # Validate input
    if not args.input:
        print("❌ Error: Input file or directory required when not using --resume-from")
        sys.exit(1)

    # Batch processing
    if args.batch:
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
            result = processor.process_image(str(img), args.translate, args.context, args.output, args.quick)
            if result:
                successful += 1
            else:
                failed += 1

        for pdf in pdfs:
            result = processor.process_pdf(str(pdf), args.translate, args.context, args.output, args.quick, args.max_pages, args.start_page)
            if result:
                successful += 1
            else:
                failed += 1

        print("=" * 60)
        print(f"\n📊 Processing complete:")
        print(f"   ✓ Successful: {successful}")
        print(f"   ✗ Failed: {failed}")

    else:
        # Single file processing
        if not Path(args.input).exists():
            print(f"❌ Error: File {args.input} not found")
            sys.exit(1)

        file_ext = Path(args.input).suffix.lower()
        if file_ext == '.pdf':
            if not PDF_SUPPORT:
                print("❌ Error: PDF support not available")
                print("   Install with: pip install pdf2image")
                sys.exit(1)
            processor.process_pdf(args.input, args.translate, args.context, args.output, args.quick, args.max_pages, args.start_page)
        else:
            processor.process_image(args.input, args.translate, args.context, args.output, args.quick)

        print("\n✓ Processing complete!")

if __name__ == "__main__":
    main()
