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
    ocr_hard_cap: int = 1200          # Truncate if above this

    # Text Processing Settings
    max_text_length: int = 2400
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
            r'\n注：.*', r'\n说明：.*',
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
                headers=headers
            )
            
            if response.status_code != 200:
                return None
            
            collections = response.json()
            for collection in collections:
                if collection['data']['name'] == collection_name:
                    return collection['data']['key']
            
            print(f"⚠️  Collection '{collection_name}' not found in Zotero library")
            return None
            
        except Exception as e:
            print(f"❌ Error looking up collection: {e}")
            return None
    
    @staticmethod
    def export_to_zotero(markdown_path: Path, source_file: str,
                        qwen_model: str, pages_processed: int,
                        title: Optional[str] = None,
                        collection_key: Optional[str] = None) -> Optional[str]:
        """
        Export processed text to Zotero library with metadata and markdown attachment
        
        Returns:
            Zotero item key on success, None on failure
        """
        api_key = os.getenv("ZOTERO_API_KEY")
        user_id = os.getenv("ZOTERO_USER_ID")
        
        if not api_key or not user_id:
            print("⚠️  Warning: ZOTERO_API_KEY or ZOTERO_USER_ID not set. Skipping Zotero export.")
            return None
        
        try:
            # Prepare metadata
            source_name = Path(source_file).stem
            display_title = title if title else source_name
            process_date = datetime.now().strftime('%Y-%m-%d')
            
            # Create Zotero item
            item_data = {
                "itemType": "book",
                "title": display_title,
                "abstractNote": f"OCR-processed classical Chinese text from {Path(source_file).name}",
                "date": process_date,
                "language": "zh",
                "extra": (
                    f"OCR Engine: Qwen-VL ({qwen_model})\n"
                    f"Source File: {Path(source_file).name}\n"
                    f"Pages Processed: {pages_processed}\n"
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
# OCR ENGINE
# ============================================================================

class OCREngine:
    """Handles Qwen-VL OCR operations"""

    def __init__(self, clients: APIClients, config: ProcessingConfig):
        self.clients = clients
        self.config = config
        self.text_processor = TextProcessor()

    def process_image(self, image_path: str, page_num: int, total_pages: int) -> Optional[Tuple[str, bool]]:
        """Process a single image and return text with loop detection flag"""
        progress = f"({page_num}/{total_pages}, {page_num*100//total_pages}%)" if total_pages > 1 else ""
        print(f"\n--- Page {page_num} {progress} ---")

        # Preprocess image
        processed_path = ImageProcessor.preprocess_image(image_path, self.config)
        temp_preprocessed = processed_path if processed_path != image_path else None

        try:
            # Read and encode image as base64
            with open(processed_path, 'rb') as image_file:
                image_content = base64.b64encode(image_file.read()).decode('utf-8')

            # API call with retry logic
            text = None
            for attempt in range(self.config.retry_attempts):
                try:
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
                                        "4. 忽略页码、印章、水印、图书馆标记等元数据。若页面文字内容少于10个字符，或主要区域为图画/图表，输出'[图]。若页面完全无文字内容，或仅有仅有页码/图书馆标记，输出'[空页]'\n"
                                        "5. 如果遇到模糊或破损文字,用【?】标注\n"
                                        "6. 只提取页面主体文字区域,忽略装订边和页边距\n"
                                        "7. 输出纯文本,不要添加任何说明或标题\n"
                                        "8. 特殊格式处理： 如遇表格、名册、账目等行列对齐的版式，请优先保持其行列结构。可使用换行和空格来区分不同条目，确保同一行的数据保持在同一行\n"
                                        "9. 严格保持原文用字：如原文为繁体字，输出一律使用繁体；仅当原文确为简化字时方用简体"
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
                        temperature=0.1,
                        timeout=self.config.api_timeout
                    )
                    
                    text = response.choices[0].message.content.strip()
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < self.config.retry_attempts - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        print(f"  ⚠️  OCR error (attempt {attempt+1}/{self.config.retry_attempts}): {e}")
                        print(f"  Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ All OCR retry attempts failed: {e}")
                        return None

            if text is None:
                return None

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

# ============================================================================
# TEXT CLEANING ENGINE
# ============================================================================

class TextCleaner:
    """Handles Kimi text cleaning operations"""

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

    def create_chunks_with_overlap(self, text: str) -> List[Tuple[str, int, int]]:
        """Create overlapping chunks for processing"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.config.max_text_length, len(text))
            chunk = text[start:end]
            chunks.append((chunk, start, end))
            start = end - self.config.chunk_overlap

        print(f"Created {len(chunks)} chunks with {self.config.chunk_overlap}-char overlap")
        return chunks

    def process_text_in_chunks(self, text: str, context: str = "", has_ocr_loops: bool = False) -> str:
        """Process text in chunks with overlap"""
        if len(text) <= self.config.max_text_length:
            print("Processing single chunk...", end=' ')
            return self.clean_chunk(text, context, has_ocr_loops)

        print(f"\nText length: {len(text)} chars")
        chunks = self.create_chunks_with_overlap(text)
        cleaned_chunks = []

        for i, (chunk, start, end) in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{len(chunks)} (chars {start}-{end})...", end=' ')
            cleaned = self.clean_chunk(chunk, context, has_ocr_loops)
            cleaned_chunks.append(cleaned)

        combined = '\n\n'.join(cleaned_chunks)
        print(f"\n✓ Combined all chunks into {len(combined)} chars")
        return combined

    def final_polish(self, text: str, context: str = "") -> str:
        """Final polish pass on the complete text"""
        print("\n=== Final Polish Pass ===")
        prompt = f"""请对以下已整理的文言文进行最终润色。这是明清时期的文献。

要求：
1. 移除Kimi可能添加的任何说明性文字（如"改写说明"、"润色说明"等）
2. 修正剩余的明显错误
3. 确保标点符号正确且一致
4. 确保段落之间有适当的连贯性
5. 删除任何重复的段落或文本块
6. 保持原文的古汉语特征
7. **不要添加任何新的说明、标题或元数据**
8. **直接输出润色后的正文，不要有任何前言或后记**

{f"上下文提示：{context}" if context else ""}

文本：
{text}

请直接输出润色后的完整文本："""

        for attempt in range(self.config.retry_attempts):
            try:
                response = self.clients.kimi.chat.completions.create(
                    model="kimi-k2-0905-preview",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.config.api_timeout,
                    max_tokens=4096
                )
                polished = response.choices[0].message.content
                print("✓ Final polish complete")
                return self.text_processor.remove_explanatory_text(polished)

            except Exception as e:
                print(f"✗ Polish error (attempt {attempt+1}/{self.config.retry_attempts}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("    All retry attempts failed. Returning unpolished text.")
                    return text

        return text

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Main document processing orchestrator"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.clients = APIClients()
        self.ocr_engine = OCREngine(self.clients, config)
        self.text_cleaner = TextCleaner(self.clients, config)
        self.zotero = ZoteroExporter()

    def process_image(self, image_path: str, context: str = "",
                     output_dir: str = "./processed",
                     quick_mode: bool = False) -> Optional[Path]:
        """Process a single image file"""
        print(f"\n{'='*60}")
        print(f"Processing image: {image_path}")
        print(f"{'='*60}")

        # OCR
        result = self.ocr_engine.process_image(image_path, 1, 1)
        if not result:
            print("❌ OCR failed")
            return None

        raw_text, has_loops = result

        # Save raw OCR
        os.makedirs(output_dir, exist_ok=True)
        doc_name = Path(image_path).stem

        # Clean text
        print("\n=== Text Cleaning Stage ===")
        cleaned_text = self.text_cleaner.process_text_in_chunks(raw_text, context, has_loops)

        # Final polish
        if not quick_mode:
            cleaned_text = self.text_cleaner.final_polish(cleaned_text, context)

        # Create consolidated note
        pages_with_loops = [1] if has_loops else []

        output_path = self._create_consolidated_note(
            doc_name, cleaned_text, pages_with_loops,
            output_dir, context, 1
        )

        print(f"\n✓ Processing complete: {output_path}")
        return output_path

    def process_pdf(self, pdf_path: str, context: str = "",
                   output_dir: str = "./processed",
                   quick_mode: bool = False, max_pages: Optional[int] = None,
                   start_page: int = 1,
                   export_zotero: bool = False,
                   zotero_title: Optional[str] = None,
                   zotero_collection: Optional[str] = None) -> Optional[Path]:
        """Process a PDF file"""
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_path}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        if not PDF_SUPPORT:
            print("❌ PDF support not available")
            return None

        try:
            # Get page info
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]
            end_page = min(start_page + max_pages - 1, total_pages) if max_pages else total_pages
            page_count = end_page - start_page + 1

            print(f"Total PDF pages: {total_pages}")
            print(f"Processing pages {start_page} to {end_page} ({page_count} pages)")
            print(f"DPI: {self.config.ocr_dpi}, Model: {self.config.qwen_model}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            doc_name = Path(pdf_path).stem

            # OCR Stage
            print("\n" + "="*60)
            print("STAGE 1: OCR EXTRACTION")
            print("="*60)

            all_ocr_texts = []
            pages_with_loops = []
            start_time = time.time()

            for page_num in range(start_page, end_page + 1):
                # Convert single page
                images = convert_from_path(
                    pdf_path,
                    dpi=self.config.ocr_dpi,
                    first_page=page_num,
                    last_page=page_num
                )

                if not images:
                    print(f"⚠️  Skipping page {page_num} - conversion failed")
                    continue

                # Save as temporary image
                temp_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                images[0].save(temp_image.name, 'PNG')

                try:
                    # Process with OCR
                    result = self.ocr_engine.process_image(temp_image.name, page_num, page_count)
                    if result:
                        text, has_loops = result
                        all_ocr_texts.append(text)
                        if has_loops:
                            pages_with_loops.append(page_num)
                finally:
                    # Clean up temp image
                    if os.path.exists(temp_image.name):
                        os.unlink(temp_image.name)

            ocr_time = time.time() - start_time

            if not all_ocr_texts:
                print("❌ No text extracted from any pages")
                return None

            # Combine OCR texts
            combined_ocr = '\n\n'.join(all_ocr_texts)
            print(f"\n✓ OCR complete: {len(combined_ocr)} chars from {len(all_ocr_texts)} pages")
            print(f"   Time: {ocr_time:.1f}s")
            print(f"   Pages with loops: {len(pages_with_loops)}")

            # Save OCR backup
            backup_data = {
                "document": doc_name,
                "source": str(pdf_path),
                "pages_processed": page_count,
                "pages_with_loops": pages_with_loops,
                "raw_ocr_text": combined_ocr,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "model": self.config.qwen_model,
                    "dpi": self.config.ocr_dpi,
                }
            }

            backup_path = Path(output_dir) / f"{doc_name}_raw_ocr_latest.json"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            print(f"   Backup saved: {backup_path}")

            # Cleaning Stage
            print("\n" + "="*60)
            print("STAGE 2: TEXT CLEANING")
            print("="*60)

            has_loops = len(pages_with_loops) > 0
            cleaned_text = self.text_cleaner.process_text_in_chunks(
                combined_ocr, context, has_loops
            )

            # Final polish
            if not quick_mode:
                cleaned_text = self.text_cleaner.final_polish(cleaned_text, context)

            # Create consolidated note
            output_path = self._create_consolidated_note(
                doc_name, cleaned_text, pages_with_loops,
                output_dir, context, page_count
            )

            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"✓ Processing complete in {total_time:.1f}s")
            print(f"✓ Output: {output_path}")
            print(f"{'='*60}")

            # Export to Zotero if requested
            if export_zotero:
                print(f"\n📚 Exporting to Zotero...")
                collection_key = None
                if zotero_collection:
                    collection_key = self.zotero.get_collection_key(zotero_collection)
                
                self.zotero.export_to_zotero(
                    output_path,
                    pdf_path,
                    self.config.qwen_model,
                    page_count,
                    title=zotero_title,
                    collection_key=collection_key
                )

            return output_path

        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def resume_from_backup(self, backup_path: str, context: str = "",
                          output_dir: str = "./processed",
                          quick_mode: bool = False,
                          export_zotero: bool = False,
                          zotero_title: Optional[str] = None,
                          zotero_collection: Optional[str] = None):
        """Resume processing from a raw OCR backup file"""
        print(f"\n{'='*60}")
        print(f"Resuming from backup: {backup_path}")
        print(f"{'='*60}")

        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            doc_name = backup_data["document"]
            combined_ocr = backup_data["raw_ocr_text"]
            pages_with_loops = backup_data.get("pages_with_loops", [])
            page_count = backup_data.get("pages_processed", 0)
            saved_context = backup_data.get("context", "")

            # Use provided context or fall back to saved context
            final_context = context if context else saved_context

            print(f"Document: {doc_name}")
            print(f"Text length: {len(combined_ocr)} chars")
            print(f"Pages with loops: {len(pages_with_loops)}")

            # Cleaning Stage
            print("\n" + "="*60)
            print("STAGE 2: TEXT CLEANING (RESUMED)")
            print("="*60)

            has_loops = len(pages_with_loops) > 0
            cleaned_text = self.text_cleaner.process_text_in_chunks(
                combined_ocr, final_context, has_loops
            )

            # Final polish
            if not quick_mode:
                cleaned_text = self.text_cleaner.final_polish(cleaned_text, final_context)

            # Create consolidated note
            output_path = self._create_consolidated_note(
                doc_name, cleaned_text, pages_with_loops,
                output_dir, final_context, page_count
            )

            print(f"\n✓ Resume processing complete: {output_path}")

            # Export to Zotero if requested
            if export_zotero:
                print(f"\n📚 Exporting to Zotero...")
                collection_key = None
                if zotero_collection:
                    collection_key = self.zotero.get_collection_key(zotero_collection)
                
                source_file = backup_data.get("source", doc_name)
                self.zotero.export_to_zotero(
                    output_path,
                    source_file,
                    self.config.qwen_model,
                    page_count,
                    title=zotero_title,
                    collection_key=collection_key
                )

            return output_path

        except Exception as e:
            print(f"❌ Error resuming from backup: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_consolidated_note(self, document_name: str, full_cleaned_text: str,
                                pages_with_loops: List[int],
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

  # Export to Zotero
  %(prog)s document.pdf --zotero --zotero-collection "Primary Sources" --output ./processed

  # Batch process directory
  %(prog)s ./scans/ --batch --context "泉州府志" --output ~/Obsidian/Primary-Sources/
"""
    )

    parser.add_argument('input', nargs='?', help='Image file, PDF, or directory (not required with --resume-from)')
    parser.add_argument('--context', default='', help='Contextual information')
    parser.add_argument('--output', default='./processed', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Process all files in directory')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for PDF conversion')
    parser.add_argument('--model', default='qwen-vl-max', choices=['qwen-vl-plus', 'qwen-vl-max'], help='Qwen model')
    parser.add_argument('--quick', action='store_true', help='Skip final polish step')
    parser.add_argument('--max-pages', type=int, help='Limit processing to first N pages')
    parser.add_argument('--start-page', type=int, default=1, help='Start processing from page N (default: 1)')
    parser.add_argument('--resume-from', help='Resume from raw OCR JSON file')
    parser.add_argument('--zotero', action='store_true', help='Export to Zotero after processing')
    parser.add_argument('--zotero-title', help='Custom title for Zotero item (defaults to filename)')
    parser.add_argument('--zotero-collection', help='Zotero collection name to file the item in')

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
            args.resume_from, args.context, args.output, args.quick,
            export_zotero=args.zotero,
            zotero_title=args.zotero_title,
            zotero_collection=args.zotero_collection
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
            result = processor.process_image(str(img), args.context, args.output, args.quick)
            if result:
                successful += 1
            else:
                failed += 1

        for pdf in pdfs:
            result = processor.process_pdf(
                str(pdf), args.context, args.output, args.quick, 
                args.max_pages, args.start_page,
                export_zotero=args.zotero,
                zotero_title=args.zotero_title,
                zotero_collection=args.zotero_collection
            )
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
            processor.process_pdf(
                args.input, args.context, args.output, args.quick, 
                args.max_pages, args.start_page,
                export_zotero=args.zotero,
                zotero_title=args.zotero_title,
                zotero_collection=args.zotero_collection
            )
        else:
            processor.process_image(args.input, args.context, args.output, args.quick)

        print("\n✓ Processing complete!")

if __name__ == "__main__":
    main()
