#!/usr/bin/env python3
"""
Classical Chinese Document Processor (Qwen-VL + DeepSeek)
OCR with Qwen-VL, cleaning with DeepSeek
Outputs single consolidated markdown file with coherent text
"""

import sys
import os
import json
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
    print("⚠️  Warning: pdf2image not installed. PDF support disabled.", flush=True)

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
# CONFIGURATION MANAGEMENT
# ============================================================================


@dataclass
class ProcessingConfig:
    """Centralized configuration for document processing"""

    # OCR Settings
    ocr_dpi: int = 300
    qwen_model: str = "qwen3-vl-plus"
    max_image_width: int = 3072

    # Cleanup Settings
    cleanup_model: str = "deepseek"  # "kimi" or "deepseek"

    # OCR Quality Thresholds
    expected_chars_normal: int = 500  # Flag if above this
    ocr_soft_cap: int = 800  # Warn if above this
    ocr_hard_cap: int = 1200  # Truncate if above this

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
                "qwen3-vl-plus": 0.004,
                "qwen-vl-max": 0.008,
            }

    def update_from_args(self, args):
        """Update config from command line arguments"""
        if hasattr(args, "dpi"):
            self.ocr_dpi = args.dpi
        if hasattr(args, "model"):
            self.qwen_model = args.model
        if hasattr(args, "cleanup_model"):
            self.cleanup_model = args.cleanup_model


# ============================================================================
# CLIENT MANAGEMENT
# ============================================================================


class APIClients:
    """Manages API clients for different services"""

    def __init__(self, cleanup_model: str = "deepseek"):
        if not self.validate_keys(cleanup_model):
            raise Exception("API keys not configured")

        self.kimi = None
        self.deepseek = None

        # Initialize cleanup model client
        if cleanup_model == "deepseek":
            self.deepseek = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
        elif cleanup_model == "kimi":
            self.kimi = OpenAI(
                api_key=os.getenv("KIMI_API_KEY"), base_url="https://api.moonshot.cn/v1"
            )

        self.qwen = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    @staticmethod
    def validate_keys(cleanup_model: str = "deepseek"):
        """Validate that required API keys are set"""
        if not os.getenv("DASHSCOPE_API_KEY"):
            print(
                "❌ Error: DASHSCOPE_API_KEY environment variable not set", flush=True
            )
            return False

        if cleanup_model == "kimi" and not os.getenv("KIMI_API_KEY"):
            print("❌ Error: KIMI_API_KEY environment variable not set", flush=True)
            return False

        if cleanup_model == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
            print("❌ Error: DEEPSEEK_API_KEY environment variable not set", flush=True)
            return False

        return True


# ============================================================================
# TEXT PROCESSING UTILITIES
# ============================================================================


class TextProcessor:
    """Handles text validation and cleaning operations"""

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
    def remove_metadata_text(text: str) -> str:
        """Remove common institutional metadata from OCR text"""
        patterns = [
            r"国立公文書館",
            r"National Archives of Japan",
            r"內閣文庫",
            r"番號\s*漢?\s*\d+",
            r"冊數\s*\d+",
            r"号號\s*\d+",
            r"colorchecker",
            r"Kodak Gray Scale",
            r"x-rite",
            r"MSCCPPCC\d+",
            r"MSCCPPPE\d+",
            r"Kodak, 2007 TM: Kodak",
            r"〔.*?〕",
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up lines
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(
                keyword in line for keyword in ["綴心部", "文字為開不鮮明"]
            ):
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def remove_explanatory_text(text: str) -> str:
        """Remove any explanatory notes added by AI"""
        patterns = [
            r"---\s*\n\*\*改写说明\*\*.*?(?=---|\Z)",
            r"〔改写说明〕.*?〔/改写说明〕",
            r"〔润色说明〕.*?〔/润色说明〕",
            r"\n注：.*",
            r"\n说明：.*",
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL)

        # Remove standalone explanatory lines
        lines = text.split("\n")
        cleaned_lines = []
        skip_next = False

        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ["改写说明", "润色说明", "调整说明"]):
                skip_next = True
                continue
            if skip_next and line.startswith("---"):
                skip_next = False
                continue
            if not skip_next and line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)


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
                gray = cv2.resize(
                    gray, (config.max_image_width, new_h), interpolation=cv2.INTER_AREA
                )
                print(
                    f"    Resized image from {w}x{h} to {config.max_image_width}x{new_h}",
                    flush=True,
                )

            # Denoise and binarize
            denoised = cv2.fastNlMeansDenoising(
                gray, h=10, templateWindowSize=7, searchWindowSize=21
            )
            _, binary = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Save preprocessed image
            temp_file = tempfile.NamedTemporaryFile(
                suffix="_preprocessed.png", delete=False
            )
            cv2.imwrite(temp_file.name, binary)
            return temp_file.name

        except Exception as e:
            print(f"Preprocessing error: {e}", flush=True)
            return image_path

    @staticmethod
    def truncate_at_loop(text: str, max_length: int = 500) -> str:
        """Truncate text at first major repetition pattern"""
        for i in range(0, min(len(text), max_length), 50):
            sample = text[i : i + 100]
            if len(sample) < 100:
                continue
            # If this 100-char chunk appears again, cut before second occurrence
            second_occurrence = text.find(sample, i + 100)
            if second_occurrence != -1:
                print(
                    f"    ✂️ Cutting at char {second_occurrence} (loop detected)",
                    flush=True,
                )
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
                headers=headers,
                params={"limit": 100},
            )

            if response.status_code != 200:
                return None

            collections = response.json()
            for collection in collections:
                if collection["data"]["name"] == collection_name:
                    return collection["data"]["key"]

            print(
                f"⚠️  Collection '{collection_name}' not found in Zotero library",
                flush=True,
            )
            return None

        except Exception as e:
            print(f"❌ Error looking up collection: {e}", flush=True)
            return None

    @staticmethod
    def export_to_zotero(
        markdown_path: Path,
        source_file: str,
        qwen_model: str,
        pages_processed: int,
        title: Optional[str] = None,
        collection_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export processed text to Zotero library with metadata and markdown attachment

        Returns:
            Zotero item key on success, None on failure
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
            source_name = Path(source_file).stem
            display_title = title if title else source_name
            process_date = datetime.now().strftime("%Y-%m-%d")

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
            item_key = response.json()["successful"]["0"]["key"]
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
# OCR ENGINE
# ============================================================================


class OCREngine:
    """Handles Qwen-VL OCR operations"""

    def __init__(self, clients: APIClients, config: ProcessingConfig):
        self.clients = clients
        self.config = config
        self.text_processor = TextProcessor()

    def process_image(
        self,
        image_path: str,
        page_num: int,
        total_pages: int,
        current_index: int = None,
    ) -> Optional[Tuple[str, bool]]:
        """Process a single image and return text with loop detection flag"""
        # Use current_index for progress calculation if provided, otherwise fall back to page_num
        progress_index = current_index if current_index is not None else page_num
        progress = (
            f"({progress_index}/{total_pages}, {progress_index * 100 // total_pages}%)"
            if total_pages > 1
            else ""
        )
        print(f"\n--- Page {page_num} {progress} ---", flush=True)

        # Preprocess image
        processed_path = ImageProcessor.preprocess_image(image_path, self.config)
        temp_preprocessed = processed_path if processed_path != image_path else None

        try:
            # Read and encode image as base64
            with open(processed_path, "rb") as image_file:
                image_content = base64.b64encode(image_file.read()).decode("utf-8")

            # API call with retry logic
            text = None
            for attempt in range(self.config.retry_attempts):
                try:
                    response = self.clients.qwen.chat.completions.create(
                        model=self.config.qwen_model,
                        messages=[
                            {
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
                                        ),
                                    },
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
                        timeout=self.config.api_timeout,
                    )

                    try:
                        content = response.choices[0].message.content
                        if not content:
                            raise ValueError("Empty response content")
                        text = content.strip()
                    except (IndexError, AttributeError, KeyError) as e:
                        raise ValueError(f"Unexpected API response structure: {e}")
                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < self.config.retry_attempts - 1:
                        wait_time = self.config.retry_delay * (2**attempt)
                        print(
                            f"  ⚠️  OCR error (attempt {attempt + 1}/{self.config.retry_attempts}): {e}",
                            flush=True,
                        )
                        print(f"  Retrying in {wait_time}s...", flush=True)
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ All OCR retry attempts failed: {e}", flush=True)
                        return None

            if text is None:
                return None

            char_count = len(text)

            # Progressive quality checks with our agreed thresholds
            if char_count > self.config.expected_chars_normal:
                print(
                    f"  ⚠️  Long page: {char_count} chars (expected <{self.config.expected_chars_normal})",
                    flush=True,
                )

            if char_count > self.config.ocr_soft_cap:
                print(
                    f"  ⚠️  Very long page: {char_count} chars (soft cap: {self.config.ocr_soft_cap})",
                    flush=True,
                )

            if char_count > self.config.ocr_hard_cap:
                print(
                    f"  🔴 Excessive length: {char_count} chars - truncating to {self.config.ocr_hard_cap}",
                    flush=True,
                )
                text = text[: self.config.ocr_hard_cap]

            # Detect loops and first pass truncation
            has_loops = self.text_processor.detect_ocr_loops_simple(text)

            if has_loops:
                print(
                    f"  🔄 Potential OCR loops detected - truncating aggressively",
                    flush=True,
                )
                # Find first major repetition and cut there
                text = ImageProcessor.truncate_at_loop(text, max_length=800)

            print(
                f"  ✓ OCR success ({len(text)} chars{', has loops' if has_loops else ''})",
                flush=True,
            )

            # Remove library metadata
            cleaned_text = self.text_processor.remove_metadata_text(text)

            return cleaned_text, has_loops

        finally:
            # Clean up temporary preprocessed image
            if temp_preprocessed and os.path.exists(temp_preprocessed):
                try:
                    os.unlink(temp_preprocessed)
                except Exception:
                    pass


# ============================================================================
# TEXT CLEANING ENGINE
# ============================================================================


class TextCleaner:
    """Handles text cleaning operations with Kimi or DeepSeek"""

    def __init__(self, clients: APIClients, config: ProcessingConfig):
        self.clients = clients
        self.config = config
        self.text_processor = TextProcessor()

    def _get_cleanup_client_and_model(self):
        """Get the appropriate client and model name for cleanup operations"""
        if self.config.cleanup_model == "deepseek":
            return self.clients.deepseek, "deepseek-v4-flash"
        else:  # default to kimi
            return self.clients.kimi, "kimi-k2.6"

    def sanitize_output(self, text: str) -> str:
        """
        Remove context markers if they accidentally appear in output.
        This is a failsafe - should rarely be needed if prompts work correctly.
        """
        # Remove any [[...]] blocks that slipped through
        cleaned = re.sub(r"\[\[上文参考[^\]]*\]\]", "", text, flags=re.DOTALL)
        cleaned = cleaned.strip()
        return cleaned

    def detect_overlap_duplicate(
        self,
        prev_tail: str,
        curr_head: str,
        threshold: float = 0.85,
        min_length: int = 200,
    ) -> int:
        """
        Detect if curr_head contains a duplicate of prev_tail.
        Returns: length of duplicate if found, 0 otherwise.

        Uses character-level matching with high threshold to be conservative.
        """
        # Search for matching sequences at the boundary
        max_search = min(len(prev_tail), len(curr_head), 800)
        best_match_len = 0

        # Look for matches starting from the end of prev_tail
        for i in range(max_search, min_length, -10):  # Step by 10 for efficiency
            tail_segment = prev_tail[-i:]
            if curr_head.startswith(tail_segment):
                # Found exact match at boundary
                if i >= min_length:
                    similarity = 1.0
                    if similarity >= threshold:
                        best_match_len = i
                        break
            else:
                # Check fuzzy match
                matches = 0
                check_len = min(i, len(curr_head))
                for j in range(check_len):
                    if j < len(prev_tail) and prev_tail[-(i - j)] == curr_head[j]:
                        matches += 1

                similarity = matches / i if i > 0 else 0
                if similarity >= threshold and i >= min_length:
                    best_match_len = i
                    break

        return best_match_len

    def clean_chunk(
        self,
        text_chunk: str,
        context: str = "",
        has_ocr_loops: bool = False,
        has_marked_context: bool = False,
    ) -> str:
        """Clean a single chunk of text with retry logic"""
        start_time = time.time()
        loop_warning = ""
        if has_ocr_loops:
            loop_warning = "\n\n**重要提示**：此文本可能包含OCR识别循环导致的重复内容，请仔细检查并删除所有重复部分。"

        # Base requirements
        base_requirements = """1. 绝对必须保持原文的繁简体字形式。严禁擅自转换字形。
2. 【核心原则】严格保持所有原文字符不变。**仅在以下三种情况可考虑修正**：
   a. **字形高度相似且语境完全不通**（如「己」「已」「巳」在明显错误的语境）
   b. **明显不符合时代特征的用字**（如现代简体字混入明清文献）
   c. **同一字在文中稳定出现，仅个别处明显误写**（参考上下文一致性）
   修正时必须选择**最接近原字形**的合理汉字，并在修正处添加【？】标注。
3. 添加文言文适用的标点符号（句号、逗号、顿号、问号）。
4. 删除所有机构元数据（如页码、图书馆标记、[空页]等）。
5. 将文本整理成连贯段落，但**保持原文的章节层次和自然换行**。
6. 对于无法确定的字，用【？】标注。
7. **特别注意：如果发现重复出现的文本块（OCR识别循环错误），请删除这些重复部分**。
8. **如果文本明显不完整或被截断，保持原样即可**。
9. 【新增】**对于古汉语中的异体字、通假字、避讳字等，即使看起来不常见，也必须保留原字**。

重要提醒：当不确定是否应该修正时，一律选择**保留原字**。"""

        # Add special instructions for chunks with marked context
        if has_marked_context:
            context_instruction = """

**重要说明 - 关于上文参考**：
- 文本开头有 [[上文参考...]] 括起的部分，这是前一段的结尾，已经处理过
- [[...]] 中的内容仅供你理解上下文，帮助你正确理解接下来的文本
- **绝对不要在输出中包含或重复 [[...]] 中的任何内容**
- 只处理并输出 [[...]] 之后的文本
- 如果 [[...]] 后的文本开头是不完整的句子，请根据上下文理解其含义，但仍然不要输出上文参考的内容
- 直接从 [[...]] 后面的第一个字开始输出你的处理结果"""

            prompt = f"""请将以下OCR文本整理成连贯的文言文文献。這是明清時期的文獻。

要求：
{base_requirements}{loop_warning}{context_instruction}

{f"上下文提示：{context}" if context else ""}

待处理文本：
{text_chunk}

请直接返回整理后的文本，不要包含任何额外的说明、列表或标题，立即开始正文："""
        else:
            prompt = f"""请将以下OCR文本整理成连贯的文言文文献。這是明清時期的文獻。

要求：
{base_requirements}{loop_warning}

{f"上下文提示：{context}" if context else ""}

OCR文本：
{text_chunk}

请直接返回整理后的文本，不要包含任何额外的说明、列表或标题，立即开始正文："""

        for attempt in range(self.config.retry_attempts):
            try:
                sys.stdout.flush()  # Force flush before API call
                client, model = self._get_cleanup_client_and_model()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.config.api_timeout,
                    max_tokens=2048,
                )
                processing_time = time.time() - start_time
                print(f"✓ ({processing_time:.1f}s)", flush=True)

                # Sanitize output to remove any context markers that slipped through
                try:
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response content")
                except (IndexError, AttributeError, KeyError) as e:
                    raise ValueError(f"Unexpected API response structure: {e}")
                cleaned_output = self.sanitize_output(content)
                return cleaned_output

            except Exception as e:
                print(
                    f"\n✗ Error (attempt {attempt + 1}/{self.config.retry_attempts}): {e}",
                    flush=True,
                )
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2**attempt)
                    print(f"    Retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(
                        "    All retry attempts failed. Returning original text.",
                        flush=True,
                    )
                    return text_chunk

        return text_chunk

    def create_chunks_with_overlap(self, text: str) -> List[Tuple[int, int]]:
        """Create overlapping chunks for processing (stores indices only, not text)"""
        # Validate configuration to prevent infinite loops and data loss
        if self.config.max_text_length <= 0:
            raise ValueError(
                f"max_text_length ({self.config.max_text_length}) must be positive"
            )
        if self.config.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap ({self.config.chunk_overlap}) must be non-negative"
            )
        if self.config.chunk_overlap >= self.config.max_text_length:
            raise ValueError(
                f"chunk_overlap ({self.config.chunk_overlap}) must be less than "
                f"max_text_length ({self.config.max_text_length}) to avoid infinite loops"
            )

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.config.max_text_length, len(text))
            chunks.append((start, end))  # Store indices only, not chunk text

            # If we've reached the end of the text, stop
            if end >= len(text):
                break

            start = end - self.config.chunk_overlap

        print(
            f"Created {len(chunks)} chunks with {self.config.chunk_overlap}-char overlap",
            flush=True,
        )
        return chunks

    def process_text_in_chunks(
        self, text: str, context: str = "", has_ocr_loops: bool = False
    ) -> str:
        """
        Process text in chunks with marked context to prevent duplication.

        Strategy:
        1. First chunk: process normally
        2. Subsequent chunks: add [[上文参考...]] marker with previous overlap
        3. Apply conservative deduplication as safety net
        """
        if len(text) <= self.config.max_text_length:
            print("Text fits in single chunk, processing...", flush=True)
            return self.clean_chunk(
                text, context, has_ocr_loops, has_marked_context=False
            )

        print(f"Text length: {len(text)} chars - splitting into chunks...", flush=True)
        chunk_indices = self.create_chunks_with_overlap(text)
        cleaned_chunks = []

        # Process chunks one at a time
        for i, (start, end) in enumerate(chunk_indices, 1):
            print(
                f"\nProcessing chunk {i}/{len(chunk_indices)} (chars {start}-{end})... ",
                end="",
                flush=True,
            )

            # Extract chunk text from original
            chunk_text = text[start:end]

            # For non-first chunks, add context marker
            if i > 1:
                # Get context from original text (previous overlap region)
                context_start = max(0, start - self.config.chunk_overlap)
                context_text = text[context_start:start]

                # Create marked chunk with explicit context
                marked_chunk = f"""[[上文参考（已处理完毕，请勿在输出中重复）：
{context_text}
]]

{chunk_text}"""

                print(
                    f"With marked context, sending to Kimi API... ", end="", flush=True
                )
                cleaned = self.clean_chunk(
                    marked_chunk, context, has_ocr_loops, has_marked_context=True
                )
            else:
                print(f"First chunk, sending to Kimi API... ", end="", flush=True)
                cleaned = self.clean_chunk(
                    chunk_text, context, has_ocr_loops, has_marked_context=False
                )

            cleaned_chunks.append(cleaned)
            del chunk_text  # Explicitly release references

        # Clear chunk indices
        del chunk_indices

        # SAFETY NET: Apply conservative deduplication between chunks
        print(f"\n\nApplying deduplication safety net...", flush=True)
        deduped_chunks = [cleaned_chunks[0]]  # First chunk always kept as-is

        for i in range(1, len(cleaned_chunks)):
            prev_chunk = deduped_chunks[-1]
            curr_chunk = cleaned_chunks[i]

            # Check for duplicates at the boundary
            duplicate_len = self.detect_overlap_duplicate(
                prev_tail=prev_chunk[-800:],  # Check last 800 chars of previous
                curr_head=curr_chunk[:800],  # Check first 800 chars of current
                threshold=0.85,
                min_length=200,
            )

            if duplicate_len > 0:
                print(
                    f"   Chunk {i + 1}: Removed {duplicate_len}-char duplicate",
                    flush=True,
                )
                deduped_chunks.append(curr_chunk[duplicate_len:])
            else:
                deduped_chunks.append(curr_chunk)

        # Join deduplicated chunks
        combined = "\n\n".join(deduped_chunks)
        print(
            f"✓ Combined {len(deduped_chunks)} chunks into {len(combined)} chars",
            flush=True,
        )

        # Clear intermediate lists
        del cleaned_chunks
        del deduped_chunks

        return combined

    def generate_review_report(self, text: str, context: str = "") -> str:
        """
        Generate quality review report for processed text.
        DOES NOT rewrite text - only identifies issues for manual review.
        Prevents data loss from token limits.
        """
        print("\n=== Quality Review Analysis ===", flush=True)

        # Sample text for analysis (first 20000 chars + last 20000 chars)
        # This ensures we check beginning and end without exceeding token limits
        text_length = len(text)
        if text_length > 40000:
            sample_text = (
                text[:20000] + "\n\n[...中间部分已省略...]\n\n" + text[-20000:]
            )
            print(
                f"   Analyzing sample: {text_length} chars (sampling first/last 20k)",
                flush=True,
            )
        else:
            sample_text = text
            print(f"   Analyzing full text: {text_length} chars", flush=True)

        prompt = f"""你是一位古籍整理专家。请仔细审阅以下已处理的文言文文本，识别可能存在的问题。

这是明清时期的文献。文本已经过OCR识别和初步整理。

请分析文本并列出以下类型的问题：

1. **繁简体混用**：检查是否有擅自将繁體字转换为简体字（或反之）的情况，这是严重错误
2. **重复内容**：明显重复的段落或句子（非原文本身的重复结构）
3. **OCR错误**：可能的误认字（形近字混淆）
4. **标点问题**：标点符号使用明显不当或不一致
5. **格式问题**：段落结构混乱或异常
6. **说明性文字**：AI可能添加的注释或说明（应删除）
7. **截断问题**：文本开头或结尾是否有截断迹象

{f"上下文提示：{context}" if context else ""}

文本样本：
{sample_text}

请按以下格式输出审查报告：

## 质量审查报告

### 1. 重复内容
[如有发现，列出具体位置和内容；如无，写"未发现"]

### 2. 可能的OCR错误
[列出可疑的字符，如"某某某（疑为XX）"；如无，写"未发现明显错误"]

### 3. 标点问题
[列出标点使用不当的地方；如无，写"标点使用基本正常"]

### 4. 格式问题
[列出格式混乱的地方；如无，写"格式基本正常"]

### 5. 说明性文字
[列出需要删除的注释或说明；如无，写"未发现"]

### 6. 文本完整性
[评估文本开头和结尾是否完整；是否有截断]

### 7. 总体评价
[简要评价处理质量，1-5星]

### 8. 建议
[给出人工审校的重点建议]

请客观、具体地指出问题，便于人工审校。"""

        for attempt in range(self.config.retry_attempts):
            try:
                client, model = self._get_cleanup_client_and_model()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.config.api_timeout,
                    max_tokens=4096,  # Sufficient for review report
                )
                try:
                    review_report = response.choices[0].message.content
                    if not review_report:
                        raise ValueError("Empty response content")
                except (IndexError, AttributeError, KeyError) as e:
                    raise ValueError(f"Unexpected API response structure: {e}")
                print("✓ Quality review complete", flush=True)
                return review_report

            except Exception as e:
                print(
                    f"✗ Review error (attempt {attempt + 1}/{self.config.retry_attempts}): {e}",
                    flush=True,
                )
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2**attempt)
                    print(f"    Retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                else:
                    print("    All retry attempts failed. Skipping review.", flush=True)
                    return "质量审查失败：API调用超时或出错\n\n建议人工全面审校文本。"

        return "质量审查失败：达到最大重试次数\n\n建议人工全面审校文本。"


class HighConcurrencyTextCleaner(TextCleaner):
    """
    Text cleaner optimized for high-concurrency APIs (Kimi Tier 3, DeepSeek unlimited).
    Dynamically adjusts concurrency based on service and document size.
    """

    # Service-specific limits (conservative defaults)
    SERVICE_LIMITS = {
        "kimi": {
            "max_concurrent": 50,  # Conservative: 50 of 200 available
            "max_tokens_per_minute": 2500000,  # 2.5M of 3M TPM
            "requests_per_minute": 600,  # 10 RPS
            "tokens_per_request": 3500,  # Average estimate
        },
        "deepseek": {
            "max_concurrent": 100,  # No official limit, be reasonable
            "max_tokens_per_minute": float("inf"),  # No limit claimed
            "requests_per_minute": float("inf"),  # No limit claimed
            "tokens_per_request": 3500,
        },
    }

    def __init__(self, clients: APIClients, config: ProcessingConfig):
        super().__init__(clients, config)
        self.service = config.cleanup_model
        self.limits = self.SERVICE_LIMITS[self.service]

        # Rate limiting state
        self.token_counter = 0
        self.request_counter = 0
        self.window_start = time.time()
        self.lock = threading.Lock()

        # Performance tracking
        self.response_times = []
        self.success_count = 0
        self.failure_count = 0

    def process_text_in_chunks(
        self, text: str, context: str = "", has_ocr_loops: bool = False
    ) -> str:
        """
        Main entry point with intelligent parallelization.
        """
        chunk_indices = self.create_chunks_with_overlap(text)
        total_chunks = len(chunk_indices)

        # Decision logic for parallelization
        if total_chunks <= 3:
            print(f"📝 Small document ({total_chunks} chunks), using sequential mode")
            return super().process_text_in_chunks(text, context, has_ocr_loops)

        # Calculate optimal concurrency
        optimal_workers = self._calculate_optimal_concurrency(total_chunks)

        print(
            f"🚀 Processing {total_chunks} chunks with {optimal_workers} parallel workers "
            f"({self.service.upper()})"
        )

        return self._process_high_concurrency(
            text, chunk_indices, context, has_ocr_loops, optimal_workers
        )

    def _calculate_optimal_concurrency(self, total_chunks: int) -> int:
        """Calculate optimal number of workers."""
        max_concurrent = self.limits["max_concurrent"]

        if total_chunks <= 10:
            return min(5, total_chunks, max_concurrent)

        if total_chunks <= 50:
            return min(total_chunks // 2, max_concurrent)

        optimal = min(total_chunks, max_concurrent)

        if len(self.response_times) > 10:
            avg_time = sum(self.response_times[-10:]) / 10
            if avg_time > 30:
                optimal = max(10, optimal // 2)
                print(f"   ⚡ Reducing concurrency to {optimal} (slow responses)")

        return optimal

    def _process_high_concurrency(
        self,
        text: str,
        chunk_indices: List[Tuple[int, int]],
        context: str,
        has_ocr_loops: bool,
        max_workers: int,
    ) -> str:
        """High-concurrency parallel processing."""
        total_chunks = len(chunk_indices)
        start_time = time.time()

        # Prepare all tasks
        tasks = []
        for i, (start, end) in enumerate(chunk_indices, 1):
            chunk_text = text[start:end]
            is_first_chunk = i == 1

            prompt = self._build_chunk_prompt(
                chunk_text, context, has_ocr_loops, not is_first_chunk
            )

            tasks.append(
                {
                    "index": i - 1,
                    "chunk_text": chunk_text,
                    "prompt": prompt,
                    "is_first_chunk": is_first_chunk,
                    "estimated_tokens": len(chunk_text) * 1.5 + 500,
                }
            )

        # Process in parallel
        results = self._execute_with_adaptive_concurrency(tasks, max_workers)

        # Combine results
        combined_text = self._combine_results(results, total_chunks)

        elapsed = time.time() - start_time
        self._print_performance_summary(elapsed, total_chunks, len(combined_text))

        return combined_text

    def _build_chunk_prompt(
        self, chunk_text: str, context: str, has_loops: bool, has_marked_context: bool
    ) -> str:
        """Build prompt for a single chunk."""
        # Use the prompt building logic from your existing clean_chunk method
        # Extract and adapt it here (or call super().clean_chunk with a flag)
        loop_warning = (
            "\n\n**重要提示**：此文本可能包含OCR识别循环导致的重复内容，请仔细检查并删除所有重复部分。"
            if has_loops
            else ""
        )

        base_requirements = """1. 绝对必须保持原文的繁简体字形式。严禁擅自转换字形。
2. 【核心原则】严格保持所有原文字符不变。**仅在以下三种情况可考虑修正**：
   a. **字形高度相似且语境完全不通**（如「己」「已」「巳」在明显错误的语境）
   b. **明显不符合时代特征的用字**（如现代简体字混入明清文献）
   c. **同一字在文中稳定出现，仅个别处明显误写**（参考上下文一致性）
   修正时必须选择**最接近原字形**的合理汉字，并在修正处添加【？】标注。
3. 添加文言文适用的标点符号（句号、逗号、顿号、问号）。
4. 删除所有机构元数据（如页码、图书馆标记、[空页]等）。
5. 将文本整理成连贯段落，但**保持原文的章节层次和自然换行**。
6. 对于无法确定的字，用【？】标注。
7. **特别注意：如果发现重复出现的文本块（OCR识别循环错误），请删除这些重复部分**。
8. **如果文本明显不完整或被截断，保持原样即可**。
9. 【新增】**对于古汉语中的异体字、通假字、避讳字等，即使看起来不常见，也必须保留原字**。

重要提醒：当不确定是否应该修正时，一律选择**保留原字**。"""

        if has_marked_context:
            context_instruction = """

**重要说明 - 关于上文参考**：
- 文本开头有 [[上文参考...]] 括起的部分，这是前一段的结尾，已经处理过
- [[...]] 中的内容仅供你理解上下文，帮助你正确理解接下来的文本
- **绝对不要在输出中包含或重复 [[...]] 中的任何内容**
- 只处理并输出 [[...]] 之后的文本
- 如果 [[...]] 后的文本开头是不完整的句子，请根据上下文理解其含义，但仍然不要输出上文参考的内容
- 直接从 [[...]] 后面的第一个字开始输出你的处理结果"""

            prompt = f"""请将以下OCR文本整理成连贯的文言文文献。這是明清時期的文獻。

要求：
{base_requirements}{loop_warning}{context_instruction}

{f"上下文提示：{context}" if context else ""}

待处理文本：
{chunk_text}

请直接返回整理后的文本，不要包含任何额外的说明、列表或标题，立即开始正文："""
        else:
            prompt = f"""请将以下OCR文本整理成连贯的文言文文献。這是明清時期的文獻。

要求：
{base_requirements}{loop_warning}

{f"上下文提示：{context}" if context else ""}

OCR文本：
{chunk_text}

请直接返回整理后的文本，不要包含任何额外的说明、列表或标题，立即开始正文："""

        return prompt

    def _execute_with_adaptive_concurrency(
        self, tasks: List[dict], max_workers: int
    ) -> List[dict]:
        """Execute tasks with adaptive concurrency control."""
        results = []
        task_queue = tasks.copy()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}

            # Initial submission
            initial_batch = min(max_workers * 2, len(task_queue))
            for _ in range(initial_batch):
                if task_queue:
                    task = task_queue.pop(0)
                    future = executor.submit(self._process_single_task, task)
                    future_to_task[future] = task

            # Process completed futures
            while future_to_task:
                done, _ = concurrent.futures.wait(
                    future_to_task.keys(),
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    task = future_to_task.pop(future)

                    try:
                        result = future.result()
                        results.append(result)

                        # Update performance metrics
                        if "processing_time" in result:
                            with self.lock:
                                self.response_times.append(result["processing_time"])
                                if len(self.response_times) > 100:
                                    self.response_times.pop(0)

                        # Submit new task if queue not empty
                        if task_queue:
                            next_task = task_queue.pop(0)
                            new_future = executor.submit(
                                self._process_single_task, next_task
                            )
                            future_to_task[new_future] = next_task

                    except Exception as e:
                        print(f"   ✗ Task {task['index'] + 1} failed: {e}")
                        results.append(
                            {
                                "index": task["index"],
                                "result": task["chunk_text"],
                                "status": "failed",
                            }
                        )

        return results

    def _process_single_task(self, task: dict) -> dict:
        """Process a single task with rate limiting."""
        task_idx = task["index"] + 1
        start_time = time.time()

        # Apply rate limiting if needed
        self._apply_rate_limits(task["estimated_tokens"])

        for attempt in range(self.config.retry_attempts):
            try:
                client, model = self._get_cleanup_client_and_model()

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": task["prompt"]}],
                    timeout=self.config.api_timeout,
                    max_tokens=2048,
                )

                processing_time = time.time() - start_time
                try:
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Empty response content")
                except (IndexError, AttributeError, KeyError) as e:
                    raise ValueError(f"Unexpected API response structure: {e}")
                result = self.sanitize_output(content)

                # Update rate limiting counters
                self._update_counters(task["estimated_tokens"])

                # Print progress
                with self.lock:
                    self.success_count += 1
                    print(f"   ✓ Chunk {task_idx} ({processing_time:.1f}s)", flush=True)

                return {
                    "index": task["index"],
                    "result": result,
                    "status": "success",
                    "processing_time": processing_time,
                }

            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2**attempt)
                    print(f"   ⚠️ Chunk {task_idx} retry {attempt + 1}: {e}")
                    time.sleep(wait_time)
                else:
                    with self.lock:
                        self.failure_count += 1
                        print(f"   ✗ Chunk {task_idx} failed after retries")

                    return {
                        "index": task["index"],
                        "result": task["chunk_text"],
                        "status": "failed",
                        "error": str(e),
                    }

    def _apply_rate_limits(self, estimated_tokens: int):
        """Apply rate limits (Kimi only)."""
        if self.service != "kimi":
            return

        current_time = time.time()
        window_elapsed = current_time - self.window_start

        if window_elapsed > 60:
            with self.lock:
                self.token_counter = 0
                self.request_counter = 0
                self.window_start = current_time
            window_elapsed = 0

        with self.lock:
            projected_tokens = self.token_counter + estimated_tokens

            if projected_tokens > self.limits["max_tokens_per_minute"]:
                tokens_over = projected_tokens - self.limits["max_tokens_per_minute"]
                wait_time = (tokens_over / self.limits["max_tokens_per_minute"]) * 60
                wait_time = min(wait_time, 60 - window_elapsed)

                if wait_time > 0.1:
                    print(f"   ⏳ TPM limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)

                self.token_counter = 0
                self.window_start = time.time()

    def _update_counters(self, tokens_used: int):
        """Update rate limiting counters."""
        if self.service != "kimi":
            return

        with self.lock:
            self.token_counter += tokens_used
            self.request_counter += 1

    def _combine_results(self, results: List[dict], total_chunks: int) -> str:
        """Combine and deduplicate results."""
        sorted_results = sorted(results, key=lambda x: x["index"])
        cleaned_chunks = [r["result"] for r in sorted_results]

        print(f"\n🔗 Combining {len(cleaned_chunks)} results with deduplication...")

        if len(cleaned_chunks) <= 1:
            return cleaned_chunks[0] if cleaned_chunks else ""

        deduped_chunks = [cleaned_chunks[0]]

        for i in range(1, len(cleaned_chunks)):
            prev_chunk = deduped_chunks[-1]
            curr_chunk = cleaned_chunks[i]

            duplicate_len = self.detect_overlap_duplicate(
                prev_tail=prev_chunk[-800:],
                curr_head=curr_chunk[:800],
                threshold=0.85,
                min_length=200,
            )

            if duplicate_len > 0:
                print(
                    f"   Removed {duplicate_len}-char duplicate between chunks {i}→{i + 1}"
                )
                deduped_chunks.append(curr_chunk[duplicate_len:])
            else:
                deduped_chunks.append(curr_chunk)

        combined = "\n\n".join(deduped_chunks)

        failed_count = sum(1 for r in results if r["status"] == "failed")
        if failed_count > 0:
            print(f"⚠️  Warning: {failed_count} chunks failed and used original text")

        return combined

    def _print_performance_summary(
        self, elapsed: float, total_chunks: int, output_chars: int
    ):
        """Print performance summary."""
        chars_per_sec = output_chars / elapsed if elapsed > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"🚀 PARALLEL PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"   Service:        {self.service.upper()}")
        print(f"   Total time:     {elapsed:.1f}s")
        print(f"   Chunks:         {total_chunks}")
        print(f"   Output length:  {output_chars} chars")
        print(f"   Throughput:     {chars_per_sec:.0f} chars/sec")
        print(f"   Success rate:   {self.success_count}/{total_chunks} chunks")

        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            print(f"   Avg response:   {avg_time:.1f}s")

        print(f"{'=' * 60}")


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================


class DocumentProcessor:
    """Main document processing orchestrator"""

    # Default OCR concurrency settings
    DEFAULT_OCR_WORKERS = 15  # Conservative default to avoid memory issues

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.clients = APIClients(cleanup_model=config.cleanup_model)
        self.ocr_engine = OCREngine(self.clients, config)
        self.text_cleaner = HighConcurrencyTextCleaner(self.clients, config)
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
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.config.ocr_dpi,
                    first_page=page_num,
                    last_page=page_num,
                )

                if not images:
                    print(f"  ⚠️  Page {page_num}: conversion failed", flush=True)
                    continue

                # Save with consistent naming for easy sorting
                image_path = os.path.join(temp_dir, f"page_{page_num:04d}.png")
                images[0].save(image_path, "PNG")
                extracted.append((page_num, image_path))

                # Free PIL image immediately
                del images

            except Exception as e:
                print(f"  ⚠️  Page {page_num}: extraction error - {e}", flush=True)

        print(f"  ✓ Extracted {len(extracted)}/{total_pages} pages", flush=True)
        return extracted

    def _ocr_page_worker(
        self, page_num: int, image_path: str, total_pages: int
    ) -> Tuple[int, Optional[str], bool]:
        """
        Worker function for parallel OCR. Processes a single page.
        Returns: (page_num, text, has_loops)
        """
        try:
            # Process the image (OCREngine handles preprocessing internally)
            result = self.ocr_engine.process_image(
                image_path, page_num, total_pages, current_index=page_num
            )

            # Update progress (thread-safe)
            with self._ocr_progress_lock:
                self._ocr_completed_count += 1
                completed = self._ocr_completed_count
            print(f"  📊 OCR progress: {completed}/{total_pages} pages", flush=True)

            # Web service progress callback
            if hasattr(self, "_web_progress_callback") and self._web_progress_callback:
                percent = 10 + int((completed / total_pages) * 60)  # OCR is 10-70%
                self._web_progress_callback("ocr", page_num, percent)

            if result:
                text, has_loops = result
                return (page_num, text, has_loops)
            else:
                return (page_num, None, False)

        except Exception as e:
            print(f"  ❌ Page {page_num} OCR error: {e}", flush=True)
            return (page_num, None, False)
        finally:
            # Clean up the temp image immediately after processing
            try:
                if os.path.exists(image_path):
                    os.unlink(image_path)
            except Exception:
                pass

    def _parallel_ocr(
        self, extracted_pages: List[Tuple[int, str]], total_pages: int
    ) -> Tuple[List[str], List[int]]:
        """
        Phase 2: Process extracted images with parallel OCR.
        Returns: (all_ocr_texts, pages_with_loops)
        """
        # Reset progress counter
        self._ocr_completed_count = 0

        # Determine worker count
        num_workers = min(self.max_ocr_workers, len(extracted_pages))
        print(f"\n🚀 Starting parallel OCR with {num_workers} workers...", flush=True)

        results = {}  # Dict keyed by page number for ordering
        pages_with_loops = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(
                    self._ocr_page_worker, page_num, image_path, total_pages
                ): page_num
                for page_num, image_path in extracted_pages
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    if result is None:
                        print(f"  ❌ Page {page_num}: OCR returned None", flush=True)
                        continue
                    _, text, has_loops = result
                    if text:
                        results[page_num] = text
                        if has_loops:
                            pages_with_loops.append(page_num)
                except Exception as e:
                    print(f"  ❌ Page {page_num} failed: {e}", flush=True)

        # Phase 3: Assemble results in correct order
        print(f"\n🔗 Assembling {len(results)} pages in order...", flush=True)
        ordered_texts = [results[page_num] for page_num in sorted(results.keys())]

        return ordered_texts, pages_with_loops

    def _sequential_ocr(
        self, pdf_path: str, start_page: int, end_page: int, page_count: int
    ) -> Tuple[List[str], List[int]]:
        """
        Original sequential OCR flow (convert-OCR-delete one page at a time).
        Used when --sequential flag is set or for very small documents.
        """
        all_ocr_texts = []
        pages_with_loops = []

        current_page_index = 0
        for page_num in range(start_page, end_page + 1):
            current_page_index += 1

            # Convert single page
            images = convert_from_path(
                pdf_path,
                dpi=self.config.ocr_dpi,
                first_page=page_num,
                last_page=page_num,
            )

            if not images:
                print(f"⚠️  Skipping page {page_num} - conversion failed", flush=True)
                continue

            # Save as temporary image
            temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            images[0].save(temp_image.name, "PNG")
            del images  # Free PIL image

            try:
                # Process with OCR (pass current index for accurate progress)
                result = self.ocr_engine.process_image(
                    temp_image.name, page_num, page_count, current_page_index
                )
                if result:
                    text, has_loops = result
                    all_ocr_texts.append(text)
                    if has_loops:
                        pages_with_loops.append(page_num)
            finally:
                # Clean up temp image
                if os.path.exists(temp_image.name):
                    os.unlink(temp_image.name)

        return all_ocr_texts, pages_with_loops

    def process_image(
        self,
        image_path: str,
        context: str = "",
        output_dir: str = "./processed",
        review_mode: bool = False,
    ) -> Optional[Path]:
        """Process a single image file"""
        print(f"\n{'=' * 60}", flush=True)
        print(f"Processing image: {image_path}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # OCR
        result = self.ocr_engine.process_image(image_path, 1, 1)
        if not result:
            print("❌ OCR failed", flush=True)
            return None

        raw_text, has_loops = result

        # Save raw OCR
        os.makedirs(output_dir, exist_ok=True)
        doc_name = Path(image_path).stem

        # Clean text
        print("\n=== Text Cleaning Stage ===", flush=True)
        cleaned_text = self.text_cleaner.process_text_in_chunks(
            raw_text, context, has_loops
        )

        # Save markdown (ALWAYS)
        pages_with_loops = [1] if has_loops else []
        output_path = self._create_consolidated_note(
            doc_name, cleaned_text, pages_with_loops, output_dir, context, 1
        )
        print(f"✓ Markdown saved: {output_path}", flush=True)

        # Optional: Generate quality review report
        if review_mode:
            review_report = self.text_cleaner.generate_review_report(
                cleaned_text, context
            )
            review_path = Path(output_dir) / f"{doc_name}_review.txt"
            with open(review_path, "w", encoding="utf-8") as f:
                f.write(f"# 质量审查报告\n\n")
                f.write(f"文档: {doc_name}\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"文本长度: {len(cleaned_text)} 字符\n\n")
                f.write("=" * 60 + "\n\n")
                f.write(review_report)
            print(f"✓ Review report saved: {review_path}", flush=True)

        print(f"\n✓ Processing complete: {output_path}", flush=True)
        return output_path

    def process_pdf(
        self,
        pdf_path: str,
        context: str = "",
        output_dir: str = "./processed",
        review_mode: bool = False,
        max_pages: Optional[int] = None,
        start_page: int = 1,
        export_zotero: bool = False,
        zotero_title: Optional[str] = None,
        zotero_collection: Optional[str] = None,
    ) -> Optional[Path]:
        """Process a PDF file"""
        print(f"\n{'=' * 60}", flush=True)
        print(f"Processing PDF: {pdf_path}", flush=True)
        print(f"Output directory: {output_dir}", flush=True)
        print(f"{'=' * 60}", flush=True)

        if not PDF_SUPPORT:
            print("❌ PDF support not available", flush=True)
            return None

        try:
            # Get page info
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]
            end_page = (
                min(start_page + max_pages - 1, total_pages)
                if max_pages
                else total_pages
            )
            page_count = end_page - start_page + 1

            print(f"Total PDF pages: {total_pages}", flush=True)
            print(
                f"Processing pages {start_page} to {end_page} ({page_count} pages)",
                flush=True,
            )
            print(
                f"DPI: {self.config.ocr_dpi}, Model: {self.config.qwen_model}",
                flush=True,
            )

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            doc_name = (
                Path(self._original_filename).stem
                if getattr(self, "_original_filename", None)
                else Path(pdf_path).stem
            )

            # OCR Stage
            print("\n" + "=" * 60, flush=True)
            print("STAGE 1: OCR EXTRACTION", flush=True)
            print("=" * 60, flush=True)

            start_time = time.time()

            # Choose parallel or sequential based on settings and page count
            use_parallel = self.parallel_ocr and page_count > 3

            if use_parallel:
                print(f"Mode: PARALLEL ({self.max_ocr_workers} workers)", flush=True)

                # Create temp directory for images
                temp_dir = tempfile.mkdtemp(prefix="zhiliu_ocr_")

                try:
                    # Phase 1: Extract all images sequentially
                    extracted_pages = self._extract_all_images(
                        pdf_path, start_page, end_page, temp_dir
                    )

                    if not extracted_pages:
                        print("❌ No pages extracted", flush=True)
                        return None

                    # Phase 2 & 3: Parallel OCR and assembly
                    all_ocr_texts, pages_with_loops = self._parallel_ocr(
                        extracted_pages, page_count
                    )

                finally:
                    # Clean up temp directory (any remaining files)
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass

            else:
                print(f"Mode: SEQUENTIAL", flush=True)
                all_ocr_texts, pages_with_loops = self._sequential_ocr(
                    pdf_path, start_page, end_page, page_count
                )

            ocr_time = time.time() - start_time

            if not all_ocr_texts:
                print("❌ No text extracted from any pages", flush=True)
                return None

            # Combine OCR texts
            combined_ocr = "\n\n".join(all_ocr_texts)
            print(
                f"\n✓ OCR complete: {len(combined_ocr)} chars from {len(all_ocr_texts)} pages",
                flush=True,
            )
            print(f"   Time: {ocr_time:.1f}s", flush=True)
            print(f"   Pages with loops: {len(pages_with_loops)}", flush=True)

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
                },
            }

            backup_path = Path(output_dir) / f"{doc_name}_raw_ocr_latest.json"
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            print(f"   Backup saved: {backup_path}", flush=True)

            # Cleaning Stage
            print("\n" + "=" * 60, flush=True)
            print("STAGE 2: TEXT CLEANING", flush=True)
            print("=" * 60, flush=True)

            has_loops = len(pages_with_loops) > 0
            cleaned_text = self.text_cleaner.process_text_in_chunks(
                combined_ocr, context, has_loops
            )

            # Save markdown after Stage 2 (ALWAYS - prevents data loss)
            print("\n" + "=" * 60, flush=True)
            print("SAVING OUTPUT", flush=True)
            print("=" * 60, flush=True)

            output_path = self._create_consolidated_note(
                doc_name,
                cleaned_text,
                pages_with_loops,
                output_dir,
                context,
                page_count,
            )
            print(f"✓ Markdown saved: {output_path}", flush=True)

            # Optional: Generate quality review report
            if review_mode:
                review_report = self.text_cleaner.generate_review_report(
                    cleaned_text, context
                )
                review_path = Path(output_dir) / f"{doc_name}_review.txt"
                with open(review_path, "w", encoding="utf-8") as f:
                    f.write(f"# 质量审查报告\n\n")
                    f.write(f"文档: {doc_name}\n")
                    f.write(
                        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"文本长度: {len(cleaned_text)} 字符\n\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(review_report)
                print(f"✓ Review report saved: {review_path}", flush=True)

            total_time = time.time() - start_time
            print(f"\n{'=' * 60}", flush=True)
            print(f"✓ Processing complete in {total_time:.1f}s", flush=True)
            print(f"✓ Output: {output_path}", flush=True)
            print(f"{'=' * 60}", flush=True)

            # Export to Zotero if requested
            if export_zotero:
                print(f"\n📚 Exporting to Zotero...", flush=True)
                collection_key = None
                if zotero_collection:
                    collection_key = self.zotero.get_collection_key(zotero_collection)

                self.zotero.export_to_zotero(
                    output_path,
                    pdf_path,
                    self.config.qwen_model,
                    page_count,
                    title=zotero_title,
                    collection_key=collection_key,
                )

            return output_path

        except Exception as e:
            print(f"❌ Error processing PDF: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return None

    def resume_from_backup(
        self,
        backup_path: str,
        context: str = "",
        output_dir: str = "./processed",
        review_mode: bool = False,
        export_zotero: bool = False,
        zotero_title: Optional[str] = None,
        zotero_collection: Optional[str] = None,
    ):
        """Resume processing from a raw OCR backup file"""
        print(f"\n{'=' * 60}", flush=True)
        print(f"Resuming from backup: {backup_path}", flush=True)
        print(f"{'=' * 60}", flush=True)

        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)

            doc_name = backup_data["document"]
            combined_ocr = backup_data["raw_ocr_text"]
            pages_with_loops = backup_data.get("pages_with_loops", [])
            page_count = backup_data.get("pages_processed", 0)
            saved_context = backup_data.get("context", "")

            # Use provided context or fall back to saved context
            final_context = context if context else saved_context

            print(f"Document: {doc_name}", flush=True)
            print(f"Text length: {len(combined_ocr)} chars", flush=True)
            print(f"Pages with loops: {len(pages_with_loops)}", flush=True)

            # Cleaning Stage
            print("\n" + "=" * 60, flush=True)
            print("STAGE 2: TEXT CLEANING (RESUMED)", flush=True)
            print("=" * 60, flush=True)
            print("", flush=True)  # Empty line for readability

            has_loops = len(pages_with_loops) > 0
            print("Starting text chunking and cleaning...", flush=True)
            cleaned_text = self.text_cleaner.process_text_in_chunks(
                combined_ocr, final_context, has_loops
            )

            # Save markdown (ALWAYS)
            print("\n" + "=" * 60, flush=True)
            print("SAVING OUTPUT", flush=True)
            print("=" * 60, flush=True)

            output_path = self._create_consolidated_note(
                doc_name,
                cleaned_text,
                pages_with_loops,
                output_dir,
                final_context,
                page_count,
            )
            print(f"✓ Markdown saved: {output_path}", flush=True)

            # Optional: Generate quality review report
            if review_mode:
                review_report = self.text_cleaner.generate_review_report(
                    cleaned_text, final_context
                )
                review_path = Path(output_dir) / f"{doc_name}_review.txt"
                with open(review_path, "w", encoding="utf-8") as f:
                    f.write(f"# 质量审查报告\n\n")
                    f.write(f"文档: {doc_name}\n")
                    f.write(
                        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"文本长度: {len(cleaned_text)} 字符\n\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(review_report)
                print(f"✓ Review report saved: {review_path}", flush=True)

            print(f"\n✓ Resume processing complete: {output_path}", flush=True)

            # Export to Zotero if requested
            if export_zotero:
                print(f"\n📚 Exporting to Zotero...", flush=True)
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
                    collection_key=collection_key,
                )

            return output_path

        except Exception as e:
            print(f"❌ Error resuming from backup: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return None

    def _create_consolidated_note(
        self,
        document_name: str,
        full_cleaned_text: str,
        pages_with_loops: List[int],
        output_dir: str,
        context: str,
        page_count: int,
    ) -> Path:
        """Create consolidated markdown note"""
        collection = (
            Path(output_dir).name
            if output_dir != "./processed"
            else Path(document_name).stem
        )

        # Generate quality section if there were problematic pages
        quality_section = ""
        if pages_with_loops:
            quality_section = f"""
## ⚠️ OCR质量提示

检测到 {len(pages_with_loops)} 页可能包含OCR循环错误，已标记供Kimi清理：

**需要检查的页面**: {", ".join(map(str, pages_with_loops[:20]))}
{f"...及其他 {len(pages_with_loops) - 20} 页" if len(pages_with_loops) > 20 else ""}

这些页面的重复内容应该已被自动清理，但建议人工复核。

---
"""

        # Build the document
        note_content = f"""---
type: primary-source
source: {document_name}
collection: {collection}
date-processed: {datetime.now().strftime("%Y-%m-%d")}
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
> - **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
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
- {datetime.now().strftime("%Y-%m-%d")}: OCR (Qwen-VL) and initial processing
- Pages processed: {page_count}
- Model used: {self.config.qwen_model}
- Pages with loops: {len(pages_with_loops)}
"""

        # Save file
        output_path = Path(output_dir) / f"{document_name}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(note_content)

        print(f"\n✓ Created consolidated note: {output_path}", flush=True)
        return output_path


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Process classical Chinese documents with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  %(prog)s scan.jpg --output ./processed

  # Process PDF with context
  %(prog)s document.pdf --context "明代文集，竖排无标点" --output ~/Obsidian/Primary-Sources/

  # Generate quality review report (optional)
  %(prog)s document.pdf --review --output ./processed

  # Use alternative OCR model
  %(prog)s document.pdf --model qwen-vl-max --output ./processed

  # Use DeepSeek for text cleanup (cheaper alternative to Kimi)
  %(prog)s document.pdf --cleanup-model deepseek --output ./processed

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
  %(prog)s ./scans/ --batch --context "泉州府志" --output ~/Obsidian/Primary-Sources

  # Force sequential processing (for debugging)
  %(prog)s document.pdf --sequential --output ./processed

  # Limit maximum concurrent requests
  %(prog)s document.pdf --max-concurrent 20 --output ./processed
""",
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Image file, PDF, or directory (not required with --resume-from)",
    )
    parser.add_argument("--context", default="", help="Contextual information")
    parser.add_argument("--output", default="./processed", help="Output directory")
    parser.add_argument(
        "--batch", action="store_true", help="Process all files in directory"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF conversion (default: 300 for optimal OCR)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-plus",
        choices=["qwen3-vl-plus", "qwen-vl-max"],
        help="Qwen OCR model",
    )
    parser.add_argument(
        "--cleanup-model",
        default="deepseek",
        choices=["kimi", "deepseek"],
        help="Text cleanup model (default: deepseek)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Generate quality review report after processing",
    )
    parser.add_argument(
        "--max-pages", type=int, help="Limit processing to first N pages"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Start processing from page N (default: 1)",
    )
    parser.add_argument(
        "--end-page", type=int, help="Last page to process (alternative to max-pages)"
    )
    parser.add_argument("--resume-from", help="Resume from raw OCR JSON file")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential processing (disables parallel)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent requests (default: auto-detected)",
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

    # Define start and end pages
    if args.end_page:
        if args.start_page:
            # If both start and end specified, calculate max_pages
            args.max_pages = args.end_page - args.start_page + 1
        else:
            # If only end specified, process from page 1 to end
            args.max_pages = args.end_page

    # Initialize processor
    processor = DocumentProcessor(config)

    # Handle sequential mode - affects both OCR and text cleaning
    if args.sequential:
        print("⚠️  Sequential mode forced (--sequential flag)", flush=True)
        processor.parallel_ocr = False
        processor.text_cleaner = TextCleaner(processor.clients, config)
    # Apply max_concurrent override if specified - affects both OCR and text cleaning
    elif args.max_concurrent:
        processor.max_ocr_workers = args.max_concurrent
        processor.text_cleaner.limits["max_concurrent"] = args.max_concurrent
        print(f"⚙️  Concurrency limit set to {args.max_concurrent} workers", flush=True)

    # Handle resume case
    if args.resume_from:
        if not Path(args.resume_from).exists():
            print(f"❌ Error: Backup file {args.resume_from} not found", flush=True)
            sys.exit(1)
        processor.resume_from_backup(
            args.resume_from,
            args.context,
            args.output,
            args.review,
            export_zotero=args.zotero,
            zotero_title=args.zotero_title,
            zotero_collection=args.zotero_collection,
        )
        return

    # Validate input
    if not args.input:
        print(
            "❌ Error: Input file or directory required when not using --resume-from",
            flush=True,
        )
        sys.exit(1)

    # Batch processing
    if args.batch:
        input_path = Path(args.input)
        if not input_path.is_dir():
            print(f"❌ Error: {args.input} is not a directory", flush=True)
            sys.exit(1)

        images = (
            list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpeg"))
        )
        pdfs = list(input_path.glob("*.pdf"))

        if not images and not pdfs:
            print(f"❌ No images or PDFs found in {args.input}", flush=True)
            sys.exit(1)

        print(
            f"Found {len(images)} image(s) and {len(pdfs)} PDF(s) to process\n",
            flush=True,
        )
        print("=" * 60, flush=True)

        successful = 0
        failed = 0

        for img in images:
            result = processor.process_image(
                str(img), args.context, args.output, args.review
            )
            if result:
                successful += 1
            else:
                failed += 1

        for pdf in pdfs:
            result = processor.process_pdf(
                str(pdf),
                args.context,
                args.output,
                args.review,
                args.max_pages,
                args.start_page,
                export_zotero=args.zotero,
                zotero_title=args.zotero_title,
                zotero_collection=args.zotero_collection,
            )
            if result:
                successful += 1
            else:
                failed += 1

        print("=" * 60, flush=True)
        print(f"\n📊 Processing complete:", flush=True)
        print(f"   ✓ Successful: {successful}", flush=True)
        print(f"   ✗ Failed: {failed}", flush=True)

    else:
        # Single file processing
        if not Path(args.input).exists():
            print(f"❌ Error: File {args.input} not found", flush=True)
            sys.exit(1)

        file_ext = Path(args.input).suffix.lower()
        if file_ext == ".pdf":
            if not PDF_SUPPORT:
                print("❌ Error: PDF support not available", flush=True)
                print("   Install with: pip install pdf2image", flush=True)
                sys.exit(1)
            processor.process_pdf(
                args.input,
                args.context,
                args.output,
                args.review,
                args.max_pages,
                args.start_page,
                export_zotero=args.zotero,
                zotero_title=args.zotero_title,
                zotero_collection=args.zotero_collection,
            )
        else:
            processor.process_image(args.input, args.context, args.output, args.review)

        print("\n✓ Processing complete!", flush=True)


def process_pdf_complex(
    input_path: str,
    output_dir: str,
    cleanup_model: str = "deepseek",
    context: str = "",
    start_page: int = 1,
    end_page: int = None,
    progress_callback=None,
    original_filename: str = None,
) -> str:
    """
    Web service wrapper for complex OCR processing.

    Args:
        input_path: Path to input PDF file
        output_dir: Directory to save output files
        cleanup_model: "deepseek" or "kimi"
        context: Context hint for OCR (e.g., "明代地方志")
        start_page: First page to process (1-indexed)
        end_page: Last page to process (None = all pages)
        progress_callback: Called as progress_callback(stage, page, percent)
                          Stages: "preprocessing", "ocr", "cleanup", "postprocessing"
        original_filename: Original upload filename (used for markdown title)

    Returns:
        Path to output markdown file

    Raises:
        Exception on processing failure
    """
    config = ProcessingConfig(cleanup_model=cleanup_model)
    processor = DocumentProcessor(config)
    processor._web_progress_callback = progress_callback
    processor._original_filename = original_filename
    max_pages = None
    if end_page is not None:
        max_pages = end_page - start_page + 1
    output_path = processor.process_pdf(
        pdf_path=input_path,
        context=context,
        output_dir=output_dir,
        review_mode=False,
        max_pages=max_pages,
        start_page=start_page,
    )
    if output_path is None:
        raise Exception("OCR processing failed - no output generated")
    return str(output_path)


if __name__ == "__main__":
    main()
