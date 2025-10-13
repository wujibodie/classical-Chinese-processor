This is a simple script that turns PDF scans of Ming-Qing era Classical Chinese documents into clean, readable, searchable markdown files (easy to convert to any kind of text file). 
It does this by leveraging the current best OCR engine for classical Chinese, Qwen-vl-max, with specific instructions suitable for left-right, vertical column printed text. The raw OCR output is polished by another model, Kimi K2 (you can configure it to use DeepSeek easily as well).

The full process is: input PDF -> PDF is split into single page PNGs -> send to Qwen-vl-max for OCR -> output raw OCR as one combined file -> chunk raw output into manageable blocks -> send to Kimi K2 for editing -> recombine Kimi's output into a single .md file and export.

The price of the whole process averages around $0.01 per page, and the final results are better than you'll generally find anywhere else for obscure texts, if your scan is good quality.
In general it takes around 30s-1m per page, so a full book will run for a few hours. The log will let you know how things are going and roughly how long everything is taking. It's normally safe to leave it until it finishes, but you might want to check if any errors pop up. 

There are various commands listed below to change how it processes your file(s). You can also poke around in the configs inside the .py.

YOU WILL NEED
1. AliCloud and Moonshot API keys, as well as sufficient funds to use them.
2. Basic familiarity with running python scripts in terminal.
     - Make sure you have Python installed, as well as all necessary dependencies: pdf2image, openai, and requests. Non-Windows users may           need to separately install Poppler as well (it installs with Pdf2Image on PC)
3. A GOOD PDF scan of your document. If Qwen can't make any sense of your document (at least ~75-80% accurate) things will break down.
   - If you want to test it, you can upload 1-5 pages of your PDF (converted) in a chat with the OCR-bot here:           https://modelstudio.console.alibabacloud.com/?tab=dashboard#/efm/prompt?modelId=qwen-vl-max
   - There are several processes built in to clean up OCR mistakes, but the bulk of the text needs to be usable for those to work.
  
SETUP

1. **Install Python** (if you don't have it):
   - Download from python.org
   - Or use: `brew install python` on Mac
2. **Install this script**:
    ```bash
    pip install pdf2image opencv-python openai requests
   git clone [https://github.com/wujibodie/classical-Chinese-processor]
   cd classical-Chinese-processor
   python3 cc-prox.py document.pdf --output ./processed
    ```
   
4. **Set up API keys**
    ```bash
   export DASHSCOPE_API_KEY="your_alibaba_key"
   export KIMI_API_KEY="your_moonshot_key"
    ```

COMMANDS (Run these in terminal afer the script name)
(Example: python3 cc-prox.py document.pdf --context "明代地方志" --output ./results)
- context "xx": Passes contextual info to LLMs
- output: choose output directory
- batch: process all files in a directory (replace input w/ this)
- dpi: choose DPI for PDF conversion (default is 200)
- model: choose Qwen model, default is qwen-vl-max, but qwen-vl-plus performs ok and half the price.
- max-pages: limit processing to first N pages
- start-page: start processing from page N
- resume-from: start processing from raw OCR JSON file (in case a run was interrupted).

TROUBLESHOOTING
"No API keys found":
- Make sure you set the environment variables
- Restart your terminal after setting them

"No images were converted":
- Check that pdf2image is installed: pip install pdf2image
- On Mac: brew install poppler

Poor OCR quality:
- Test a page first with --max-pages 1
- Make sure --context is set and accurate
- Compare raw OCR output (JSON file in your output folder) to the final product to see whether the issue is at the first OCR step or in the post-processing.
