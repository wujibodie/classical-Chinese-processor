This is a simple script that turns PDF scans of Ming-Qing era Classical Chinese documents into clean, readable, searchable markdown files (easy to convert to any kind of text file). 
It does this by leveraging the current best OCR engine for classical Chinese, Qwen3-vl-plus, with specific instructions suitable for right-left, vertical column printed text. The raw OCR output is polished by another model, DeepSeek V3.2 (you can configure it to use other models easily as well).

The full process is: input PDF -> PDF is split into single page PNGs -> send to Qwen for OCR -> output raw OCR as one combined file -> chunk raw output into manageable blocks -> send to DeepSeek for editing -> recombine output into a single .md file and export.

There is also a simplified version of this script, called "simpleprox.py," which is intended for modern, punctuated reprints of Chinese texts. If your PDF already includes punctuation you can use this text to skip the second stage of processing, since Qwen can handle everything on its own, making the process faster and cheaper. In theory this script could probably be applied to any language with a little tweaking to Qwen's prompt.

The price of the whole process averages around $0.01 per page, and the final results are better than you'll generally find anywhere else for obscure texts, if your scan is good quality. It won't give 100% accurate results, but it normally hits 90+% of characters correct in my tests. This is generally high enough for the full context to be clear, although you'll want to double check sections that are important to you against the original scan. Also, it is possible for whole chunks of text to get corrupted or lost, but this is rare unless the scanned text is in an unusual format or somehow damaged. In such cases the damage is normally limited to single pages or less.

In the most recent updates I've managed to speed up OCR processing to just a few minutes per bundle of 20-30 pages. In general a full book should finish in under half an hour. The log will let you know how things are going and roughly how long everything is taking. It's normally safe to leave the program alone until it finishes, but you might want to keep an eye on it to see if any errors pop up. 

There are various commands listed below to change how the script processes your file(s). You can also poke around in the configs inside the .py.

YOU WILL NEED
1. AliCloud and DeepSeek API keys, as well as sufficient funds to use them.
     - If you don't have these, go [here](https://www.alibabacloud.com/help/en/model-studio/get-api-key) and [here](https://api-docs.deepseek.com/) and follow the steps. Note that for Qwen, you have the choice between a Chinese or international (Singapore) API key. The Chinese key will be cheaper, but requires a Chinese phone number and ID verification. My script assumes you are using the international version, but you can easily adjust it to make calls to the Chinese servers. DeepSeek doesn't distinguish and also won't require ID verification. 
     - I've implemented basic Zotero integration, so if you want to export directly to your Zotero account, you'll also need to set your Zotero API and user ID as environmental variables. 
3. Basic familiarity with running python scripts in terminal.
     - Make sure you have Python installed, as well as all necessary dependencies: pdf2image, openai, and requests. Non-Windows users may need to separately install Poppler as well (it installs with Pdf2Image on PC)
4. A GOOD PDF scan of your document. If Qwen can't make any sense of your document things will break down. If it's not legible to you it likely won't be to the AI.
   - If you want to test it, you can upload 1-5 pages of your PDF in a chat with the OCR-bot [here]([https://modelstudio.console.alibabacloud.com/?tab=dashboard#/efm/prompt?modelId=qwen3-vl-plus](https://modelstudio.console.alibabacloud.com/ap-southeast-1/?tab=dashboard#/efm/model_experience_center/vision))
   - There are several processes built in to clean up OCR mistakes, but the bulk of the text needs to be usable for those to work!
  
### **SETUP**

1. **Install Python** (if you don't have it) and Poppler if not on PC. 

**Linux:**
```bash
sudo apt install poppler-utils
```

**Mac:**
```bash
brew install poppler
```

2. **Install this script and necessary packages**:
    ```bash
    pip install pdf2image opencv-python openai requests numpy PyPDF2
   git clone [https://github.com/wujibodie/classical-Chinese-processor]
    ```
3. **Set up API keys**
    ```bash
   export DASHSCOPE_API_KEY="your_alibaba_key"
   export DEEPSEEK_API_KEY="your_deepseek_key"
    ## IF USING ZOTERO EXPORT
   export ZOTERO_API_KEY="your_zotero_key"
   export ZOTERO_USER_ID="your_user_ID"
    ```
   (Windows users replace "export" with "set")
4. **Run the script**
    ```bash
     python3 cc-prox.py yourdocument.pdf --output ./processed
    ```

COMMANDS (Run these in terminal afer the script name)

(Example: python3 cc-prox.py document.pdf --context "明代地方志" --output ./results)
- context "xx": Passes contextual info to LLMs
- output: choose output directory
- batch: process all files in a directory (replace input w/ this); while order output by filename order
- dpi: choose DPI for PDF conversion (default is 300)
- model: choose Qwen model, default is qwen3-vl-plus.
- cleanup-model: change cleanup model. script comes with DeepSeek enabled by default, Kimi K2.5 as backup, but you can add your own preferred model.
- max-pages: limit processing to first N pages
- start-page: start processing from page N
- resume-from: start processing from raw OCR JSON file (in case a run was interrupted).
- zotero: exports to your Zotero account (you need to define your account API as environment variable in your system)
- zotero-title: if exporting to Zotero, choose a title (defaults to filename)
- zotero-collection: choose a collection to export to. 

TROUBLESHOOTING

"No API keys found":
- Make sure you set the environment variables　（your API keys)
- Restart your terminal after setting them
- Remember you can also use Kimi K2 (Moonshot) instead of DeepSeek, but you'll have to use a command or edit the configs.

"No images were converted":
- Check that pdf2image is installed: pip install pdf2image
- On Mac: brew install poppler

Poor OCR quality:
- Test a page first with --max-pages 1
- Make sure --context is set and accurate
- Compare raw OCR output (JSON file in your output folder) to the final product to see whether the issue is at the first OCR step or in the post-processing.
- Consider altering the OCR configs in the script under ProcessingConfig; by default dpi is set to 300 and max image width to 3072 pixels. If you're feeding it very large PDF pages, the compression might damage their legibility. Note that higher pixel count will increase costs.

Pages missing:
- Sometimes this is due to timeouts with the API calls to Alibaba servers. You should see this in the log: you'll see timeouts and several retries. In that case, just rerun the page and it should work.
- Very occasionally the API will refuse to output a given chunk of text due to some string registering as "sensitive" and getting censored. This seems to be a problem with DeepSeek more than other services. If you have trouble with this, try switching providers to Gemini (more expensive but less strict content filters). 
