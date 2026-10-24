# Egyptian Civil Code Q&A

**Methodology record: data, models, files, and how each was used**

Osman Osama · Draft · 15 October 2026

> **About this document.** It records what was built, with which files, models and settings. Cells marked **[FILL]** are placeholders for numbers from the final full run; no result has been invented here. Sections 3 to 5 answer "which file does what and which model does it use"; sections 6 to 10 give the method details.

# 1. What the system does

The system answers questions about the Egyptian Civil Code (Law 131 of 1948) in formal Arabic. The question can be in formal Arabic or Egyptian colloquial, typed or placed inside a Word or PDF file. It works in two stages:

1.  **Retrieve:** find the most likely article in the whole code.

2.  **Generate:** a fine-tuned Llama 3.1 8B writes the answer using only that article.

```text
Civil Code (.docx) → parse → articles JSON
articles JSON → Qwen2.5-7B-Instruct → synthetic QA pairs
QA pairs → clean, filter, split by article → QLoRA fine-tune Llama 3.1 8B → adapter
Question → bge-m3 + BM25 retrieval → top article → Llama + LoRA → answer
```

Why retrieval: the model answers from the text of one article, so it must be given the right article. Why synthetic data: the project needed article-grounded question and answer pairs for the whole code, so they were generated and then filtered.

# 2. What was done, in order

1.  **Pilot on Jordanian law.** A Llama 3.1 8B LoRA was evaluated on a Jordanian civil-law QA dataset (4_final_qa.json, 5,869 entries). This is a separate pilot; the Egyptian pipeline does not use it.

2.  **Moved to Egyptian civil law.** The model should read a question from a Word or PDF file and answer it; the work was then moved to the Egyptian Civil Code.

3.  **Retrieval added.** The dataset covers only articles that had questions written for them, so retrieval over the whole code was added.

4.  **Corpus built.** The Egyptian Civil Code Word file was parsed into a clean article corpus.

5.  **QA generated.** Question and answer pairs were written with Qwen2.5-7B-Instruct, with no paid API.

6.  **Fine-tuned and evaluated.** Llama 3.1 8B was fine-tuned with QLoRA and tested on articles it never saw.

7.  **Full pipeline evaluated** (retrieval plus generation) against baselines, with confidence intervals.

8.  **Streamlit app built** to test the system by hand and to record human ratings.

**Status to confirm before writing:** the full-size run of every notebook [FILL: done or not]; the human evaluation (not yet done); a hand-written gold test set (not yet done). Early pipeline checks used a small test-run subset (12 questions from 4 articles), which is too small to report.

# 3. How the files connect

The notebooks do not connect automatically. Each runs in its own Colab session, so you move files by downloading from one and uploading to the next.

| **Step**                            | **Upload to Colab**                                          | **Produces**                                                    | **Passed on**                                  |
|-------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------|
| **Notebook 0 (generate QA)**        | egyptian_civil_code_articles.json                            | egyptian_civil_qa.json, qa_progress.jsonl                       | QA file goes to Notebooks 1 and 2              |
| **Notebook 1 (retrieval)**          | articles JSON + egyptian_civil_qa.json                       | retrieval_results.csv                                           | Note the best retrieval setting for Notebook 3 |
| **Notebook 2 (train and evaluate)** | egyptian_civil_qa.json                                       | egyptian_lora_model.zip, egyptian_law_llama_evaluation.csv      | The zip goes to Notebook 3 and the app         |
| **Notebook 3 (RAG pipeline)**       | egyptian_civil_code_articles2.json + egyptian_lora_model.zip | rag_pipeline_results.csv, rag_pipeline_confidence_intervals.csv | Final results                                  |
| **App (launcher notebook)**         | app.py + articles JSON + egyptian_lora_model.zip             | validation_log.csv                                              | Human ratings for the paper                    |

# 4. Which model each file uses

## 4.1 Usage matrix

| **File**                           | **Qwen2.5-7B** | **Llama 3.1 8B base**   | **Llama + LoRA**     | **bge-m3** | **BM25** | **MiniLM** |
|------------------------------------|----------------|-------------------------|----------------------|------------|----------|------------|
| **Notebook 0: generate QA**        | Yes            | \-                      | \-                   | \-         | \-       | \-         |
| **Notebook 1: retrieval**          | n/r            | n/r                     | n/r                  | n/r        | n/r      | n/r        |
| **Notebook 2: train and evaluate** | \-             | Yes (trained)           | Yes (tested)         | \-         | \-       | Yes        |
| **Notebook 3: RAG pipeline**       | \-             | Yes (adapter off)       | Yes                  | Yes        | Yes      | Yes        |
| **app.py**                         | \-             | Yes (optional compare)  | Yes                  | Yes        | Yes      | \-         |
| **Streamlit launcher notebook**    | \-             | \-                      | \-                   | \-         | \-       | \-         |
| **Jordanian pilot notebook**       | \-             | Yes (under the adapter) | Yes (Jordanian LoRA) | \-         | \-       | Yes        |

n/r = notebook not reviewed for this record. The launcher notebook only starts app.py; the models are loaded by app.py. Data files (the Word file, the articles JSON, the QA JSON, the CSVs) use no model.

## 4.2 The models

| **Model**               | **Hugging Face ID**                                         | **Loaded with**             | **Precision**                         | **Role**                                               | **Trained?**            |
|-------------------------|-------------------------------------------------------------|-----------------------------|---------------------------------------|--------------------------------------------------------|-------------------------|
| **Qwen2.5-7B-Instruct** | Qwen/Qwen2.5-7B-Instruct                                    | transformers + bitsandbytes | 4-bit NF4, double quant, fp16 compute | Writes the synthetic QA pairs                          | No                      |
| **Llama 3.1 8B (base)** | unsloth/Meta-Llama-3.1-8B-bnb-4bit                          | Unsloth                     | 4-bit                                 | Base for fine-tuning; baseline when the adapter is off | No                      |
| **LoRA adapter**        | egyptian_lora_model.zip                         | Unsloth / PEFT              | Adapter on the 4-bit base             | The fine-tuned part that answers questions             | Yes                     |
| **BGE-M3**              | BAAI/bge-m3                                                 | sentence-transformers       | fp16                                  | Dense retrieval of the article                         | No                      |
| **BM25**                | rank_bm25 (BM25Okapi)                                       | rank_bm25                   | \-                                    | Keyword retrieval, fused with bge-m3                   | No (not a neural model) |
| **Multilingual MiniLM** | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers       | default                               | Semantic-similarity metric only                        | No                      |

# 5. File-by-file reference

### 5.1 Civil Code Word file

|                 |                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| **File**        | القانون المدنى المصرى رقم 131 لسنة 1948 طبقآ لاحدث التعديلات (.docx)                                          |
| **Type**        | Source document                                                                                               |
| **Purpose**     | The text of the Egyptian Civil Code (Law 131 of 1948) as amended. It is the only legal source in the project. |
| **Models used** | None. It is parsed with python-docx.                                                                          |
| **Produces**    | The article corpus (5.2).                                                                                     |
| **To add**      | [FILL: where it was obtained and the date of the amendments it includes, to cite it]                        |

### 5.2 egyptian_civil_code_articles2.json

|                 |                                                                                                                         |
|-----------------|-------------------------------------------------------------------------------------------------------------------------|
| **Type**        | Data: a JSON list. The earlier egyptian_civil_code_articles.json has identical content.                                 |
| **Purpose**     | The clean article corpus. It is the knowledge base that retrieval searches and the source text for question generation. |
| **Fields**      | article (the number) and context (the text, starting with 'المادة N').                                                  |
| **Size**        | 1,093 articles numbered from 1 to 1149. The 56 absent numbers are repealed ranges (54 to 80 and 389 to 417).            |
| **Models used** | None. Its text is indexed by bge-m3 and BM25.                                                                           |
| **Used by**     | Notebook 0 (input), Notebook 1, Notebook 3 (retrieval corpus and to find the correct article), app.py.                  |

### 5.3 Egyptian_Law_0_Generate_QA_no_API.ipynb

|                   |                                                                                                                                                                                                                                                          |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Type**          | Colab notebook (needs a GPU)                                                                                                                                                                                                                             |
| **Purpose**       | Builds the synthetic QA dataset, three pairs per article: two in formal Arabic and one in Egyptian colloquial.                                                                                                                                           |
| **Model used**    | Qwen/Qwen2.5-7B-Instruct, 4-bit NF4, through transformers and bitsandbytes. No other model.                                                                                                                                                              |
| **Input**         | egyptian_civil_code_articles.json, uploaded from a local machine (the code expects that file name).                                                                                                                                                          |
| **Output**        | egyptian_civil_qa.json (egyptian_civil_qa_test.json in a test run) and a progress file, qa_progress.jsonl, used to resume.                                                                                                                               |
| **Key settings**  | TEST_RUN (every 45th article, about 25, for a quick check); batch size 8; up to 900 new tokens; greedy decoding first, then sampling (temperature 0.5, top-p 0.9) for articles that failed; repetition penalty 1.05; articles of 60 to 3,000 characters. |
| **Quality rules** | Three hand-written examples (articles 147, 163, 374); questions must be understandable without the article; a filter removes questions that mention 'this article' or an article number.                                                                 |

### 5.4 egyptian_civil_qa.json

|                 |                                                                            |
|-----------------|----------------------------------------------------------------------------|
| **Type**        | Data: a JSON list                                                          |
| **Fields**      | id, question, context (the article), answer, style (formal or colloquial). |
| **Models used** | None; it was generated by Qwen (5.3).                                      |
| **Used by**     | Notebook 1 (to test retrieval) and Notebook 2 (training data).             |
| **Caution**     | Synthetic. Check a random sample by hand before trusting results.          |

### 5.5 Retrieval notebook (Notebook 1)

|                            |                                                                                                                                |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Type**                   | Colab notebook                                                                                                                 |
| **Status**                 | Not reviewed for this record.                                                                                                  |
| **Known from other files** | It produces retrieval_results.csv, and Notebook 3 offers three retrieval settings (dense bge-m3, BM25, hybrid) chosen from it. |
| **To add**                 | [FILL: the methods it compared, the metrics, and which one won]                                                              |

### 5.6 Egyptian_Law_2_Train_and_Evaluate_Llama_FIXED.ipynb

|                 |                                                                                                                                                                              |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Type**        | Colab notebook (needs a GPU)                                                                                                                                                 |
| **Purpose**     | Cleans the QA data, splits it by article, fine-tunes Llama 3.1 8B with QLoRA, and evaluates it on held-out articles with the correct article given.                          |
| **Models used** | unsloth/Meta-Llama-3.1-8B-bnb-4bit (trained with a LoRA adapter, then used to generate the test answers) and paraphrase-multilingual-MiniLM-L12-v2 (similarity metric only). |
| **Libraries**   | Unsloth, TRL SFTTrainer, rouge_score, NLTK (BLEU), sentence-transformers.                                                                                                    |
| **Input**       | egyptian_civil_qa.json                                                                                                                                                       |
| **Output**      | egyptian_lora_model.zip and egyptian_law_llama_evaluation.csv.                                                                                                               |
| **Metrics**     | Exact match, ROUGE-1/2/L, BLEU, semantic similarity (match at 0.75), and answer_in_article.                                                                                  |
| **Settings**    | See section 7.                                                                                                                                                               |

### 5.7 egyptian_lora_model.zip

|                 |                                                                                                                 |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| **Type**        | Model artifact                                                                                                  |
| **Contents**    | The LoRA adapter weights and configuration, the tokenizer files, and test_split.json (the held-out test pairs). |
| **Purpose**     | The trained model. It is loaded on top of the 4-bit Llama base and fits only that base.                         |
| **Models used** | It is the fine-tuned part of Llama + LoRA.                                                                      |
| **Used by**     | Notebook 3 and app.py.                                                                                          |

### 5.8 egyptian_law_llama_evaluation.csv

|             |                                                                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Type**    | Results table from Notebook 2                                                                                                                                             |
| **Columns** | article, style, question, context, reference_answer, generated_answer, exact_match, rouge1, rouge2, rougeL, bleu, semantic_similarity, answer_in_article, semantic_match. |
| **Meaning** | Quality of the model when the correct article is given (an upper bound for the full system).                                                                              |

### 5.9 Egyptian_Law_3_RAG_Pipeline.ipynb

|                  |                                                                                                                                                                              |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Type**         | Colab notebook (needs a GPU)                                                                                                                                                 |
| **Purpose**      | Tests retrieval and generation together on articles the model never saw, in several modes and against baselines.                                                             |
| **Models used**  | bge-m3 (dense retrieval); BM25 (keyword retrieval); Llama + LoRA (answers); the same Llama with the adapter switched off (base_oracle baseline); MiniLM (similarity metric). |
| **Input**        | egyptian_civil_code_articles2.json and egyptian_lora_model.zip (which contains test_split.json).                                                                             |
| **Output**       | rag_pipeline_results.csv and rag_pipeline_confidence_intervals.csv.                                                                                                          |
| **Key settings** | Hybrid retrieval with rank fusion; top-50 candidates; top-3 for the top3 mode; N_EVAL 300 on the first pass (None for the final run); seed 42; article-number lookup on.     |
| **Modes**        | oracle, top1, top3, base_oracle, lora_no_article (base_no_article is also available).                                                                                        |
| **Extra**        | An ask() function answers one question the way a user would.                                                                                                                 |

### 5.10 rag_pipeline_results.csv and rag_pipeline_confidence_intervals.csv

|                      |                                                                                                                                                                                                                          |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Type**             | Results tables from Notebook 3                                                                                                                                                                                           |
| **Results columns**  | mode, style, question, gold_article, articles_given, retrieval_hit (the correct article was in the prompt), reference_answer, generated_answer, answer_in_provided, answer_in_gold, rouge1, rougeL, semantic_similarity. |
| **Interval columns** | mode, metric, mean, ci_low, ci_high, n_articles, n_questions (95% bootstrap over articles).                                                                                                                              |
| **Models used**      | None; they are outputs.                                                                                                                                                                                                  |

### 5.11 app.py (the Streamlit app)

|                            |                                                                                                                                                                                                                                                                                                                   |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Type**                   | Streamlit web app (Python)                                                                                                                                                                                                                                                                                        |
| **Purpose**                | Ask a question in Arabic, see the model's answer next to the article it used, and rate whether the answer is right. Every rating is saved.                                                                                                                                                                        |
| **Models used**            | Llama 3.1 8B (4-bit) + the LoRA adapter, loaded with Unsloth, to answer; the same Llama with the adapter off for the optional 'compare with base' view; bge-m3 and BM25 for retrieval. It does not use Qwen or MiniLM.                                                                                            |
| **Input**                  | egyptian_civil_code_articles2.json; egyptian_lora_model.zip (unzipped automatically); optionally a Word or PDF file containing the question (read with python-docx or PyMuPDF).                                                                                                                                   |
| **Output**                 | validation_log.csv, and the on-screen answer.                                                                                                                                                                                                                                                                     |
| **How it works**           | It retrieves the top three articles with rank fusion, puts an article named by number in the question first, and sends the top-1 article (default), the top-3, or a chosen article number to the model. The prompt is identical to training. Decoding is greedy, up to 256 tokens, cut at the next prompt marker. |
| **Checks it shows**        | The share of the answer's words found in the article (a warning appears below 60%), and the top-3 retrieved articles so you can tell a retrieval error from a model error.                                                                                                                                        |
| **Runtime notes**          | Models load once per session; a lock prevents two simultaneous generations; it needs a CUDA GPU. It is a test tool, not a deployment.                                                                                                                                                                             |
| **Relation to Notebook 3** | Same retriever, prompt and adapter; only the interface differs.                                                                                                                                                                                                                                                   |

### 5.12 Egyptian_Law_4_Streamlit_App.ipynb (the Streamlit launcher)

|                 |                                                                                                                                                                                                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Type**        | Colab notebook                                                                                                                                                                                                                                                                 |
| **Purpose**     | Runs app.py on a Colab GPU and exposes it through a public link.                                                                                                                                                                                                               |
| **Models used** | None itself. The models are loaded by app.py when the page opens.                                                                                                                                                                                                              |
| **Steps**       | Install unsloth, streamlit, sentence-transformers, rank_bm25, python-docx and pymupdf; upload app.py, the articles JSON and the adapter zip; start Streamlit on port 8501; start a Cloudflare quick tunnel and print the link; optional log cell; download validation_log.csv. |
| **Notes**       | The link works only while the Colab session runs, and anyone with the link can use the app.                                                                                                                                                                                    |

### 5.13 validation_log.csv

|                      |                                                                                                            |
|----------------------|------------------------------------------------------------------------------------------------------------|
| **Type**             | Human-rating record written by the app                                                                     |
| **Columns**          | time, question, articles_given, retrieved_top3, mode, answer, answer_in_article, rating, note.             |
| **Ratings**          | Correct and supported; partly correct or incomplete; wrong or not in the article; wrong article retrieved. |
| **Use in the paper** | The human evaluation table and the error analysis.                                                         |

### 5.14 Jordanian pilot: 4_final_qa.json and the Jordanian notebook

|                 |                                                                                                                                                                                                                                                                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Files**       | 4_final_qa.json and Llama_3_1_8B_Jordanian_Law_LoRA_Evaluation_ONLY.ipynb                                                                                                                                                                                                                                                                         |
| **Data**        | 5,869 entries (id, question, context, answer) from Jordanian civil law.                                                                                                                                                                                                                                                                           |
| **Notebook**    | Evaluation only; it does no training. It loads an adapter trained earlier (lora_model.zip) with Unsloth on the Llama 3.1 8B base recorded in the adapter configuration, recreates a random 80/20 split of the rows (test size 0.2, seed 42), generates greedily, and scores with exact match, ROUGE, BLEU and MiniLM similarity (threshold 0.75). |
| **Models used** | Llama 3.1 8B with the Jordanian LoRA, and paraphrase-multilingual-MiniLM-L12-v2.                                                                                                                                                                                                                                                                  |
| **Caution**     | Its split is by row, so one article can sit in both training and test, while the Egyptian split is by article. Do not compare the two sets of numbers directly.                                                                                                                                                                                   |

# 6. Data

## 6.1 Legal corpus

- **Parsing** (python-docx): an article starts at a header of the form 'مادة N –', or 'مادة N' alone on a line. The body is the text up to the next header. Paragraphs without a header (chapter titles, amendment notes, repeal notices) are dropped.

- **Cleaning:** two watermark strings were removed; the first two articles (the articles of the issuing law, which reuse the numbers 1 and 2) were skipped; one article numbered 922 twice in the source was corrected to 992, because it sits between 991 and 993.

- **Output:** 1,093 articles. Average length is about 245 characters and the longest is 1,417. The corpus has no chapter-heading field.

## 6.2 Synthetic question and answer generation

- **Prompt:** an Arabic system prompt plus three hand-written examples given as earlier chat turns.

- **Output per article:** 3 pairs, two formal Arabic and one simple Egyptian colloquial. Answers are always formal Arabic, one or two sentences, faithful to the article wording and taken only from it. The reply must be a JSON list.

- **Self-contained questions:** the prompt forbids phrases such as 'this article' and article numbers inside the question. This rule was added after the early check showed questions that could not be answered or retrieved without seeing the article.

- **Validity checks:** a JSON list; non-empty question and answer; question different from answer; style either formal or colloquial; at most 3 pairs.

## 6.3 Cleaning before training

1.  Records missing a question, answer or context are dropped.

2.  Questions that cannot stand alone are dropped (they mention this article, this text, or an article number).

3.  Duplicates are dropped (same article and same normalised question).

4.  Answers whose words are mostly absent from the article are dropped: at least 40% of the answer's words must appear in the article.

5.  Examples longer than 2,048 tokens are dropped.

Word matching uses Arabic normalisation: diacritics and tatweel removed, alef variants unified, ya and ta marbuta unified, Arabic-Indic digits converted, punctuation removed, and common attached prefixes stripped.

Counts to report: total generated pairs [FILL]; self-referential [FILL]; duplicates [FILL]; unfaithful [FILL]; too long [FILL]; final training pairs [FILL].

## 6.4 Train, validation and test split

- The split is **by article**: all questions about one article stay in the same split.

- Unique articles are shuffled with seed 42. 20% go to test, 10% of the rest to validation, the remainder to training. An assertion fails if an article appears in two splits.

- The test pairs are saved inside the adapter zip (test_split.json) so later notebooks evaluate on exactly the same unseen articles.

- Sizes: training [FILL] pairs / [FILL] articles; validation [FILL] / [FILL]; test [FILL] / [FILL].

# 7. Model and training

## 7.1 Prompt format

Training and inference use the same Alpaca-style template. The instruction says: a question about a law in Egypt (the Egyptian Civil Code, Law 131 of 1948), paired with an input that gives the text of the relevant article; write an answer in formal Arabic using only the article. The sections are \### Question:, \### Input: (the article) and \### Answer:. An end-of-sequence token is appended during training.

## 7.2 QLoRA configuration

| **Setting**                     | **Value**                                   |
|---------------------------------|---------------------------------------------|
| **Base model**                  | unsloth/Meta-Llama-3.1-8B-bnb-4bit (4-bit)  |
| **Library**                     | Unsloth with TRL SFTTrainer                 |
| **LoRA rank / alpha / dropout** | 16 / 16 / 0                                 |
| **Target modules**              | q, k, v, o, gate, up and down projections   |
| **Max sequence length**         | 2,048 tokens                                |
| **Epochs**                      | 3                                           |
| **Learning rate**               | 2e-4, linear schedule, warm-up ratio 0.05   |
| **Batch size**                  | 2 per device x 4 accumulation = effective 8 |
| **Optimiser**                   | AdamW 8-bit, weight decay 0.01              |
| **Precision**                   | bf16 if the GPU supports it, otherwise fp16 |
| **Seeds**                       | split 42; training 3407                     |
| **Loss**                        | Answer tokens only                          |

## 7.3 Procedure

- The loss is computed only on the answer tokens, not on the copied article. An automatic check confirms the masking worked.

- Validation loss is computed each epoch, a checkpoint is saved each epoch, and the epoch with the lowest validation loss is kept.

- The adapter, tokenizer and test_split.json are saved and zipped as egyptian_lora_model.zip.

- Evaluation on held-out articles uses greedy decoding (max 256 new tokens), so results are reproducible.

- Training time [FILL]; best validation loss [FILL]; GPU [FILL].

# 8. Retrieval

- **Dense:** every article is embedded with bge-m3 (max length 1,024, fp16, normalised) and compared with the question by cosine similarity.

- **Keyword:** BM25 over the normalised, prefix-stripped tokens of each article.

- **Hybrid:** the two top-50 lists are merged by reciprocal rank fusion (k = 60). The setting used is the one chosen in the retrieval notebook. [FILL: methods compared and which won]

- **Article-number lookup:** if the question names an article number, that article is placed first.

- The **top-1 article** is given to the model, because it was trained with one article per question.

- **Metrics:** hit@1, hit@3 and hit@50 on the held-out questions, split by formal and colloquial.

# 9. End-to-end evaluation (Notebook 3)

All modes use the same held-out questions from articles never seen in training.

| **Mode**            | **Article given**         | **Model**          | **Question it answers**            |
|---------------------|---------------------------|--------------------|------------------------------------|
| **oracle**          | The correct article       | Llama + LoRA       | Upper bound with perfect retrieval |
| **top1**            | Best retrieved article    | Llama + LoRA       | The real system                    |
| **top3**            | 3 best retrieved articles | Llama + LoRA       | Does more context help or hurt?    |
| **base_oracle**     | The correct article       | Llama without LoRA | Did fine-tuning help?              |
| **lora_no_article** | None                      | Llama + LoRA       | Is the article actually needed?    |

- **Metrics:** ROUGE-1, ROUGE-2 and ROUGE-L with an Arabic tokeniser (normalised, whitespace split); sentence BLEU with smoothing; exact match after normalisation.

- **Semantic similarity:** cosine similarity of MiniLM embeddings between answer and reference; a pair counts as a semantic match at 0.75.

- **Faithfulness proxies:** the share of the answer's words found in the article. answer_in_article (Notebook 2) and answer_in_gold (Notebook 3) are comparable across modes; answer_in_provided depends on how much text was given and is not comparable between top1 and top3.

- **Statistics:** 95% confidence intervals from a bootstrap over articles (2,000 resamples), because questions from one article are not independent; paired differences between modes use the same bootstrap.

- **Sampling:** a first pass evaluates about 300 questions with whole articles kept together; the final run uses all held-out questions. [FILL: which setting the reported numbers use]

# 10. Interactive app and human validation

- app.py uses the same retriever, prompt and adapter as Notebook 3. The user types a question or uploads a Word or PDF file, sees the answer next to the article it used and the top-3 retrieved articles, and can compare with the base model.

- It records a rating for each answer in validation_log.csv: correct and supported; partly correct or incomplete; wrong or not in the article; or wrong article retrieved. The last label separates retrieval failures from generation failures.

- **Protocol to run (not yet done):** rate 50 to 100 answers, ideally with a second rater such as a law student, and report agreement. Rate against the article text, not against the generated reference answer.

- **Also planned:** 50 to 100 questions written by hand (not by Qwen), with the correct article and answer, run through the whole pipeline.

# 11. Results to report (fill from the full run)

| **Table**              | **Contents**                                                                              | **Status** |
|------------------------|-------------------------------------------------------------------------------------------|------------|
| **Dataset**            | Pairs and articles per split; style share; removal counts                                 | [FILL]   |
| **Retrieval**          | hit@1, hit@3, hit@50; formal vs colloquial                                                | [FILL]   |
| **Answer quality**     | ROUGE-L, semantic similarity, answer_in_gold with confidence intervals, per mode          | [FILL]   |
| **Paired differences** | oracle - base_oracle; oracle - top1; top1 - top3; oracle - lora_no_article                | [FILL]   |
| **Human evaluation**   | Percent supported, correct, complete; rater agreement                                     | [FILL]   |
| **Gold test set**      | Same metrics on the hand-written questions                                                | [FILL]   |
| **Error analysis**     | Retrieval miss, wrong paragraph, multi-article question, colloquial or ambiguous question | [FILL]   |

The small test-run numbers (12 questions, 4 articles) should not appear in the paper.

# 12. Limitations to state

1.  The training and test pairs are synthetic, written by a 7B open model that is weaker at Egyptian colloquial, so the test set may be easier than real questions.

2.  Reference answers are generated and can be wrong or cover a different paragraph of the same article; ROUGE and similarity scores measure closeness to them, not legal correctness.

3.  Only the civil code is covered. Repealed articles are absent. Questions that need several articles or cross-references are a known weak point.

4.  The model was trained with one article per question; giving three was not part of training.

5.  Retrieval errors dominate the end-to-end result, so oracle scores overstate real performance.

6.  Colloquial questions are harder to match to formal legal text.

7.  This is a research prototype and not legal advice.

# 13. Reproducibility checklist

| **Item**                                                                       | **Value**                                                                                      |
|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **GPU and runtime**                                                            | [FILL]                                                                                       |
| **Library versions (unsloth, transformers, trl, peft, sentence-transformers)** | [FILL: from pip list]                                                                        |
| **Seeds**                                                                      | split 42; training 3407; evaluation sampling 42                                                |
| **Decoding**                                                                   | Qwen greedy, then sampling (temperature 0.5, top-p 0.9) on retry; Llama greedy, 256 new tokens |
| **Code and data availability**                                                 | [FILL: repository link, whether the data will be shared]                                     |

# 14. Disclosure and ethics

- State that the question and answer pairs were generated by Qwen2.5-7B-Instruct and checked by hand (describe the sample size once the check is done).

- AI assistance: the notebooks and this record were prepared with help from Claude (Anthropic). Follow your university or journal policy on disclosing it.

- Cite the Civil Code source, and check the licences of Llama 3.1 and Qwen2.5 before sharing the adapter or data.

- Say clearly that the system is for research and does not replace legal advice.
