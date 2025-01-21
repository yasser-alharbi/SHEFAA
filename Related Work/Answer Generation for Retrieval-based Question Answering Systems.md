# Answer Generation for Retrieval-based Question Answering Systems

**Paper Reference:**
[Hsu, Chao-Chun et al., *“Answer Generation for Retrieval-based Question Answering Systems.”* (ACL-IJCNLP 2021)](https://aclanthology.org/2021.findings-acl.374.pdf)

---

## Overview
This paper addresses **Answer Sentence Selection (AS2)** in QA but proposes a new approach called **GenQA**: instead of selecting the single top candidate sentence, they **generate** a final answer from multiple top-ranked sentences. The motivation is to fix cases where the best single candidate is either incomplete or unnatural.

---

## Dataset 
They test on:
1. **WikiQA** (Bing queries + Wikipedia)
2. **ASNQ** (subset of Natural Questions)
3. **WQA** (internal dataset from a commercial assistant)
4. **MSNLG** (MS MARCO’s short human-written answers)

---

## Methodology
1. **AS2 Ranking**  
   - Use TANDA (a top-performing RoBERTa-based ranker) to get top-k sentences for each question.

2. **GenQA**  
   - A seq2seq model (T5 or BART) sees the **question + top-5** retrieved sentences concatenated.
   - It then **generates** a concise, single-sentence answer.

3. **Fine-Tuning**  
   - Train GenQA on MSNLG or WQA data to adapt the model to the style of short, user-friendly answers.
   - Evaluate correctness via human annotation.

---

## Results & Findings
- **GenQA** outperforms the best AS2 approach by up to 32% in accuracy, as judged by human annotators.
- The generative approach fuses partial info from multiple retrieved sentences, yielding shorter, more natural answers.
- Automatic metrics like BLEU and ROUGE did **not** correlate well with correctness—human review is necessary.
