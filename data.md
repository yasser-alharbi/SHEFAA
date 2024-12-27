# Dataset Summary
## Dataset: AHQAD Arabic Healthcare Q&A Dataset

This document provides a concise overview of the **AHQAD** dataset collection, cleaning, and preparation steps for modeling and visualization. 

---

## 1. Dataset

- **Data Source**: Downloaded from `abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset`.
- **Original Shape**: Approximately **808,472** rows and **4** columns (`Unnamed: 0`, `Question`, `Answer`, `Category`).

**Sample Rows**:

| Unnamed: 0 | Question                                                                                         | Answer                                                                                                                                           | Category                             |
|------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| 211833     | اذا كان الغشاء به كدمات يعني متصدع لكنه سليم و...                                                 | ما دام الغشاء موجود ، فأنت عذراء!!!التشقق لا ي...                                                                                                  | الصحة الجنسية                        |
| 674769     | متزوجة عمري 18دورتي كانت منتضمة قبل الزواج وبع...                                                | اختى الفاضله\nمن المؤكد انكى تعانين من لخبطة ب...                                                                                                 | الحمل والولادة                       |
| 513034     | طفل عند شهرين اخد طعيم الخماسي ياخد اية علشان ...                                                | باراسيتامول تحاميل او شراب عند اللزوم.سلامته                                                                                                      | صحة الطفل                            |
| 687216     | انا كنت امارس العادة السرية كل يوم مرة او مرتي...                                                | يرجى فحص السائل المنوي وعمل زراعة للبول والسائ...                                                                                                 | أمراض المسالك البولية والتناسلية    |
| 531929     | السلام عليكم ابي نظام غذائي مناسب لخساره الوزن...                                                | اهلا بك.... عزيزي عليك باتباع نظام غذائي يحتوي...                                                                                                 | تغذية                               |

---

## 2. Data Cleaning

1. **Drop Unnecessary Features**  
   - Removed the `Unnamed: 0` column.

2. **Handle Missing Values**  
   - Dropped rows with null questions or answers.

3. **Remove Duplicates**  
   - Dropped rows where `Question` or `Answer` were repeated.

4. **Normalize Text**  
   - Trimmed leading/trailing spaces.  
   - Replaced newline characters with spaces.  
   - Removed square-bracketed text.  
   - Removed punctuation and special characters.  
   - Replaced **أ**/**إ** with **ا** for consistency.  

The final cleaned data shape was approximately **715,187** rows and **3** columns (`Question`, `Answer`, `Category`).

---

## 3. Data Preparation

1. **Combine Question and Category**  
   - Created a new column `input_text` using the format:  
     `سؤال: {Question} | التصنيف: {Category}`

2. **Format Answers**  
   - Prefixed each answer with:  
     `الإجابة: {Answer}`

3. **Train/Test/Validation Split**  
   - **Train**: 80% of the data  
   - **Test**: 10% of the data  
   - **Validation**: 10% of the data

---

## 4. Data Visualization

To understand the dataset’s most frequent words:

- **Word Cloud for Questions**  
  Displays the top terms frequently mentioned in the **Question** column.

- **Word Cloud for Answers**  
  Displays the most common terms appearing in the **Answer** column.

Below is a sample visualization of the word clouds:
- Word Cloud for Questions:
  ![WordCloud Q](https://github.com/user-attachments/assets/65c726c6-7a12-475f-91cd-c9fad423cebb)

- Word Cloud for Answers:
  ![WordCloud A](https://github.com/user-attachments/assets/56f51883-8232-4984-b0da-5eae2838b6ea)



---

## 5. Summary

1. **Data Collection**  
   - Downloaded from Kaggle.
   - Link: https://www.kaggle.com/datasets/abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset/data
2. **Data Cleaning**  
   - Removed irrelevant columns, handled nulls and duplicates, and normalized Arabic text.  
3. **Data Preparation**  
   - Created combined text for input and formatted answers. Split data into train, test, and validation sets.  
4. **Visualization**  
   - Generated word clouds to identify frequently used words in questions and answers.

This dataset is now ready for building Arabic healthcare question-answer models.

