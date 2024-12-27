# Dataset: AHQAD Arabic Healthcare Q&A Dataset
## Intoduction
This document provides a concise overview of the **AHQAD** dataset with **csv** format collection, cleaning, and preparation steps for modeling and visualization. 

---

## 1. Dataset

- **Data Source**: Downloaded from `abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset`.
- **Original Shape**: Approximately **808,472** rows and **4** columns (`Unnamed: 0`, `Question`, `Answer`, `Category`).

**Sample Rows**:

| Unnamed: 0 | Question                                                                                                                                                                                                                                                            | Answer                                                                                                        | Category           |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------|
| 49324      | السلام عليكم ..<br>اصبت منذ ٣ اسابيع بالتهاب لوزتين حاد والتهاب الجيوب والمجاري التنفسيه واستمر معي المرض ٣ اسابيع حتى بدأت ارتاح لكن اليوم ظهرت اعراض خدر ووخز بالجانب الايسر...                                                                                       | يفضل مراجعة الطبيب و فحص الاذن و البلعلوم . الاعراض العصبية                                                   | أنف، أذن وحنجرة   |
| 37627      | سلام عليكم ورحمة الله أريد تفسير لخالتي أحس بالارهاق والالم في المفاصل مع ضهور كدمات بلون ازرق وشعيرات دموية مع ظهور حبوب حمراء في كعين مع حكة ما سبب في...                                                                                                             | يرجى مراجعة طبيب باطني لعمل الفحوصات اللازمة وتقييم الحالة وإعطاء الدواء المناسب.                               | الطب العام         |
| 2271       | ابني منذ الولادة كانت الغدة الدرقية كسولة وتعالج منها على عمر الثلاث سنوات ومن ثم اختلت على عمر الست سنوات ونتيجة t4طبيعية وtsh فوق المعدل الطبيعي...ممكن تشرح لنا السبب وهل...                                                                                    | نعم بحاجة للعلاج وزيارة اختصاصي غدد صماء أطفال.سلامته                                                        | أمراض الأطفال      |
| 2216       | انا بديت بحبوب سيرازيت باليوم ال٣٠ من بعد الولاده وبعد ٥ حبات تاخرت ٣ ساعات بحبه اليوم السادس السؤال هل ينبغي بعد انتهاء فتره الاربعين اخذ احتياطات ؟                                                                                                                | لاداعي للحتياط                                                                                               | أمراض نسائية      |
| 23359      | عملت تحليل سريع الجيل الرابع في يوم 46 هل المدة كافية وطلع سلبي ،وطلع عندي قرحتين في الشفاة من الداخل هل عندهم علاقة                                                                                                                                                | ليس بالضرورة وجود علاقة ، ولكن المدة غير كافية وتحتاج ل ٣ أشهر لتكون نهائية!!                                   | الصحة الجنسية      |

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
   - Link: [AHQAD Dataset](https://www.kaggle.com/datasets/abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset/data)

2. **Data Cleaning**  
   - Removed irrelevant columns, handled nulls and duplicates, and normalized Arabic text.  

3. **Data Preparation**  
   - Created combined text for input and formatted answers. Split data into train, test, and validation sets.  

4. **Visualization**  
   - Generated word clouds to identify frequently used words in questions and answers.

This dataset is now ready for building Arabic healthcare question-answer models.
