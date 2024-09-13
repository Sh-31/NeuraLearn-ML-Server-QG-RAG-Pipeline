# NeuraLearn Academy

NeuraLearn Academy is an innovative educational platform that utilizes generative AI and Large Language Models (LLMs) to elevate the online learning experience. Through AI-driven workflows, the platform offers features such as intelligent feedback, automated question generation, and content summarization, providing value to both instructors and students.

![NeuraLearn Platform Overview](https://github.com/Sh-31/NeuraLearn-ML-Server-QG-RAG-Pipeline/assets/83607474/e528cce7-f259-4519-8be7-fa37fd650e4e)

---

### System Overview

NeuraLearn Academy was developed as my graduation project at the Faculty of Computers and Artificial Intelligence, Fayoum University. The system architecture consists of multiple interconnected servers, communicating via APIs:

- **Backend (Django)**: [GitHub Repository](https://github.com/KhaledMohamed422/NeuraLearn-Platform)
- **Frontend (React)**: [GitHub Repository](https://github.com/FrontEndDevo/NeuraLearnAcademy)
- **Machine Learning (AI Services)**: [GitHub Repository](https://github.com/Sh-31/NeuraLearn-ML-Server-QG-RAG-Pipeline)
- **Project Documentation**: [NeuraLearn_documentaion](https://github.com/MahmoudZaytoni/NeuraLearn-documentaion/blob/main/NeuraLearn_documentaion.pdf)

![System Architecture](https://github.com/user-attachments/assets/944ec84f-5611-4293-8d3c-7492aaac21eb)

---
## Usage
---

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sh-31/NeuraLearn-ML-Server-QG-RAG-Pipeline.git
   ```

2. **Install the required dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the main script**:
   ```bash
   python3 server.py
   ```

---

### Important Notes:

- **Question answer, Summarizer and Transcript Models**:  
  These models utilize Hugging Face's serverless API ([Hugging Face Inference API](https://huggingface.co/inference-api/serverless)) to reduce local system resource usage, making them suitable for demo purposes.
  
- **Question Generation Model**:  
  This model is loaded into the system's memory, requiring at least **6GB of GPU RAM** to function efficiently. If you're limited on resources, I recommend using **Lightning AI**, which offers free GPU usage hours.


# NeuraLearn ML API Documentation

This API provides machine learning services for the NeuraLearn Academy platform, including summarization, transcript generation, question answering, and comparison of answers.

## Base URL
`http://127.0.0.1:8080`

## Endpoints

### 1. **Summarization**
Generate a summary from the provided text input.

- **URL:** `/neuarlearn/ml/summarizer`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "text": "string",
    "max_length": 100,
    "min_length": 50
  }
  ```
  - `text`: The text to be summarized (required).
  - `max_length`: The maximum length of the summary (optional).
  - `min_length`: The minimum length of the summary (optional).

- **Response:**
  ```json
  {
    "summary": "string"
  }
  ```

### 2. **Transcript Generation**
Generate a transcript from an audio file.

- **URL:** `/neuarlearn/ml/transcript`
- **Method:** `POST`
- **Request Body:**
  Raw `audio` bytes.

- **Response:**
  ```json
  {
    "transcript": "string"
  }
  ```

### 3. **Question Answering**
Answer questions based on the provided context and chat history.

- **URL:** `/neuarlearn/ml/chat`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "question": "string",
    "context": "string",
    "k": 5,
    "type": 1,
    "chat_history": ["string"],
    "do_splitting": true,
    "add_to_history": true,
    "chunk_size": 2000,
    "chunk_overlap": 1000
  }
  ```
  - `question`: The question to be answered (required).
  - `context`: The context for the question (optional).
  - `k`: The number of relevant passages to retrieve (optional).
  - `type`: The question type (optional).
  - `chat_history`: Previous chat history (optional).
  - `do_splitting`: Flag to split long documents (optional).
  - `add_to_history`: Flag to add the question to chat history (optional).
  - `chunk_size`: Size of chunks for splitting (optional).
  - `chunk_overlap`: Overlap between chunks (optional).

- **Response:**
  ```json
  {
    "answer": "HTML (Hyper Text Markup Language) is used for structuring the content of a web page, such as headings, paragraphs, images, links, etc. CSS (Cascading Style Sheets) is used for styling the content, like colors, fonts, layout, and animations. HTML provides the structure, while CSS provides the look and feel of a web page. In the provided context, the user is modifying CSS properties like font size, line height, and color to change the appearance of the web page.",
    "chat_history": [
      "What is the difference between HTML and CSS?",
      "HTML (Hyper Text Markup Language) is used for structuring the content of a web page, such as headings, paragraphs, images, links, etc. CSS (Cascading Style Sheets) is used for styling the content, like colors, fonts, layout, and animations. HTML provides the structure, while CSS provides the look and feel of a web page. In the provided context, the user is modifying CSS properties like font size, line height, and color to change the appearance of the web page."
    ]
  }

  ```

### 4. **Answer Comparison**
Compare a student's answer with the correct answer.

- **URL:** `/neuarlearn/ml/CompareAnswers`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "student_answer": "string",
    "correct_answer": "string",
    "question": "string"
  }
  ```
  - `student_answer`: The student's answer (required).
  - `correct_answer`: The correct answer (required).
  - `question`: The question being answered (required).

- **Response:**
  ```json
  {
    "comparison_result": "string"
  }
  ```

### 5. **Question Generation**
Generate questions based on the provided transcript.

- **URL:** `/neuarlearn/ml/QuestionGeneration`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "transcript": "string",
    "types": [1, 2],
    "chunk_size": 2000,
    "chunk_overlap": 1000
  }
  ```
  - `transcript`: The transcript text to generate questions from (required).
  - `types`: List of question types (required), where:
    - `1`: Multiple Choice Questions (MCQ)
    - `2`: True/False Questions
    - `3`: Open-ended Questions
  - `chunk_size`: Size of chunks for processing (optional, default: 2000).
  - `chunk_overlap`: Overlap between chunks (optional, default: 1000).

- **Response:**
```json
   {
    "0": {
      "Question": "What is the maximum width for the website?",
      "Answer": "300 pixels",
      "Choices": null
    },
    "1": {
      "Question": "What can you do to make the website responsive?",
      "Answer": "A",
      "Choices": [
        "A. Test the website in different screen sizes to ensure responsiveness.",
        "B. Use appropriate selectors and pseudo-classes for better specificity.",
        "C. Decide whether to use the 12-column or 16-column grid, and adjust your HTML structure accordingly.",
        "D. Apply CSS styles to the respective elements."
      ]
    }
  }
```
- You can test the script by navigating to `Test/request_test.py`. Simply uncomment any satisfied endpoint as shown below to execute the corresponding test:

![Test Script Example](https://github.com/user-attachments/assets/0c35a313-f3b1-4301-9037-79582e9f1eaf)

---
### Modeling Details

#### Question Generation Model
- **Technique**: We utilized **QLoRA** (Quantized Low-Rank Adaptation) to fine-tune the **LLaMA-3-8B** model efficiently. This method allows for efficient adaptation of large language models with reduced computational and memory overhead.
- **Dataset**: The fine-tuning was conducted using the [QG_QuestionsData Dataset](https://huggingface.co/datasets/shredder-31/QG_QusestionsData). which was created by combining various open-source datasets and applying a standardization schema.
- **Documentation**: For detailed metadata and information about similar open-source datasets, refer to the [DataSets MetaData](https://docs.google.com/document/d/1G_crHyknxzClI3VHSQGyNr-HkD75s0ZDGUqZWEehFBk/edit?usp=sharing).
- **Evaluation**: Model performance was evaluated using the ROUGE evolution metric. Comparisons were made with other open-source LLMs. Evaluation metrics and results are available in the linked documentation.

![image](https://github.com/user-attachments/assets/9a4cd30f-9128-4819-af66-c44a3c84cdd7)


- **Hyperparameters**: Detailed hyperparameter settings are provided in the [doc](https://github.com/MahmoudZaytoni/NeuraLearn-documentaion/blob/main/NeuraLearn_documentaion.pdf).

#### Summarization Model
- **Technique**: We fine-tuned the [LED-base-book-summary model](https://huggingface.co/pszemraj/led-base-book-summary), which employs the longformer architecture. Longformer is known for its efficient handling of long documents due to its sparse attention mechanism.
- **Dataset**: Fine-tuning was performed on the [PubMed Summarization dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization), which provides a range of medical abstracts and summaries useful for fine-tuning summarization models.

---

### Learned Lessons

- **Challenges in Question Generation**:
  - **Dataset Diversity**: Many question generation datasets suffer from limited diversity. For example, a common issue is the prevalence of naive questions, such as "What is the aim of this paragraph?" This lack of variety can constrain the model's ability to generate nuanced questions.
  - **Filtering Unimportant Details**: Another challenge is the model's tendency to overlook or misinterpret unimportant details within transcripts, which can affect the relevance of the generated questions.

- **Observations and Improvements**:
  - **LLM Utilization**: Utilizing the LLM itself for filtering unnecessary information from transcripts has shown to improve performance. This empirical approach helps in refining the data for more accurate question generation.
  - **Synthetic Data and Sophisticated Filtering**: Building synthetic data to enhance diversity or developing advanced methods to filter out unnecessary data points can significantly improve the quality of question generation.
