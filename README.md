# CView
Resume Scorer is a sophisticated Streamlit application designed to evaluate and score resumes based on job descriptions. It leverages the power of GPT models, semantic search, and advanced NLP techniques to provide a comprehensive analysis of a resume, giving valuable insights to both job seekers and recruiters.

**Features**
**PDF Resume Upload:** Users can upload their resumes in PDF format. The application extracts text for analysis.
**Job Description Generation:** Enter a job title to automatically generate a relevant job description.
**Semantic Search:** Analyzes companies and universities mentioned in the resume using advanced similarity search techniques.
**Grammar and Spelling Check:** Evaluates the resume for grammatical and spelling errors.
**Resume Scoring:** Calculates a composite score based on various criteria like work experience, educational background, and relevance to the job description.
**Score Visualization:** Presents a detailed score breakdown with a graphical representation for easy understanding.
**Grammar and Spelling Suggestions:** Offers suggestions for improvement in grammar and spelling, displayed in the sidebar.

**Setup and Installation**
**1. Install Required Libraries:** 
pip install streamlit openai faiss numpy pandas sentence_transformers language_tool_python matplotlib pymupdf

**2. API Key Configuration:**
Set your OpenAI API key in the script:
add_openai_api = "<Your-OpenAI-API-Key>"

**3. Running the App:**
Launch the application by running:
streamlit run your_script_name.py

**Usage**
**Start by Entering a Job Title:** Enter the job title you're applying for. The application will generate a corresponding job description.
**Upload Your Resume:** Click on the 'Upload your resume' button and select your resume in PDF format.
**Get the Resume Score:** Once the resume and job title are inputted, click on 'Get the Resume Score' to start the analysis.
**Review Results:** The application will display the extracted resume data, perform semantic searches, and show the final scores along with suggestions for improvement.

**Technologies Used**
**Streamlit:** For creating the web application interface.
**OpenAI's GPT:** For generating job descriptions and processing resume data.
**FAISS & Sentence Transformers:** For semantic search capabilities.
**PyMuPDF:** For extracting text from PDF files.
**Language Tool Python:** For grammar and spelling checks.
**Pandas & Matplotlib:** For data handling and visualization.
