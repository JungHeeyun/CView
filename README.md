# CView: Streamlit Resume Scoring Application

## Project Overview

CView is a Streamlit-based application designed to deliver actionable insights to job seekers and recruiters by evaluating and scoring resumes against job descriptions. Utilizing GPT models for semantic analysis, CView provides a multi-faceted evaluation that encompasses work experience, educational background, and other critical criteria.

## Features

- **PDF Resume Upload**: Easily submit resumes in PDF format for analysis.
- **Job Description Generation**: Automatically generates a job description based on a specified job title using GPT-powered semantic search.
- **Semantic Search**: Examines companies and universities listed in a resume to find matches with the job description.
- **Grammar and Spelling Check**: Assesses the resume for grammatical and spelling accuracy.
- **Resume Scoring**: Generates a score reflecting the resume's alignment with job requirements.
- **Score Visualization**: Displays the scoring breakdown with a graphical representation.
- **Grammar and Spelling Suggestions**: Provides actionable feedback to enhance resume quality.

## Limitations

- **Contextual Scoring**: The scoring system based on overall similarity between job descriptions and resumes may provide vague results and low correlation due to its broad evaluation criteria.
- **Data Extraction Stability**: Utilizing GPT for extracting data from resumes can be unstable, leading to inconsistent analysis outcomes.
- **Subjective Scoring Metrics**: The scoring metrics are currently subjective, which may not accurately reflect an objective fit for the job.

## Potential for Expansion

- **Resume Writer Development**: To further assist users, the development of a Resume Writer feature could complement the Scorer, offering suggestions for improving resumes based on scores and feedback.
- **Continuous Enhancement of Scoring Criteria**: Ongoing refinement of scoring standards will enhance the objectivity and reliability of the scoring system.

## Setup and Installation

1. Install Required Libraries:
    
    Copy code
    
    `pip install streamlit openai faiss numpy pandas sentence_transformers language_tool_python matplotlib pymupdf`
    
2. API Key Configuration:
    
    makefileCopy code
    
    `add_openai_api = ""`
    
3. Running the App:
    
    arduinoCopy code
    
    `streamlit run your_script_name.py`
    

## Usage

1. **Start by Entering a Job Title**: Input the job title relevant to your search.
2. **Upload Your Resume**: Submit your resume in PDF format.
3. **Get the Resume Score**: Obtain a comprehensive analysis and scoring of your resume, with detailed feedback.

## Technologies Used

- **Streamlit**: For streamlined web app development.
- **OpenAI's GPT**: Empowers job description synthesis and semantic analysis.
- **FAISS & Sentence Transformers**: Provide enhanced semantic search capabilities.
- **PyMuPDF**: For robust text extraction from PDFs.
- **Language Tool Python**: Offers extensive grammar and spelling checks.
- **Pandas & Matplotlib**: For data handling and visualization.
