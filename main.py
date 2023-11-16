import openai
import streamlit as st
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import language_tool_python
from similarity_searcher import SimilaritySearcher
import matplotlib.pyplot as plt
import fitz
import os

# SimilaritySearcher 인스턴스 생성
searcher = SimilaritySearcher(
    'embeddingwithcsv/company_embeddings.npy',
    'embeddingwithcsv/faiss_index.index',
    'embeddingwithcsv/university_embeddings.npy',
    'embeddingwithcsv/faiss_index_for_uni.index'
)

def format_search_results_for_gpt(companies, universities):
    # Convert single string to list if necessary
    if isinstance(companies, str):
        companies = [companies]
    if isinstance(universities, str):
        universities = [universities] 

    # Perform search only if companies list is not empty
    if companies:
        company_results = [{"name": company, "results": searcher.search_companies(company)} for company in companies]
    else:
        company_results = []

    # Perform search only if universities list is not empty
    if universities:
        university_results = [{"name": university, "results": searcher.search_universities(university)} for university in universities]
    else:
        university_results = []
    
    data_for_gpt = {
        "Companies worked at": company_results,
        "University name": university_results
    }
    return json.dumps(data_for_gpt)


def search_and_rank_items(json_data):
    companies = json_data.get("Companies worked at", [])
    universities = json_data.get("University name", [])
    # GPT에 전달할 형식으로 데이터 포맷팅
    formatted_data = format_search_results_for_gpt(companies, universities)
    st.markdown("**Semantic Search Result:**")
    st.write(formatted_data)
    # GPT 모델에 데이터 전달 및 응답 받기
    gpt_response = generate_gpt_verification_request(formatted_data)
    verification_results = parse_gpt_verification_response(gpt_response)
    # 검증 결과 처리 및 순위 추출
    company_ranks = {result['name']: result.get('rank', None) for result in verification_results['Companies worked at']}
    university_ranks = {result['name']: result.get('rank', None) for result in verification_results['University name']}

    return company_ranks, university_ranks


def generate_gpt_verification_request(formatted_data):
    prompt = (
        f"""Given the Data below, identify if each company and university exists in the Data. 
        If it does, return its rank. If it doesn't, return null. Below is the example of JSON format that you need to provide:
        {{
            "Companies worked at": [
                {{ "name": "",  "rank": int or null }}
            ],
            "University name": [
                {{ "name": "",  "rank": int or null }}
            ]
        }}
        
        Be noted that the Data below is the result of semantic search regarding to companies and universities, so the you have to find the company and university of the value of "name" from the value of "result".
        Provide the response in JSON format.\n\n
        Data: {formatted_data}"""
    )

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500,
        temperature=0.01
    )
    return response.choices[0].message['content']

def parse_gpt_verification_response(response):
    try:
        # JSON 형식 시작과 끝을 찾아서 추출
        start_index = response.find("{")
        end_index = response.rfind("}")
        if start_index == -1 or end_index == -1:
            return {"error": "JSON format not found in response"}
        
        json_str = response[start_index:end_index + 1]
        json_data = json.loads(json_str.replace("'", '"'))  # JSON 형식으로 파싱
        return json_data
    
    except Exception as e:
        return {"error": str(e)}

# Function to generate a response from the OpenAI GPT model
def generate_gpt_response(resume_input):
    prompt = f"""User's input resume: \n{resume_input}\n\n 
    Extract information from the resume above and fill in the values in the JSON format below. Things to consider when filling in empty values are:
    1. Do not change key names in json format.
    2. Companies worked at and University name values are always strings
    3. Multiple values may exist in list format.
    4. Degree values must in range of “B”(for bachelor’s degree), “M”(for master’s degree), “D”(for doctoral degree), and nothing if there is no information about it.
    5. Work duration values are always based on years and should be expressed as a signgle float number which means you have to sum up all the work durations.
    6. If there is no value, leave the column empty.

    You have to only output the following JSON format after filling it:
    {{"Companies worked at": [],
    "Degree": [],
    "University name": [],
    "Work duration": []}}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  # Replace with your desired GPT model
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1000,
        temperature=0.01
    )
    return response.choices[0].message['content']

# Extract JSON from the GPT response
def extract_json_from_response(response):
    try:
        # JSON 형식 시작과 끝을 찾아서 추출
        start_index = response.find("{")
        end_index = response.rfind("}")
        if start_index == -1 or end_index == -1:
            return {"error": "JSON format not found in response"}
        
        json_str = response[start_index:end_index + 1]
        json_data = json.loads(json_str.replace("'", '"'))  # JSON 형식으로 파싱
        return json_data
    
    except Exception as e:
        return {"error": str(e)}

# GPT 모델을 사용하여 JD 생성 및 JSON 형식으로 반환
def generate_job_description(job_title):
    prompt = f"Generate an ideal job description for the position(around 200 tokens): {job_title}. \n The job description that you generate will be used for the similarity check with user's resume. And please output the description in the following JSON format: \n{{'JD': 'Your job description here'}}"
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        temperature=0.01
    )
    return response.choices[0].message['content']

def extract_jd_from_json(response):
    try:
        # JSON 형식 시작과 끝을 찾아서 추출
        start_index = response.find("{")
        end_index = response.rfind("}")
        if start_index == -1 or end_index == -1:
            return {"error": "JSON format not found in response"}

        json_str = response[start_index:end_index + 1]
        json_data = json.loads(json_str.replace("'", '"'))  # JSON 형식으로 파싱
        jd = json_data.get('JD', None)
        if jd is None:
            return {"error": "JD key not found in JSON"}

        return jd
    except Exception as e:
        return {"error": str(e)}

def update_ranks_in_json(json_data, company_ranks, university_ranks):
    # 회사 정보가 단일 문자열인 경우 리스트로 변환
    if isinstance(json_data.get("Companies worked at", []), str):
        json_data["Companies worked at"] = [json_data["Companies worked at"]]
    
    # 대학 정보가 단일 문자열인 경우 리스트로 변환
    if isinstance(json_data.get("University name", []), str):
        json_data["University name"] = [json_data["University name"]]

    # 회사 이름을 해당 순위로 업데이트
    json_data["Companies worked at"] = [company_ranks.get(company) for company in json_data["Companies worked at"]]

    # 대학 이름을 해당 순위로 업데이트
    json_data["University name"] = [university_ranks.get(university) for university in json_data["University name"]]

    return json_data


def calculate_resume_score(cv_text, jd_text):
    embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')
    jd_embedding = embedder.encode([jd_text], convert_to_tensor=True)
    cv_embedding = embedder.encode([cv_text], convert_to_tensor=True)
    dimension = jd_embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(jd_embedding))
    D, I = index.search(np.array(cv_embedding), k=1)
    modified_distance_score = 100 - D[0][0]

    # Adjust the score by multiplying by 1.25
    adjusted_score = modified_distance_score * 1.25

    # Cap the maximum score at 100
    final_score = min(adjusted_score, 100)
    return max(0, final_score)

# 문법 및 철자 검사 함수
def check_grammar_and_spelling(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return matches

# 문법 및 철자 오류 점수 계산 함수
def calculate_grammar_score(errors, max_score=100):
    total_penalty = 0

    for error in errors:
        # 각 오류 유형에 따라 다른 점수 감점을 적용합니다.
        if 'GRAMMAR' in error.ruleIssueType:
            total_penalty += 5
        elif 'SPELLING' in error.ruleIssueType:
            total_penalty += 2
        else:
            # 기타 오류 유형에 대해 기본 감점을 적용할 수 있습니다.
            total_penalty += 1

    score = max(max_score - total_penalty, 0)
    return score

def calculate_score_from_json(json_data):
    # 학위 점수 계산 로직
    def calculate_degree_score(degree_list):
        degree_scores = {"B": 30, "M": 60, "D": 100}
        return max(degree_scores.get(degree, 0) for degree in degree_list) if degree_list else 0

    def calculate_work_duration_score(duration_list):
        # Check if duration_list is not a list and convert to list if necessary
        if not isinstance(duration_list, list):
            duration_list = [duration_list]

        total_duration = sum(duration_list)
        if total_duration < 1:
            return 15
        elif total_duration < 2:
            return 35
        elif total_duration < 3:
            return 50
        elif total_duration < 4:
            return 65
        elif total_duration < 5:
            return 85
        else:
            return 100

    # 회사 점수 계산 로직
    company_scores = []
    companies = json_data.get("Companies worked at", [])
    if not companies:
        company_scores.append(0)
    elif all(rank is None for rank in companies):
        null_count = len(companies)
        company_scores.append(10 if null_count <= 2 else 20)
    else:
        score_ranges = [
            (100, 1, 100),
            (90, 100, 1000),
            (80, 1000, 2000),
            (70, 2000, 3000),
            (60, 3000, 4000),
            (50, 4000, 5000),
            (40, 5000, 6000),
            (30, 6000, 7000),
            (20, 7000, float('inf'))
        ]
        for rank in companies:
            if rank is None:
                continue
            for score, lower_bound, upper_bound in score_ranges:
                if lower_bound <= rank < upper_bound:
                    company_scores.append(score)
                    break
        if not company_scores:
            company_scores.append(20)

    # 대학 점수 계산 로직
    university_scores = []
    universities = json_data.get("University name", [])
    if not universities:
        university_scores.append(0)
    elif all(rank is None for rank in universities):
        university_scores.append(10)
    else:
        score_ranges = [
            (100, 0, 30),
            (90, 30, 100),
            (80, 100, 200),
            (70, 200, 400),
            (60, 400, 600),
            (50, 600, 800),
            (40, 800, 1000),
            (30, 1000, 1400),
            (20, 1400, float('inf'))
        ]
        for rank in universities:
            if rank is None:
                continue
            for score, lower_bound, upper_bound in score_ranges:
                if lower_bound <= rank < upper_bound:
                    university_scores.append(score)
                    break
        if not university_scores:
            university_scores.append(20)

    # 학위 점수 계산
    degree_list = json_data.get("Degree", [])
    degree_score = calculate_degree_score(degree_list)

    # 근무 기간 점수 계산
    work_durations = json_data.get("Work duration", [])
    work_duration_score = calculate_work_duration_score(work_durations)

    # 각 영역의 점수를 가중치를 적용하여 최종 점수 계산
    final_score = (
        max(company_scores, default=0) * 0.3 + 
        max(university_scores, default=0) * 0.2 +
        degree_score * 0.3 +
        work_duration_score * 0.2
    )
    return final_score

def calculate_final_score(grammar_score, company_score, resume_score):
    # 철자 및 문법 점수: 20%, 이력 점수와 JD-CV 유사도 점수: 각각 40%
    final_score = (grammar_score * 0.2) + (company_score * 0.4) + (resume_score * 0.4)
    return round(final_score, 2)  # 소수점 두 자리까지 반올림

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit 앱 구성
st.set_page_config(page_title="CView")
# Streamlit 시크릿을 사용하여 OpenAI API 키 가져오기
openai_api_key = st.secrets["add_openai_api"]

st.title("CView: Resume Scorer")

job_title_input = st.text_input("**Enter the job title you are applying for (only in en):**", key="job_title")

# JD 생성 및 수정을 위한 UI 구성
if 'jd' not in st.session_state:
    st.session_state.jd = None

jd_button_container = st.empty()

# Check if there is already a job description saved
if st.session_state.jd is None:
    # If not, provide a button to generate a new job description
    if jd_button_container.button('Generate Job Description'):
        with st.spinner('Generating Job Description...'):
            st.session_state.jd = generate_job_description(job_title_input)
        jd_button_container.empty()
        jd_button_container.button('Modify Job Description', key="modify_jd")
else:
    # Provide an option to modify the job description
    if jd_button_container.button('Modify Job Description', key="modify_jd"):
        st.session_state.jd = None

# Using an expander to show the job description
if st.session_state.jd:
    with st.expander("View Generated Job Description"):
        st.info(st.session_state.jd)


uploaded_file = st.file_uploader("Upload your resume (PDF only)", type="pdf")

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

# 레쥬메 분석 버튼
if st.button('Get the Resume Score'):
    if uploaded_file and job_title_input:
        if st.session_state.jd is None:
            st.session_state.jd = generate_job_description(job_title_input)

        jd = extract_jd_from_json(st.session_state.jd)

        with st.spinner("Getting the score of the resume..."):
            gpt_response = generate_gpt_response(uploaded_file)
            json_data = extract_json_from_response(gpt_response)

            st.markdown("**Extracted JSON from Resume:**")
            st.json(json_data)
            company_ranks, university_ranks = search_and_rank_items(json_data)
            updated_json_ranks_only = update_ranks_in_json(json_data, company_ranks, university_ranks)

            st.markdown("**Ranked JSON by Semantic Search:**")
            st.write(updated_json_ranks_only)
            company_score_from_json = calculate_score_from_json(updated_json_ranks_only)

            resume_score = calculate_resume_score(uploaded_file, jd)

            # 문법 및 철자 검사
            grammar_errors = check_grammar_and_spelling(uploaded_file)
            grammar_score = calculate_grammar_score(grammar_errors)
            
            # 최종 점수 계산 및 결과 표시
            final_score = calculate_final_score(grammar_score, company_score_from_json, resume_score)

            # 점수 데이터 프레임 생성
            scores_df = pd.DataFrame({
                "Score Type": ["NER Score", "Resume Score", "Grammar Score"],
                "Value": [company_score_from_json, resume_score, grammar_score]
            })

            # 점수 그래프 표시
            st.markdown("**Score Overview**")
            fig, ax = plt.subplots()
            scores_df.plot(kind='bar', x='Score Type', y='Value', ax=ax, legend=False)
            plt.ylim(0, 100)
            plt.ylabel("Score")
            st.pyplot(fig)

            st.success(f"**Final Score:** {final_score}")
            
            if grammar_errors:
                st.sidebar.header("Grammar and Spelling Suggestions")
                for i, error in enumerate(grammar_errors, 1):
                    with st.sidebar.expander(f"Error {i}: {error.context}"):
                        st.write(f"Suggestion: {', '.join(error.replacements)}")
                        st.write(f"Rule Description: {error.message}")
