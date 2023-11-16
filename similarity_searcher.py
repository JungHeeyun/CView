import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SimilaritySearcher:
    def __init__(self, company_embedding_file, company_index_file, university_embedding_file, university_index_file):
        self.company_embedding_file = company_embedding_file
        self.company_index_file = company_index_file
        self.university_embedding_file = university_embedding_file
        self.university_index_file = university_index_file

    def load_embeddings_and_index(self, embedding_file, index_file):
        embeddings = np.load(embedding_file)
        index = faiss.read_index(index_file)
        return embeddings, index

    def search_similar_items(self, query, index, names, name_to_index, top_k=10):
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        query_embedding = model.encode([query], convert_to_tensor=False)
        D, I = index.search(query_embedding, top_k)
        similar_items = [(names[i], name_to_index[names[i]]) for i in I[0]]
        return similar_items

    def search_companies(self, query):
        company_embeddings, company_index = self.load_embeddings_and_index(self.company_embedding_file, self.company_index_file)
        df = pd.read_csv(self.company_embedding_file.replace('company_embeddings.npy', 'company.csv'), header=None)
        company_names = df[0].tolist()
        company_name_to_index = {name: idx for idx, name in enumerate(company_names)}
        return self.search_similar_items(query, company_index, company_names, company_name_to_index)

    def search_universities(self, query):
        university_embeddings, university_index = self.load_embeddings_and_index(self.university_embedding_file, self.university_index_file)
        df = pd.read_csv(self.university_embedding_file.replace('university_embeddings.npy', 'uni.csv'), header=None)
        university_names = df[0].tolist()
        university_name_to_index = {name: idx for idx, name in enumerate(university_names)}
        return self.search_similar_items(query, university_index, university_names, university_name_to_index)

# # 예시 사용
# searcher = SimilaritySearcher(
#     '/Users/jeonghuiyun/PycharmProjects/pythonProject2/Resume/embeddingwithcsv/company_embeddings.npy',
#     '/Users/jeonghuiyun/PycharmProjects/pythonProject2/Resume/embeddingwithcsv/faiss_index.index',
#     '/Users/jeonghuiyun/PycharmProjects/pythonProject2/Resume/embeddingwithcsv/university_embeddings.npy',
#     '/Users/jeonghuiyun/PycharmProjects/pythonProject2/Resume/embeddingwithcsv/faiss_index_for_uni.index'
# )

# query_company = "google"
# print(searcher.search_companies(query_company))

# query_university = "JNTU - Kakinada, Andhra Pradesh"
# print(searcher.search_universities(query_university))
