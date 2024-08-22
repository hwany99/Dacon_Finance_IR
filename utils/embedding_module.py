from tqdm import tqdm
import concurrent.futures
import re
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("intfloat/multilingual-e5-base")

############## 1.단일 문장 임베딩 생성 코드 ##############
def embed_single_text(text):
    if not isinstance(text, str):
        text = str(text)  # 문자열이 아닌 경우 문자열로 변환
    cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    cleaned_text = re.sub(r'None', '', cleaned_text)
    cleaned_text = cleaned_text.replace("\n", " ")
    # embedding = model.encode(cleaned_text)
    embedding = model.encode(cleaned_text)['dense_vecs']
    print(embedding.shape)
    return embedding

############## 3.전체 문장 임베딩 생성 코드 ##############
def embed_texts_from_file(df):
    # 모든 값이 문자열이 되도록 변환
    df["chunk"] = df["chunk"].apply(lambda x: str(x) if not isinstance(x, str) else x)
    embeddings = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(embed_single_text, text)
                   for text in df["chunk"]]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            embeddings.append(future.result())

    df["embedding"] = embeddings
    return df
