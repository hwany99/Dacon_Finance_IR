import pandas as pd
from vector_db_module import create_milvus_index
from embedding_module import embed_texts_from_file
from preprocess_module import process_pdfs_from_dataframe, key_index
#한번 만들었으면 더이상 실행하지 않아도 됨


# # 1.pdf파일에서 chunk 추출후 저장
# data = []
# base_directory = '/home/a2024712006/dacon' # Your Base Directory
# df = pd.read_csv('./test.csv')
# data_frames = process_pdfs_from_dataframe(df, base_directory)
# for key, value in data_frames.items():
#     for i, chunk in enumerate(value):
#         data.append({"key": key, "chunk": chunk})
# df = pd.DataFrame(data)
# df.to_csv('dacon_chunk.csv', index=False)

# # 2. chunk파일에서 임베딩 생성
df = pd.read_csv('./dacon_chunk.csv')
df = embed_texts_from_file(df)
output_path = "./dacon_chunk_embedded.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

# df = pd.read_csv("./dacon_chunk_embedded.csv")

# 4. 임베딩된 데이터로 pdf마다 Milvus DB collection 생성
unique_keys = df['key'].unique()

for key in unique_keys:
    filtered_rows = df[df['key'] == key]

    # 결과 출력
    print(key, filtered_rows)
    
    # Milvus collection에는 한국어 텍스트가 들어가므로, key를 index로 변환
    index = key_index(key)
    create_milvus_index(filtered_rows, index)
    