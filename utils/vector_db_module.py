from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import time
from embedding_module import embed_single_text
import numpy as np
############## 1.Vector DB 생성안했다면 생성 ##############
def create_milvus_index(df, source_name):

    ############## Milvus 서버에 연결 ##############
    connections.connect("default", host="localhost", port="19530")

    ############# 기존 컬렉션 삭제 (있는 경우) ##############
    if utility.has_collection(source_name):
        existing_collection = Collection(source_name)
        existing_collection.drop()
        print(f"Collection {source_name} already exists. Dropping it...")

    ############## 컬렉션 및 필드 스키마 정의 ##############
    # 총 3개
    fields = [
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, is_primary=True, max_length=8192),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    schema = CollectionSchema(fields, "dacon collection")
    dacon_db = Collection(source_name, schema)

    ############## 데이터 삽입 준비 ##############
    chunks = df["chunk"].tolist()
    embeddings = df["embedding"].tolist()

    entities = [chunks, embeddings]

    ############## 데이터 삽입 및 flush ##############

    # open_ai 임베딩시 차원이 너무커져서 batch처리
    def batch_insert(collection, entities, batch_size=500):
        for i in range(0, len(entities[0]), batch_size):
            batched_entities = [entity[i:i + batch_size]
                                for entity in entities]
            collection.insert(batched_entities)

    batch_insert(dacon_db, entities)
    dacon_db.flush()

    ############## 인덱스 생성 ##############

    # dataset이 증가한다면 HNSW가 적합
    index = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    print("Creating metric: COSINE, index: IVF_FLAT")
    dacon_db.create_index("embeddings", index)
    connections.disconnect("default")


########################### 2.Vector DB 검색 함수 #####################################################


def search_milvus_vectors(text, count, source_name):
    search_vector = embed_single_text(text)
    
    # Milvus 서버에 연결
    connections.connect("default", host="localhost", port="19530")
    voice_phishing_db = Collection(source_name)
    voice_phishing_db.load()

    search_params = {
        "metric_type": "COSINE",
        # nprobe: 찾을 개수
        "params": {"nprobe": 8}
    }
    start_time = time.time()

    ############## 유사성 검색 수행 ##############
    # limit: nprobe에서 찾은 것중 상위 count개
    result = voice_phishing_db.search(
        [search_vector], "embeddings", search_params, limit=count, output_fields=["chunk"])
    end_time = time.time()

    milvus_search_time = end_time - start_time
    return_value = []

    ############## 유사성 검색 결과 ##############
    for hits in result:
        for i, hit in enumerate(hits):
            hit_distance = hit.distance
            hit_content = hit.entity.get("chunk")
            return_value.append((hit_distance, hit_content))

    connections.disconnect("default")
    return return_value, milvus_search_time