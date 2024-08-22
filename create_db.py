import pickle
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_teddynote.retrievers import KiwiBM25Retriever, OktBM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores.utils import DistanceStrategy

from preprocess import process_pdf

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

def create_vector_db(chunks):
    global embeddings
    db = FAISS.from_documents(chunks, embedding=embeddings, distance_strategy = DistanceStrategy.COSINE)
    return db

def process_single_pdf(path, base_directory):
    path = base_directory + path
    
    pdf_title = path.split('/')[-1].replace('.pdf', '')
    print(f"Processing {pdf_title}...")

    chunks = process_pdf(path)
    db = create_vector_db(chunks)
    
    kiwi_bm25_retriever = KiwiBM25Retriever.from_documents(chunks)
    faiss_retriever = db.as_retriever()
    
    retriever = EnsembleRetriever(
        retrievers=[kiwi_bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
        search_type="similarity",
    )

    return pdf_title, retriever

def process_pdfs_from_dataframe(df, base_directory):
    pdf_databases = {}
    unique_paths = df['Source_path'].unique()

    for path in tqdm(unique_paths):
        pdf_key, pdf_value = process_single_pdf(path, base_directory)
        pdf_databases[pdf_key] = pdf_value

    return pdf_databases

def save_pdf_databases(pdf_databases, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pdf_databases, f)

def load_pdf_databases(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)