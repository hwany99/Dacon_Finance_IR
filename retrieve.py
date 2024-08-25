import os
import pandas as pd
from tqdm import tqdm

from create_db import process_pdfs_from_dataframe, load_pdf_databases, save_pdf_databases
from model import get_QA_output, get_Rerank_output

import warnings
warnings.filterwarnings("ignore")

### DB ###
base_dir = './data/'
df = pd.read_csv(base_dir + 'test.csv')

db_filename = os.path.join(base_dir, 'pdf_databases.pkl')
if os.path.exists(db_filename):
    pdf_databases = load_pdf_databases(db_filename)
else:
    pdf_databases = process_pdfs_from_dataframe(df, base_dir)
    save_pdf_databases(pdf_databases, db_filename)


### Inference ###
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

def process_row(row, pdf_databases):
    source = row['Source']
    question = row['Question']

    retriever = pdf_databases[source]
    relevant_docs = retriever.invoke(question)

    context, _ = get_Rerank_output(relevant_docs, question)
    response = get_QA_output(context, question)
    response = response.replace('\n\n', '\n')
    
    print('Question:', question)
    print('Answer:', response)

    return {
        "Source": row['Source'],
        "Source_path": row['Source_path'],
        "Question": question,
        "Answer": response
    }

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering Questions"):
    result = process_row(row, pdf_databases)
    results.append(result)


### Save ###
submit_df = pd.read_csv(base_dir + "sample_submission.csv")
submit_df['Answer'] = [item['Answer'].replace('**', '') for item in results]
submit_df.to_csv(base_dir + "final_submission.csv", encoding='UTF-8-sig', index=False)
