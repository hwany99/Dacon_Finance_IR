# Dacon_Finance_IR
## Repository Structure
``` bash
.
├── README.md
├── data/
│   ├── test_source/
│   ├── train_source/
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
│
### web
├── chat/
├── static/
├── web/
├── manage.py
│
### modules
├── create_db.py
├── extract.py
├── model.py
├── preprocess.py
│
### retrieve
└── retrieve.py
```

## Install (Ubuntu)
#### shell
```bash
pip install accelerate datasets
pip install -i https://pypi.org/simple/ bitsandbytes
pip install transformers[torch] -U
pip install langchain langchain_community langchain_huggingface
pip install PyMuPDF faiss-gpu
pip install sentence-transformers peft opencv-python
pip install kiwipiepy langchain-teddynote
pip install django
```

## Inference
```bash
python retrieve.py
```

## Web
```bash
python manage.py runserver
```
access [localhost(127.0.0.1:8000)](http://127.0.0.1:8000/)
