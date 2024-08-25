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
├── create_db.py
├── extract.py
├── model.py
├── preprocess.py
├── retrieve.py
└── train.py
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
```
```bash
wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz
tar -xzf jdk-21_linux-x64_bin.tar.gz
```
#### add PATH in ~/.bashrc
```bash
export JAVA_HOME=~/jdk-21.0.4
export PATH=$JAVA_HOME/bin:$PATH
```
#### shell
```bash
source ~/.bashrc
```

## Inference
```bash
python retrieve.py
```
