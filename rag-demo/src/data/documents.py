from typing import List

def load_documents(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = file.readlines()
    return [doc.strip() for doc in documents]

def process_documents(documents: List[str]) -> List[dict]:
    processed_docs = []
    for doc in documents:
        processed_docs.append({"text": doc})
    return processed_docs