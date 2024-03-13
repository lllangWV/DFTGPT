from dotenv import load_dotenv
import os

load_dotenv()

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader
from llama_index.readers.file import PDFReader

def get_pdf_texts(directory_path):
    """
    This function will iterate over all PDF files in the given directory,
    extract their text content using the PDFReader, and return a list of text contents.
    """
    pdf_engines = []
    for file in os.listdir(directory_path):
        if file.endswith(".pdf"):
            file_name=file.split(".")[0]
            index_name=os.path.join('data','vector_databases', file_name)
            pdf_path = os.path.join(directory_path, file)
            pdf_text = PDFReader().load_data(file=pdf_path)
            doc_index = get_index(pdf_text,index_name)

            doc_engine = doc_index.as_query_engine()
            pdf_engines.append(doc_engine)
    return pdf_engines

def get_index(data,index_name):
    index = None
    if not os.path.exists(index_name):
        print(f"Creating index {index_name}")
        index = VectorStoreIndex.from_documents(data,show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))

    return index

# Directory where the PDFs are located
pdf_directory_path = os.path.join('data', 'dft_papers')

# # # Extract text from all PDFs in the directory
pdf_engines = get_pdf_texts(pdf_directory_path)

print(pdf_engines)

first_doc_engine=pdf_engines[0]

print(first_doc_engine.query("What are the functionals used in this paper?"))


