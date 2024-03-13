import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader
from llama_index.readers.file import PDFReader

def get_pdf_texts(directory_path):
    """
    This function will iterate over all PDF files in the given directory,
    extract their text content using the PDFReader, and return a list of text contents.
    """
    pdf_texts = []
    for file in os.listdir(directory_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, file)
            pdf_text = PDFReader().load_data(file=pdf_path)
            pdf_texts.append(pdf_text)
    return pdf_texts

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


documnets= SimpleDirectoryReader(pdf_directory_path).load_data()
# # # Extract text from all PDFs in the directory
# pdf_texts = get_pdf_texts(pdf_directory_path)
# print(pdf_texts[:5])
# # Create or load the index
dft_index = get_index(documnets, 'data/dft_index')

# # Create a query engine from the index
dft_engine = dft_index.as_query_engine()



