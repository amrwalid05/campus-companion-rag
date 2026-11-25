from rag_engine import rebuild_index

if __name__ == "__main__":
    file_count, chunk_count = rebuild_index()
    print(f"Index ready: {file_count} documents, {chunk_count} chunks.")
