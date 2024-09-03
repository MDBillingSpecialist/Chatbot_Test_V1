import os
import logging
import hashlib

def chunk_document_and_save(text, output_dir, chunk_size=10000, overlap_size=1000):
    """Chunk the document into manageable pieces with overlap, avoiding repetition, and save each chunk."""
    logging.debug("Starting the document chunking process.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunks = []
    text_length = len(text)
    i = 0
    chunk_num = 0

    while i < text_length:
        end = min(i + chunk_size, text_length)
        chunk = text[i:end]
        
        # Try to avoid cutting off in the middle of a paragraph
        if end < text_length and text[end] != '\n':
            while end < text_length and text[end] != '\n':
                end += 1
            chunk = text[i:end]
        
        chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
        if not chunks or chunk_hash != hashlib.md5(chunks[-1].encode('utf-8')).hexdigest():
            chunk_num += 1
            chunk_file = os.path.join(output_dir, f"chunk_{chunk_num:03d}.txt")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            chunks.append(chunk_file)
            logging.debug(f"Chunk {chunk_num} saved to {chunk_file}. Length: {len(chunk)} characters.")
        i = end - overlap_size

    logging.info(f"Document split into {len(chunks)} chunks with overlap.")
    return chunks

if __name__ == "__main__":
    # Load text from the previously saved file
    with open('output/full_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    output_dir = 'output/chunks'
    chunk_files = chunk_document_and_save(text, output_dir)
    logging.info("Document chunking complete.")
