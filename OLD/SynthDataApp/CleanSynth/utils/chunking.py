def chunk_document(text, chunk_size=10000):
    chunks = []
    text_length = len(text)
    i = 0

    while i < text_length:
        end = min(i + chunk_size, text_length)
        chunk = text[i:end]

        if end < text_length and text[end] != '\n':
            while end < text_length and text[end] != '\n':
                end += 1
            chunk = text[i:end]

        chunks.append(chunk)
        i = end

    return chunks
