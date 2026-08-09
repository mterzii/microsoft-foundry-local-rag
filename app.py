import os

# 1. BİLGİSAYARIN KASMASINI VE DİLİNMESİNİ ÖNLEMEK İÇİN THREAD SINIRLAMASI
# CPU'nun tüm çekirdeklerini tüketmesini engeller
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import time
import faiss
import numpy as np
from foundry_local_sdk import (
    FoundryLocalManager, 
    Configuration, 
    EmbeddingsSession, 
    ChatSession,
    Request,
    TextItem,
    MessageItem
)

def get_embedding_vector(session, text: str) -> np.ndarray:
    """Metni float32 numpy vektörüne dönüştürür."""
    req = Request()
    req.add_item(TextItem(text))
    response = session.process_request(req)
    
    for item in response:
        if hasattr(item, "data") and isinstance(item.data, bytes):
            return np.frombuffer(item.data, dtype=np.float32)
        elif hasattr(item, "get_data"):
            data = item.get_data()
            if isinstance(data, bytes):
                return np.frombuffer(data, dtype=np.float32)
            return np.array(data, dtype=np.float32)
        elif hasattr(item, "data"):
            return np.array(item.data, dtype=np.float32)
            
        return np.array(list(item), dtype=np.float32)
        
    raise ValueError("Vektör çıkarılamadı!")

def main():
    config = Configuration(app_name="foundry_rag_app")
    manager = FoundryLocalManager(config)
    models = manager.catalog.list_models()

    # Modelleri Seçme (4k Model)
    emb_model = next((m for m in models if "embedding" in getattr(m, "alias", "").lower()), None)
    chat_model = next((m for m in models if "4k" in getattr(m, "alias", "").lower()), None)

    if not chat_model:
        chat_model = next((m for m in models if "embedding" not in getattr(m, "alias", "").lower() and "128k" not in getattr(m, "alias", "").lower()), None)

    print(f"Embedding Modeli: {emb_model.alias}")
    print(f"Chat Modeli: {chat_model.alias}\n")

    if not getattr(emb_model, "is_loaded", False):
        print("Embedding modeli yükleniyor...")
        emb_model.load()

    if not getattr(chat_model, "is_loaded", False):
        print("LLM Chat Modeli belleğe yükleniyor (CPU)...")
        chat_model.load()
        print("Chat modeli yüklendi!\n")

    emb_session = EmbeddingsSession(emb_model)
    chat_session = ChatSession(chat_model)

    # Doküman Okuma ve FAISS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "bilgi_tabani.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]

    print("Vektörler oluşturuluyor...")
    vector_list = [get_embedding_vector(emb_session, chunk) for chunk in chunks]
    vectors_np = np.vstack(vector_list).astype(np.float32)

    index = faiss.IndexFlatL2(1024)
    index.add(vectors_np)

    # Soru ve Bağlam
    query = "FAISS kütüphanesi nedir ve ne işe yarar?"
    query_vec = get_embedding_vector(emb_session, query).reshape(1, -1)
    distances, indices = index.search(query_vec, k=2)

    retrieved_context = "\n".join([chunks[idx] for idx in indices[0]])

    # Prompt
    prompt = (
        "Sen yardımcı bir asistansın. Verilen bağlama göre kısa cevap ver.\n\n"
        f"Bağlam:\n{retrieved_context}\n\n"
        f"Soru: {query}\n"
        "Cevap:"
    )

    print("\n--- LLM Yanıtı (Akan Metin) ---")
    chat_req = Request()
    chat_req.add_item(MessageItem.user(prompt))

    start_time = time.time()
    
    # Bekletmeden anlık çıktı almak için
    response = chat_session.process_request(chat_req)

    first_token = True
    for item in response:
        if first_token:
            print(f"(İlk yanıt {round(time.time() - start_time, 2)} saniyede geldi):\n")
            first_token = False

        text_content = ""
        if hasattr(item, "text") and item.text:
            text_content = item.text
        elif hasattr(item, "get_simple_text"):
            text_content = item.get_simple_text()
        elif hasattr(item, "data"):
            text_content = str(item.data)

        print(text_content, end="", flush=True)

    print("\n\n--- Tamamlandı ---")

if __name__ == "__main__":
    main()