import os
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

# CPU Thread optimizasyonu
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

def get_embedding_vector(session, text: str) -> np.ndarray:
    """Metni float32 numpy vektörüne dönüştürür ve L2 normalize eder."""
    req = Request()
    req.add_item(TextItem(text))
    response = session.process_request(req)
    
    for item in response:
        if hasattr(item, "data") and isinstance(item.data, bytes):
            vec = np.frombuffer(item.data, dtype=np.float32)
        elif hasattr(item, "get_data"):
            data = item.get_data()
            vec = np.frombuffer(data, dtype=np.float32) if isinstance(data, bytes) else np.array(data, dtype=np.float32)
        elif hasattr(item, "data"):
            vec = np.array(item.data, dtype=np.float32)
        else:
            vec = np.array(list(item), dtype=np.float32)
            
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    raise ValueError("Vektör çıkarılamadı!")

def build_vector_index(emb_session, file_path: str):
    """Dokümanı okur ve FAISS indeksini hazırlar."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' bulunamadı!")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]

    print(f"[{len(chunks)}] metin parçası için vektörler üretiliyor...")
    vector_list = [get_embedding_vector(emb_session, chunk) for chunk in chunks]
    vectors_np = np.vstack(vector_list).astype(np.float32)

    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_np)
    
    print(f"✓ FAISS İndeksi Hazır (Vektör Boyutu: {dimension})")
    return index, chunks

def main():
    config = Configuration(app_name="foundry_rag_app")
    manager = FoundryLocalManager(config)
    models = manager.catalog.list_models()

    emb_model = next((m for m in models if "embedding" in getattr(m, "alias", "").lower()), None)
    chat_model = next((m for m in models if "4k" in getattr(m, "alias", "").lower()), None)

    if not chat_model:
        chat_model = next((m for m in models if "embedding" not in getattr(m, "alias", "").lower() and "128k" not in getattr(m, "alias", "").lower()), None)

    print(f"Embedding Modeli: {emb_model.alias}")
    print(f"Chat Modeli: {chat_model.alias}\n")

    if not getattr(emb_model, "is_loaded", False):
        emb_model.load()
    if not getattr(chat_model, "is_loaded", False):
        chat_model.load()

    emb_session = EmbeddingsSession(emb_model)
    chat_session = ChatSession(chat_model)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "bilgi_tabani.txt")

    index, chunks = build_vector_index(emb_session, file_path)

    print("\n" + "="*50)
    print(" Local RAG Asistanı (Çıkış için 'q')")
    print("="*50)

    while True:
        try:
            print("\n" + "-"*40)
            user_query = input("Soru Sorun > ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["q", "exit", "cikis", "çıkış"]:
                print("Oturum kapatılıyor...")
                break

            # Vektör Araması
            query_vec = get_embedding_vector(emb_session, user_query).reshape(1, -1)
            distances, indices = index.search(query_vec, k=2)
            retrieved_context = "\n---\n".join([chunks[idx] for idx in indices[0]])

            # Phi-3 Özgün Chat Şablonu (Sonsuz Döngüyü ve Yavaşlığı Önler)
            prompt = (
                "<|user|>\n"
                "Aşağıdaki bağlamı kullanarak soruya tek bir kısa cümle ile cevap ver. "
                "Eğer bilgi bağlamda yoksa sadece 'Bu bilgi veri tabanımda yer almıyor.' de.\n\n"
                f"Bağlam:\n{retrieved_context}\n\n"
                f"Soru: {user_query}<|end|>\n"
                "<|assistant|>\n"
            )

            chat_req = Request()
            chat_req.add_item(MessageItem.user(prompt))

            print("\n[LLM Yanıtı]: ", end="", flush=True)
            response = chat_session.process_request(chat_req)

            full_response = ""
            for item in response:
                text_content = ""
                if hasattr(item, "text") and item.text:
                    text_content = item.text
                elif hasattr(item, "get_simple_text"):
                    text_content = item.get_simple_text()
                elif hasattr(item, "data"):
                    text_content = str(item.data)

                # Özel etiketi ekrana basmadan temizleyelim
                clean_text = text_content.replace("<|end|>", "").replace("<|user|>", "")
                print(clean_text, end="", flush=True)
                
                full_response += text_content
                if "<|end|>" in full_response:
                    break
            
            print()

        except KeyboardInterrupt:
            print("\nİşlem iptal edildi.")
            break
        except Exception as e:
            print(f"\nBir hata oluştu: {e}")

if __name__ == "__main__":
    main()