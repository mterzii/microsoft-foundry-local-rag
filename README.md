<img width="1619" height="466" alt="Ekran görüntüsü 2026-08-11 173009" src="https://github.com/user-attachments/assets/c2a793a2-513d-4f43-8a43-493ef5a24b50" />
Foundry Local RAG

Microsoft stajım kapsamında geliştirdiğim bir proje. Temel fikir şu: bir dil modeline soru sorduğunda, model sana kendi eğitim verisinden değil — senin verdiğin belgelerden — cevap versin. Buna RAG (Retrieval-Augmented Generation) deniyor.

Tüm pipeline yerel çalışıyor. Sorgu, embedding, vektör arama, model çıkarımı — hepsi cihazda. İnternet bağlantısı yok, bulut yok, API anahtarı yok.

Nasıl çalışıyor?
Belgeyi parçalara ayır, her parçayı vektöre dönüştür (embedding)
Kullanıcı soru sorunca soruyu da vektöre dönüştür
FAISS ile en yakın belge parçalarını bul
Bunları phi-3'e bağlam olarak ver, cevabı ürettir

Model sadece verilen bağlamı kullanıyor. Bilmiyorsa "bilmiyorum" diyor — halüsinasyon değil.

Kullandıklarım
Microsoft Foundry Local — modeli cihazda çalıştıran runtime
phi-3 — on-device LLM
FAISS — vektör benzerlik araması
Python 3.11+
Kurulum
bash
git clone https://github.com/mterzii/microsoft-foundry-local-rag.git
cd microsoft-foundry-local-rag

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python app.py

İlk çalıştırmada Foundry Local phi-3 modelini indiriyor (~2GB). Bir kez inince sonrası tamamen offline.

Bilgi tabanı

bilgi_tabani.txt dosyasına istediğin metni ekle, uygulamayı yeniden başlat. Embedding'ler otomatik yeniden oluşturuluyor.

Bilinen eksikler
Şu an sadece .txt destekliyor, PDF/Word yok
FAISS index diske kaydedilmiyor, her başlatmada yeniden hesaplanıyor
Tek kullanıcı için tasarlandı — Foundry Local zaten çok kullanıcılı senaryo için değil
Kaynaklar
Foundry Local Docs  https://learn.microsoft.com/en-us/azure/foundry-local/
Building Your First Local RAG App https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968
