
import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini():
    """
    API anahtarını .env dosyasından yükler, Gemini modelini yapılandırır
    ve basit bir test istemi gönderir.
    """
    try:
        # .env dosyasındaki ortam değişkenlerini yükle
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("HATA: GOOGLE_API_KEY ortam değişkeni bulunamadı.")
            print("Lütfen .env dosyanızı kontrol edin.")
            return

        # API anahtarı ile SDK'yı yapılandır
        genai.configure(api_key=api_key)

        # Kullanılacak doğru modeli seç
        model = genai.GenerativeModel('models/gemini-pro-latest')

        # Modele göndermek için bir istem (prompt) oluşturun
        prompt = "Bir hisse senedi için RSI (Göreceli Güç Endeksi) 75 ise, bu durum genellikle alım veya satım sinyali olarak nasıl yorumlanır? Kısa bir özet ver."

        print(f"Gönderilen İstem: '{prompt}'")
        print("-" * 20)

        # Modeli çalıştır ve yanıtı al
        response = model.generate_content(prompt)

        # Yanıtı yazdır
        print("Gemini Modelinden Gelen Yanıt:")
        print(response.text)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    test_gemini()
