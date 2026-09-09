# =====================================================================
#  MAKİNE ÖĞRENMESİNİN TEMELİ: SIFIRDAN DOĞRUSAL REGRESYON
# =====================================================================
#
#  Bu dosya, makine öğrenmesinin en temel fikrini hiçbir hazır ML
#  kütüphanesi kullanmadan (scikit-learn yok, tensorflow yok) sadece
#  NumPy ile gösterir. Amaç: "model nasıl öğreniyor?" sorusunu
#  satır satır anlamak.
#
#  Öğreneceğimiz kavramlar:
#    1) Veri (X girdileri, y çıktıları)
#    2) Model / hipotez           ->  y_tahmin = w * x + b
#    3) Parametreler (w, b)       ->  modelin "öğrendiği" sayılar
#    4) Kayıp fonksiyonu (MSE)    ->  model ne kadar yanılıyor?
#    5) Gradyan (türev)           ->  hangi yöne gidersek hata azalır?
#    6) Gradyan inişi             ->  parametreleri adım adım düzeltmek
#    7) Öğrenme oranı (lr)        ->  adım büyüklüğü
#    8) Epoch (tur)               ->  tüm veriyi kaç kez gördük?
#
# ---------------------------------------------------------------------

import numpy as np              # sayısal işlemler (vektör/matris) için
import matplotlib.pyplot as plt # sonucu grafikle görmek için


# =====================================================================
#  1. ADIM: VERİYİ HAZIRLA
# =====================================================================
#
#  Makine öğrenmesi "veriden" öğrenir. Gerçek hayatta veri bir dosyadan
#  gelir; burada biz kendimiz üretiyoruz ki "doğru cevabı" bilelim.
#
#  Gizli gerçek kural (modelin bulmasını istediğimiz):
#        y = 2 * x + 5
#
#  Üstüne biraz "gürültü" (noise) ekliyoruz, çünkü gerçek veriler
#  hiçbir zaman kusursuz bir çizgi üzerinde olmaz.

np.random.seed(42)  # rastgeleliği sabitler -> her çalıştırmada aynı sonuç

GERCEK_W = 2.0   # gizli eğim (slope)
GERCEK_B = 5.0   # gizli kesişim (bias / intercept)

N = 100                                   # örnek (veri noktası) sayısı
X = np.random.uniform(-5, 5, size=N)      # -5 ile 5 arasında 100 rastgele girdi
gurultu = np.random.normal(0, 2, size=N)  # ortalaması 0, sapması 2 olan gürültü
y = GERCEK_W * X + GERCEK_B + gurultu     # gerçek kural + gürültü = gözlemlenen çıktı

# Not: Model X'ten y'yi tahmin etmeye çalışacak. w=2, b=5 değerlerini
# ASLA doğrudan görmüyor; sadece (X, y) çiftlerine bakarak tahmin edecek.


# =====================================================================
#  2. ADIM: MODELİ TANIMLA
# =====================================================================
#
#  "Model" = girdiyi çıktıya çeviren bir fonksiyon.
#  Doğrusal regresyonda model bir doğru denklemidir:
#
#        y_tahmin = w * x + b
#
#  w (weight/ağırlık) ve b (bias) modelin PARAMETRELERİDİR.
#  Öğrenme = bu iki sayıyı iyi değerlere getirmek.

def tahmin_et(x, w, b):
    """Verilen x için modelin çıktısını hesaplar (ileri geçiş / forward pass)."""
    return w * x + b


# =====================================================================
#  3. ADIM: KAYIP (LOSS) FONKSİYONU
# =====================================================================
#
#  Modelin ne kadar kötü olduğunu tek bir sayıyla ölçmeliyiz ki
#  "iyileşiyor mu?" diyebilelim.
#
#  Kullanacağımız ölçü: Ortalama Kare Hata (Mean Squared Error, MSE)
#
#        MSE = (1/n) * Σ (y_tahmin_i - y_i)^2
#
#  Neden karesini alıyoruz?
#    - Negatif ve pozitif hatalar birbirini götürmesin diye
#    - Büyük hataları daha çok cezalandırmak için
#    - Türevi kolay ve pürüzsüz (gradyan inişi için şart)

def kayip_hesapla(y_tahmin, y_gercek):
    """MSE: tahminlerin gerçek değerlerden ne kadar saptığının ölçüsü."""
    hata = y_tahmin - y_gercek
    return np.mean(hata ** 2)


# =====================================================================
#  4. ADIM: GRADYAN (TÜREV) HESABI
# =====================================================================
#
#  Kayıp fonksiyonu w ve b'ye bağlı bir "yüzey"dir. Bu yüzeyin en
#  alçak noktasını arıyoruz (kayıp minimum).
#
#  Gradyan = kaybın parametrelere göre türevi = "en dik yokuş yukarı" yönü.
#  Biz kaybı AZALTMAK istediğimiz için gradyanın TERS yönüne gideceğiz.
#
#  MSE'nin türevleri (matematiği bir kez çıkarılır, sonra hep kullanılır):
#
#        dL/dw = (2/n) * Σ ( (w*x + b - y) * x )
#        dL/db = (2/n) * Σ ( (w*x + b - y) )

def gradyan_hesapla(x, y_gercek, w, b):
    """Kaybın w ve b'ye göre eğimini (türevini) döndürür."""
    n = len(x)
    y_tahmin = tahmin_et(x, w, b)
    hata = y_tahmin - y_gercek           # her nokta için ne kadar saptık
    dL_dw = (2 / n) * np.sum(hata * x)   # w yönündeki eğim
    dL_db = (2 / n) * np.sum(hata)       # b yönündeki eğim
    return dL_dw, dL_db


# =====================================================================
#  5. ADIM: EĞİTİM DÖNGÜSÜ (GRADYAN İNİŞİ)
# =====================================================================
#
#  Algoritma çok basit ve TÜM derin öğrenmenin de kalbidir:
#
#     tekrar tekrar:
#        1. tahmin yap            (forward)
#        2. kaybı ölç
#        3. gradyanı hesapla      (backward)
#        4. parametreleri gradyanın tersine küçük bir adım kaydır
#
#  yeni_w = w - lr * dL/dw
#  yeni_b = b - lr * dL/db
#
#  lr (learning rate / öğrenme oranı): adım büyüklüğü.
#     - çok büyük -> ıraksar (patlar), minimumu ıskalar
#     - çok küçük -> çok yavaş öğrenir

def egit(x, y_gercek, ogrenme_orani=0.01, epoch_sayisi=200):
    """Gradyan inişi ile en iyi w ve b'yi bulur. Geçmişi de kaydeder."""

    # Parametreleri "boş" bir yerden başlatıyoruz. Model hiçbir şey bilmiyor.
    w = 0.0
    b = 0.0

    # Eğitim boyunca değerleri saklayalım (sonra grafik/animasyon için)
    gecmis = {"w": [], "b": [], "kayip": []}

    for epoch in range(epoch_sayisi):
        # --- ileri geçiş: tahmin ve kayıp ---
        y_tahmin = tahmin_et(x, w, b)
        kayip = kayip_hesapla(y_tahmin, y_gercek)

        # --- geri geçiş: gradyan ---
        dL_dw, dL_db = gradyan_hesapla(x, y_gercek, w, b)

        # --- parametre güncellemesi: minimuma doğru bir adım ---
        w = w - ogrenme_orani * dL_dw
        b = b - ogrenme_orani * dL_db

        # --- kayıt ---
        gecmis["w"].append(w)
        gecmis["b"].append(b)
        gecmis["kayip"].append(kayip)

        # Her 20 epoch'ta bir ilerlemeyi yazdır
        if epoch % 20 == 0 or epoch == epoch_sayisi - 1:
            print(f"epoch {epoch:3d} | kayip = {kayip:7.3f} | "
                  f"w = {w:5.3f} | b = {b:5.3f}")

    return w, b, gecmis


# =====================================================================
#  6. ADIM: ÇALIŞTIR VE SONUCU GÖSTER
# =====================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Gizli gerçek kural:  y = {:.1f} * x + {:.1f}".format(GERCEK_W, GERCEK_B))
    print("  Model bunu sadece verilere bakarak bulmaya çalışacak")
    print("=" * 55)

    w_ogrenilen, b_ogrenilen, gecmis = egit(
        X, y,
        ogrenme_orani=0.01,   # <- bu hiperparametreyle oynamayı dene
        epoch_sayisi=200,
    )

    print("=" * 55)
    print(f"  ÖĞRENİLEN model:  y = {w_ogrenilen:.3f} * x + {b_ogrenilen:.3f}")
    print(f"  GERÇEK  model:    y = {GERCEK_W:.3f} * x + {GERCEK_B:.3f}")
    print("=" * 55)

    # --- İki grafik: (solda) veri + öğrenilen doğru, (sağda) kayıp eğrisi ---
    fig, (eksen1, eksen2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sol: dağılım grafiği + modelin bulduğu doğru
    eksen1.scatter(X, y, alpha=0.6, label="veri (X, y)")
    x_cizgi = np.linspace(X.min(), X.max(), 100)
    eksen1.plot(x_cizgi, tahmin_et(x_cizgi, w_ogrenilen, b_ogrenilen),
                color="red", linewidth=2, label="öğrenilen doğru")
    eksen1.plot(x_cizgi, tahmin_et(x_cizgi, GERCEK_W, GERCEK_B),
                color="green", linestyle="--", linewidth=1.5, label="gerçek doğru")
    eksen1.set_title("Model veriye uyuyor mu?")
    eksen1.set_xlabel("X (girdi)")
    eksen1.set_ylabel("y (çıktı)")
    eksen1.legend()

    # Sağ: kayıp her epoch'ta nasıl düştü?
    eksen2.plot(gecmis["kayip"], color="purple")
    eksen2.set_title("Kayıp (MSE) her turda azalıyor")
    eksen2.set_xlabel("epoch (tur)")
    eksen2.set_ylabel("kayıp")
    eksen2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sonuc.png", dpi=110)
    print("\n'sonuc.png' kaydedildi. Pencereyi kapatınca program biter.")
    plt.show()
