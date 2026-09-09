# =====================================================================
#  EĞİTİMİN ANİMASYONU
# =====================================================================
#
#  dogrusal_regresyon.py modelin nasıl öğrendiğini anlatıyor.
#  Bu dosya ise o öğrenmeyi GÖRSELLEŞTİRİR: gradyan inişi çalışırken
#  doğrunun veriye nasıl oturduğunu ve kaybın nasıl düştüğünü
#  kare kare izleyip 'egitim_animasyonu.gif' olarak kaydeder.
#
#  Çalıştırmak için:  python animasyon.py
# ---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Kendi yazdığımız modeli ve veriyi buradan alıyoruz (kod tekrarı yok)
from dogrusal_regresyon import (
    X, y, GERCEK_W, GERCEK_B,
    tahmin_et, egit,
)


# ---------------------------------------------------------------------
#  1) Önce modeli eğit ve her adımın geçmişini al
# ---------------------------------------------------------------------
OGRENME_ORANI = 0.01
EPOCH_SAYISI = 120

w_son, b_son, gecmis = egit(X, y,
                            ogrenme_orani=OGRENME_ORANI,
                            epoch_sayisi=EPOCH_SAYISI)

w_gecmis = gecmis["w"]          # her epoch sonundaki w
b_gecmis = gecmis["b"]          # her epoch sonundaki b
kayip_gecmis = gecmis["kayip"]  # her epoch sonundaki kayıp


# ---------------------------------------------------------------------
#  2) Grafik iskeletini kur (2 panel)
# ---------------------------------------------------------------------
fig, (sol, sag) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Makine öğrenmesi çalışırken: gradyan inişi adım adım",
             fontsize=13, fontweight="bold")

# --- SOL PANEL: veri + hareket eden doğru ---
sol.scatter(X, y, alpha=0.5, label="veri")
x_cizgi = np.linspace(X.min(), X.max(), 100)

# Gerçek (hedef) doğru - sabit, kesikli yeşil
sol.plot(x_cizgi, GERCEK_W * x_cizgi + GERCEK_B,
         "g--", linewidth=1.5, label="gerçek doğru (hedef)")

# Modelin doğrusu - animasyonda güncellenecek
(model_dogrusu,) = sol.plot([], [], "r-", linewidth=2.5,
                            label="modelin doğrusu")

sol.set_xlim(X.min() - 0.5, X.max() + 0.5)
sol.set_ylim(y.min() - 2, y.max() + 2)
sol.set_xlabel("X (girdi)")
sol.set_ylabel("y (çıktı)")
sol.legend(loc="upper left")
bilgi_yazisi = sol.text(0.03, 0.06, "", transform=sol.transAxes,
                        fontsize=10, family="monospace",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))

# --- SAĞ PANEL: kayıp eğrisi (soldan sağa çizilir) ---
sag.set_xlim(0, EPOCH_SAYISI)
sag.set_ylim(0, max(kayip_gecmis) * 1.05)
sag.set_xlabel("epoch (tur)")
sag.set_ylabel("kayıp (MSE)")
sag.set_title("Kayıp azaldıkça model 'öğreniyor'")
sag.grid(alpha=0.3)
(kayip_cizgisi,) = sag.plot([], [], color="purple", linewidth=2)
(kayip_noktasi,) = sag.plot([], [], "o", color="purple")


# ---------------------------------------------------------------------
#  3) Her kare için ne çizileceğini söyle
# ---------------------------------------------------------------------
def kare_ciz(kare):
    """kare = o anki epoch numarası."""
    w = w_gecmis[kare]
    b = b_gecmis[kare]

    # Sol panel: modelin o anki doğrusu
    model_dogrusu.set_data(x_cizgi, tahmin_et(x_cizgi, w, b))
    bilgi_yazisi.set_text(
        f"epoch : {kare:3d}\n"
        f"w     : {w:6.3f}\n"
        f"b     : {b:6.3f}\n"
        f"kayip : {kayip_gecmis[kare]:6.3f}"
    )

    # Sağ panel: kaybın o ana kadarki eğrisi
    x_ekseni = range(kare + 1)
    kayip_cizgisi.set_data(x_ekseni, kayip_gecmis[:kare + 1])
    kayip_noktasi.set_data([kare], [kayip_gecmis[kare]])

    return model_dogrusu, bilgi_yazisi, kayip_cizgisi, kayip_noktasi


# ---------------------------------------------------------------------
#  4) Animasyonu oluştur ve GIF olarak kaydet
# ---------------------------------------------------------------------
animasyon = FuncAnimation(fig, kare_ciz,
                          frames=EPOCH_SAYISI,
                          interval=60, blit=True)

plt.tight_layout()

cikti = "egitim_animasyonu.gif"
print(f"'{cikti}' oluşturuluyor... (birkaç saniye sürebilir)")
animasyon.save(cikti, writer=PillowWriter(fps=20))
print(f"Bitti -> {cikti}")

# Ekranda da göstermek istersen aşağıdaki satırı aç:
# plt.show()
