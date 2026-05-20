import streamlit as st
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# إعدادات الصفحة في Streamlit
st.set_page_config(page_title="Bulanık Mantık Klima Kontrolü", layout="wide")
st.title("🧠 Bulanık Mantık ile Otomatik Klima Kontrol Sistemi")
st.write("BST Dönem Projesi - @Alaa Alattar")

# --- 1. تدوير نطاقات المتغيرات (Universes) ---
x_sicaklik = np.arange(0, 51, 1)    # 0 - 50 derece
x_nem = np.arange(0, 101, 1)         # %0 - %100
x_boyut = np.arange(0, 101, 1)       # 0 - 100 m²
x_fan = np.arange(0, 101, 1)         # %0 - %100 fan hızı

# --- 2. تعريف دالات الانتماء (Membership Functions) ---
# Sıcaklık
sicaklik_dusuk = fuzz.trimf(x_sicaklik, [0, 0, 22])
sicaklik_normal = fuzz.trimf(x_sicaklik, [18, 25, 32])
sicaklik_yuksek = fuzz.trimf(x_sicaklik, [28, 50, 50])

# Nem
nem_dusuk = fuzz.trimf(x_nem, [0, 0, 40])
nem_normal = fuzz.trimf(x_nem, [30, 50, 70])
nem_yuksek = fuzz.trimf(x_nem, [60, 100, 100])

# Oda Boyutu
boyut_kucuk = fuzz.trimf(x_boyut, [0, 0, 35])
boyut_orta = fuzz.trimf(x_boyut, [25, 50, 75])
boyut_buyuk = fuzz.trimf(x_boyut, [65, 100, 100])

# Fan Hızı (Output)
fan_dusuk = fuzz.trimf(x_fan, [0, 0, 40])
fan_orta = fuzz.trimf(x_fan, [30, 50, 70])
fan_yuksek = fuzz.trimf(x_fan, [60, 100, 100])

# --- 3. واجهة المستخدم (المدخلات عبر Sliders) ---
st.sidebar.header("🎛️ Giriş Değerleri (Manuel Ayar)")
input_sicaklik = st.sidebar.slider("Sıcaklık (°C)", 0, 50, 26)
input_nem = st.sidebar.slider("Nem (%)", 0, 100, 55)
input_boyut = st.sidebar.slider("Oda Boyutu (m²)", 0, 100, 40)

# --- 4. حساب درجات الانتماء للمدخلات الحالية (Fuzzification) ---
st_dusuk = fuzz.interp_membership(x_sicaklik, sicaklik_dusuk, input_sicaklik)
st_normal = fuzz.interp_membership(x_sicaklik, sicaklik_normal, input_sicaklik)
st_yuksek = fuzz.interp_membership(x_sicaklik, sicaklik_yuksek, input_sicaklik)

nm_dusuk = fuzz.interp_membership(x_nem, nem_dusuk, input_nem)
nm_normal = fuzz.interp_membership(x_nem, nem_normal, input_nem)
nm_yuksek = fuzz.interp_membership(x_nem, nem_yuksek, input_nem)

by_kucuk = fuzz.interp_membership(x_boyut, boyut_kucuk, input_boyut)
by_orta = fuzz.interp_membership(x_boyut, boyut_orta, input_boyut)
by_buyuk = fuzz.interp_membership(x_boyut, boyut_buyuk, input_boyut)

# --- 5. القواعد الـ 15 وتفعيلها (Inference Engine) ---
r1 = np.fmin(np.fmin(st_dusuk, nm_dusuk), fan_dusuk)
r2 = np.fmin(np.fmin(st_dusuk, nm_normal), fan_dusuk)
r3 = np.fmin(np.fmin(st_dusuk, nm_yuksek), fan_orta)
r4 = np.fmin(np.fmin(st_normal, nm_dusuk), fan_dusuk)
r5 = np.fmin(np.fmin(st_normal, nm_normal), fan_orta)
r6 = np.fmin(np.fmin(st_normal, nm_yuksek), fan_yuksek)
r7 = np.fmin(np.fmin(st_yuksek, nm_dusuk), fan_orta)
r8 = np.fmin(np.fmin(st_yuksek, nm_normal), fan_yuksek)
r9 = np.fmin(np.fmin(st_yuksek, nm_yuksek), fan_yuksek)
r10 = np.fmin(np.fmin(st_normal, by_kucuk), fan_dusuk)
r11 = np.fmin(np.fmin(st_normal, by_orta), fan_orta)
r12 = np.fmin(np.fmin(st_normal, by_buyuk), fan_yuksek)
r13 = np.fmin(np.fmin(st_yuksek, by_kucuk), fan_orta)
r14 = np.fmin(np.fmin(st_yuksek, by_orta), fan_yuksek)
r15 = np.fmin(np.fmin(st_yuksek, by_buyuk), fan_yuksek)

# دمج القواعد المفعلة (Aggregation)
out_dusuk = np.fmax(r1, np.fmax(r2, np.fmax(r4, r10)))
out_orta = np.fmax(r3, np.fmax(r5, np.fmax(r7, np.fmax(r11, r13))))
out_yuksek = np.fmax(r6, np.fmax(r8, np.fmax(r9, np.fmax(r12, r14))))

aggregated = np.fmax(out_dusuk, np.fmax(out_orta, out_yuksek))

# --- 6. إلغاء العشوائية / إزالة الغموض (Defuzzification - Centroid) ---
try:
    defuzz_fan = fuzz.defuzz(x_fan, aggregated, 'centroid')
except AssertionError:
    defuzz_fan = 0

# --- 7. العرض المرئي على الواجهة (GUI) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Grafiksel Gösterim (Üyelik Fonksiyonları)")
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    
    axs[0].plot(x_sicaklik, sicaklik_dusuk, 'b', label='Düşük')
    axs[0].plot(x_sicaklik, sicaklik_normal, 'g', label='Normal')
    axs[0].plot(x_sicaklik, sicaklik_yuksek, 'r', label='Yüksek')
    axs[0].axvline(x=input_sicaklik, color='purple', linestyle='--')
    axs[0].set_title('Sıcaklık')
    axs[0].legend()
    
    axs[1].plot(x_nem, nem_dusuk, 'b', label='Düşük')
    axs[1].plot(x_nem, nem_normal, 'g', label='Normal')
    axs[1].plot(x_nem, nem_yuksek, 'r', label='Yüksek')
    axs[1].axvline(x=input_nem, color='purple', linestyle='--')
    axs[1].set_title('Nem')
    axs[1].legend()

    axs[2].plot(x_boyut, boyut_kucuk, 'b', label='Küçük')
    axs[2].plot(x_boyut, boyut_orta, 'g', label='Orta')
    axs[2].plot(x_boyut, boyut_buyuk, 'r', label='Büyük')
    axs[2].axvline(x=input_boyut, color='purple', linestyle='--')
    axs[2].set_title('Oda Boyutu')
    axs[2].legend()

    axs[3].fill_between(x_fan, np.zeros_like(x_fan), aggregated, facecolor='orange', alpha=0.5)
    axs[3].plot(x_fan, fan_dusuk, 'b', linestyle='--')
    axs[3].plot(x_fan, fan_orta, 'g', linestyle='--')
    axs[3].plot(x_fan, fan_yuksek, 'r', linestyle='--')
    axs[3].axvline(x=defuzz_fan, color='black', linewidth=2)
    axs[3].set_title(f'Çıktı: Fan Hızı (Centroid: %{defuzz_fan:.2f})')
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("🎯 Hesaplanan Çıktı Sonucu")
    st.metric(label="Önerilen Fan Hızı", value=f"% {defuzz_fan:.2f}")
    
    st.subheader("📜 Aktif Kural Listesi")
    st.markdown(f"""
    * **IF** Sıcaklık({input_sicaklik}) Düşük **AND** Nem({input_nem}) Düşük **THEN** Fan=Düşük
    * **IF** Sıcaklık({input_sicaklik}) Normal **AND** Nem({input_nem}) Normal **THEN** Fan=Orta
    * **IF** Sıcaklık({input_sicaklik}) Yüksek **AND** Nem({input_nem}) Yüksek **THEN** Fan=Yüksek
    * **IF** Sıcaklık({input_sicaklik}) Normal **AND** Oda Boyutu({input_boyut}) Büyük **THEN** Fan=Yüksek
    * *(Arka planda toplam 15 kural kontrol edilmektedir).*
    """)

    