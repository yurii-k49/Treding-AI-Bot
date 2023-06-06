# R-DEV TREDING AI Savdo Boti

Sun'iy intellekt yordamida texnik tahlil, fundamental tahlil va sentiment tahlilini birlashtirib, MetaTrader 5 orqali avtomatlashtirilgan savdo qilish uchun ilg‘or savdo boti.

## 🌟 Asosiy xususiyatlar

### 📊 Texnik tahlil
- Bir nechta vaqt oralig‘ida tahlil qilish  
- 20+ texnik ko‘rsatkichlar  
- Patternlarni aniqlash  
- Narx harakati tahlili  
- Hajm tahlili  

### 📈 Fundamental tahlil
- Makroiqtisodiy ko‘rsatkichlar  
- Foiz stavkalarini tahlil qilish  
- YaIM va inflyatsiya ta'siri  
- Ish bilan bandlik bo‘yicha ma'lumotlar  
- Savdo balansi tahlili  

### 🤖 AI Modellar
- Patternlarni aniqlash uchun chuqur o‘rganish (Deep Learning)  
- Signal yaratish uchun mashina o‘rganishi (Machine Learning)  
- Yangiliklarni tahlil qilish uchun tabiiy tilni qayta ishlash (Natural Language Processing)  
- Optimizatsiya uchun mustahkamlovchi o‘rganish (Reinforcement Learning)  
- Real vaqt rejimida model yangilanishlari  

### 💹 Xavfni boshqarish
- Dinamik pozitsiya o‘lchamini belgilash  
- Murakkab stop-loss hisoblash  
- Foyda olishni optimallashtirish (Take-profit optimization)  
- Portfel korrelyatsiyasini tahlil qilish  
- Chuqur yo‘qotishlardan himoya  

## 🚀 O‘rnatish

### Tizim talablari
- Python 3.8+  
- MetaTrader 5  
- 8GB RAM (minimum)  
- Internet aloqasi  

### O‘rnatish bosqichlari

1. **Repository’ni klonlash**:
   ```bash
   git clone https://github.com/yourusername/ai-trading-bot.git
   cd ai-trading-bot
   ```

2. **Virtual muhit yaratish**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Paketlarni o‘rnatish**:
   ```bash
   pip install -r requirements.txt
   ```

4. **MT5 terminalini o‘rnatish va sozlash**

5. **Atrof-muhitni sozlash**:
   ```bash
   cp .env.example .env
   # .env faylini to‘ldirish va moslashtirish
   ```

## 🎯 Ishlatish


### AI ni o'qitish
```bash
python main.py learn
```

### Model tayyor bo'lgach uni yangi data bilan validatsiya qilish
```bash
python main.py validate
```

### Demo rejimda ishga tushirish
```bash
python main.py
```

### Real savdo rejimida ishga tushirish
```bash
MODE=real python main.py
```

### AI modellarni qayta o‘qitish
```bash
TRAIN=true python main.py
```

## 📊 Natijalarni kuzatish

Savdo natijalarini kuzatish uchun:  
1. Log fayllarni tahlil qilish  
2. Savdo hisobotlarini o‘rganish  
3. Model metrikalarini monitoring qilish  

## 🔧 Konfiguratsiya

### Savdo sozlamalari
- Vaqt oralig‘ini tanlash  
- Simvolni tanlash  
- Pozitsiya o‘lchamini belgilash  
- Xavf parametrlarini sozlash  

### AI Model sozlamalari
- Xususiyatlar yaratish (Feature engineering)  
- Model tanlash  
- Trening parametrlarini sozlash  
- Yangilanish chastotasi  

### Xavfni boshqarish
- Maksimal kunlik yo‘qotish miqdori  
- Pozitsiya o‘lchami cheklovlari  
- Korrelyatsiya cheklovlari  
- Chuqur yo‘qotishdan himoya  

## 📁 Loyihaning tuzilishi

```
trading_ai/
├── config/          # Konfiguratsiya fayllari
├── models/          # AI modellar
├── analysis/        # Tahlil modullari
├── trading/         # Savdo logikasi
├── utils/           # Yordamchi funksiyalar
└── data/            # Ma'lumotlar
```

## 🔍 Test qilish

### Unit testlar
```bash
pytest tests/
```

### Backtesting
```bash
python backtesting/run.py
```

## 📈 Performance optimizatsiyasi

Performance’ni yaxshilash uchun:  
1. Asinxron operatsiyalar  
2. Ma’lumotlarni kechiktirish (Data caching)  
3. Batch qayta ishlash  
4. Multi-threading  

## 🛠 Rivojlantirish

### Kod uslubi
- Black formatter  
- Flake8 linter  
- Type hints  
- Hujjatlashtirish  

### Hissa qo‘shish
1. Repository’ni fork qilish  
2. Yangi branch yaratish  
3. O‘zgarishlarni commit qilish  
4. Branch’ni push qilish  
5. Pull request yaratish  

## 📝 Litsenziya

Ushbu loyiha [MIT License](LICENSE) ostida litsenziyalangan. Batafsil ma’lumot uchun LICENSE faylini ko‘ring.

## ⚠️ Ogohlantirish

Savdo qilish xavfli jarayon. Ushbu bot faqat o‘quv maqsadlari uchun mo‘ljallangan. Foydalanish o‘zingizning xavf-xataringiz ostida amalga oshiriladi.

## 📧 Qo‘llab-quvvatlash

Qo‘llab-quvvatlash uchun:  
- Issue yaratish  
- Email yuborish  
- Jamoaga qo‘shilish  

## 🔄 Yangilanishlar

Doimiy yangilanishlar o‘z ichiga oladi:  
- Yangi funksiyalar  
- Xatoliklarni tuzatish  
- Modelni yaxshilash  
- Hujjatlarni yangilash  