# MT5 AI Trading Bot

O'zini o'zi o'qituvchi algoritmik trading bot MT5 platformasi uchun.

## O'rnatish

1. Python 3.11 o'rnating
2. Requirements o'rnating:
```bash
pip install -r requirements.txt
```
3. MetaTrader 5 o'rnating va account sozlang

## Ishga tushirish

### Training rejimida:
```bash
python main.py train --symbol EURUSD --timeframe 5 --episodes 100
```

Parametrlar:
- `--symbol`: Trading simvoli (default: EURUSD)
- `--timeframe`: Timeframe minutlarda (default: 5)
- `--episodes`: Training episodlar soni (default: 100)

### Trading rejimida:
```bash
python main.py trade --symbol EURUSD --model models/latest_model.h5
```

Parametrlar:
- `--model`: O'qitilgan model fayli
- `--symbol`: Trading simvoli
- `--timeframe`: Timeframe

## Monitoring

- Logs papkasida kunlik log fayllar
- Training holatini `training_state.json` da kuzatish mumkin
- Failed episodelar `failed_episodes.json` da saqlanadi

## Strategiyalar

Bot 3 xil strategiyadan foydalanadi:
1. Scalping (qisqa muddatli)
2. Breakout (trend break)
3. OrderBlock (trend davomiyligida)

## Performance Optimizatsiya

Bot CPU core soniga qarab:
- Batch size avtomatik moslashadi
- Memory limit optimallashadi
- Data processing parallel bajariladi

## Xavfsizlik

- Har bir savdo uchun risk menejment
- Maximum risk: account balansining 1%
- Stop-loss va Take-profit dinamik hisoblanadi

## Model Saqlash

Models papkasida:
- `latest_model.h5`: Eng so'nggi model
- `model_episode_N.h5`: Har 100 episoddan keyin
- `interrupted_model.h5`: Ctrl+C bilan to'xtatilganda