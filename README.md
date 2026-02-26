# Education Equity Policy Demo (Dataset A) — Streamlit Web App

โทนสี: **ส้ม + น้ำเงิน + ขาว** (พลัง/เด็ก–วัยรุ่น)  
เป้าหมาย: Demo วิเคราะห์เชิงนโยบายเพื่อลดความเหลื่อมล้ำทางการศึกษา (ไม่เหมารวม) โดยใช้ Dataset A

## 4 โมดูล (Tabs)
1) **EnrollScope** — การเข้าถึง/การอยู่ในระบบ (enrolled, attendance, device/internet, online)
2) **LearnPulse** — ผลการเรียนรู้ + ช่องว่าง (scores, gaps)
3) **PersistPath** — dropout / promotion / ความเสี่ยงสะสม
4) **EquityLens Lab** — “ใครเสียมากที่สุด” + Policy Simulation (what-if) + drill-down ดูรายการ

## ไฟล์ข้อมูล
วางไฟล์ **Dataset_A_2558_2567.xlsx** ไว้โฟลเดอร์เดียวกับ `app.py`  
หรืออัปโหลดผ่าน sidebar ในหน้าเว็บได้

## Run local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1) สร้าง GitHub repo แล้วอัปโหลดไฟล์ทั้งหมดในโฟลเดอร์นี้  
2) Streamlit Cloud → New app → เลือก repo  
3) Main file: `app.py`  
4) Deploy

## Notes
- Policy Simulation เป็น **what-if demo** (โมเดลเชิงสถิติแบบง่ายจาก Dataset A) ไม่ใช่ causal proof
