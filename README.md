# HVAC Health Analysis — SPC Dashboard
## Beximco Pharmaceuticals | AHU FF-075 | Unit-03

### Setup (MacBook)

1. Open Terminal
2. Navigate to this folder:
   ```
   cd Desktop/beximco_hvac_spc
   ```

3. Install dependencies:
   ```
   pip3 install -r requirements.txt
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

5. Browser opens automatically at http://localhost:8501

### Project Structure
```
beximco_hvac_spc/
├── app.py           ← Main Streamlit dashboard
├── ahu_data.csv     ← Parsed FF-075 EMS data (9,217 readings)
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

### Dashboard Tabs
1. SPC Control Chart — X-bar and MR charts with GMP limits
2. Process Capability — Cp, Cpk, Cpl, Cpu, Pp, Ppk with distributions
3. Event Analysis — Root cause classification + anomaly heatmap
4. Daily Trend — All parameters over 33 days
5. ML Anomaly Detection — Isolation Forest with 3D feature space
6. Project Report — Full written analysis for submission
