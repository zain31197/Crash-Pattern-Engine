<div align="center">

# 🚦 TrafficRisk-USA
### *Spatiotemporal & Environmental Intelligence for Road Safety*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-7.7M%20Records-orange?style=flat-square)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
[![Coverage](https://img.shields.io/badge/Coverage-49%20US%20States-green?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)]()
[![Author](https://img.shields.io/badge/Author-Zain%20Shahid-red?style=flat-square)]()

> **Every year, over 38,000 people die on US roads.** Most analyses ask *where* crashes happen. This project asks *why* — and *when*, *under what conditions*, and *for whom it's most dangerous*.

[Problem Statement](#-the-problem) • [Dataset](#-dataset) • [Methodology](#-methodology) • [Visualizations](#-visualization-deep-dive) • [Key Findings](#-key-findings) • [Setup](#-setup--installation) • [Usage](#-usage)

</div>

---

## 🎯 The Problem

Road accident analysis has historically been reactive — emergency responders arrive *after* crashes happen, and safety improvements come *years* after patterns are documented. The core challenge is that **accidents are not random events**. They emerge from a predictable convergence of:

- **Time** (rush hour, weekday vs. weekend, season)
- **Environment** (temperature, visibility, weather conditions)
- **Geography** (urban density, highway type, regional infrastructure)
- **Infrastructure** (signals, junctions, crossings)

Despite this, most public safety dashboards treat accidents as isolated data points. This project builds a **multi-dimensional risk intelligence system** — transforming 7.7 million raw accident records into actionable, spatiotemporal risk patterns that can inform proactive intervention.

**The question this project answers:**  
> *"Given time, location, weather, and infrastructure conditions — what is the risk profile of any given road segment in the US?"*

---

## 📦 Dataset

| Attribute | Detail |
|-----------|--------|
| **Name** | US Accidents (2016–2023) |
| **Source** | Kaggle — Sobhan Moosavi et al. |
| **Volume** | ~7.7 Million Records |
| **Geographic Scope** | 49 US States |
| **Temporal Span** | February 2016 – March 2023 |
| **Core Features Used** | 15 variables across Location, Time, Environment, and Infrastructure |

**Selected Feature Dimensions:**
- 📍 **Location:** Lat, Lng, City, State
- ⏱️ **Time:** Start_Time → decomposed to Hour, Month, Weekday
- 🌤️ **Environment:** Temperature(F), Visibility(mi), Weather_Condition, Precipitation(in)
- 🛣️ **Infrastructure:** Traffic_Signal, Junction, Crossing → Hazard Score (0–3)

---

## ⚙️ Methodology

### Data Preprocessing

**Missing Value Strategy:**
- `Precipitation (~28% missing)` → Imputed as `0.0` based on the domain assumption that absent sensor readings indicate dry conditions
- Rows with missing `Start_Time`, `City`, or `Weather_Condition` were dropped entirely — justified by dataset scale ensuring no statistical bias

**Feature Engineering:**
```python
# Temporal decomposition
df['Hour']    = df['Start_Time'].dt.hour
df['Month']   = df['Start_Time'].dt.month
df['Weekday'] = df['Start_Time'].dt.day_name()

# Weather bucketing (high-cardinality → 7 groups)
weather_map = {
    'Fair': 'Clear', 'Clear': 'Clear',
    'Rain': 'Rain', 'Drizzle': 'Rain',
    'Snow': 'Snow', 'Blizzard': 'Snow',
    'Fog': 'Fog', 'Haze': 'Fog',
    'Cloudy': 'Cloudy', 'Overcast': 'Cloudy'
}

# Hazard Score: infrastructure complexity index
df['Hazard_Score'] = df['Traffic_Signal'].astype(int) + \
                     df['Junction'].astype(int) + \
                     df['Crossing'].astype(int)
```

**Analytical Techniques Used:**
| Technique | Purpose |
|-----------|---------|
| KDE Heatmaps | Geospatial severity density |
| K-Means Clustering (K=15) | Hotspot identification |
| Hierarchical Clustering | State-level accident profile similarity |
| Sankey / Alluvial Diagrams | Multi-condition flow to severity outcomes |
| Ridgeline Plots | Temporal distribution by weather type |
| Contour Mapping (2D Topographic) | Temperature × Visibility interaction surface |
| Correlation Matrix | Environmental factor independence testing |

---

## 📊 Visualization Deep Dive

### 1. Accident Frequency Heatmap — Hour × Day of Week

![Accident Frequency Heatmap](12.png)

The temporal fingerprint of accidents is unmistakable. Weekday mornings (6–9 AM) and evenings (3–6 PM) are the highest-risk windows — a direct reflection of commuter activity. Critically, the **risk is not weather-driven** at this scale; it is **human-schedule-driven**. Weekends show a flatter, mid-day distribution consistent with leisure travel.

**Intervention implication:** Dynamic speed limit enforcement and alert systems should activate specifically during the 7–9 AM and 4–6 PM weekday windows.

---

### 2. USA Accident Severity Zones — High-Risk Cluster Analysis

![USA Accident Severity Zones](11.png)

A severity-weighted heatmap across the continental US reveals three structural zones:
- **Eastern Seaboard:** Near-continuous high-severity corridor (population density-driven)
- **West Coast Corridors:** Major north-south intensity along I-5, I-101
- **Central Void:** Mountain West and Great Plains show dramatically lower density

The Eastern density is not just a volume problem — it represents a systemic infrastructure challenge where congestion compounds severity.

---

### 3. Weather Impact Analysis — Severity Breakdown

![Weather Impact Analysis](10.png)

One of the most counter-intuitive findings in the dataset: **88% of accidents occur in Clear (45%) or Cloudy (43%) conditions.** Rain accounts for just 8%, while Snow and Fog together contribute ~5%.

This does not mean weather is irrelevant — it means **driver behavioral overconfidence in good conditions** is a larger systemic risk than adverse weather itself. Severity 2 (moderate) dominates across all weather types (74–80%).

---

### 4. Multi-Dimensional Risk Flow — Conditions → Severity

![Multi-Dimensional Risk Flow](7.png)

This Sankey diagram traces the path from **Lighting → Weather → Signal Presence → Impact Severity**. Key structural flow:
- The widest bands originate from "Day" + "Clear/Cloudy" → confirming the volume finding above
- A significant flow passes through **Signal: False** — indicating highways (without signals) generate more accidents than intersections
- "Night" time conditions disproportionately route toward Severity 3+ outcomes

---

### 5. Correlation Matrix — Environmental Factors vs. Severity

![Correlation Matrix](8.png)

The correlation matrix reveals a critical analytical truth: **environmental variables are near-independent predictors of severity.** The strongest correlation with severity is `Traffic_Signal` at just -0.12 — and it's *negative*, meaning signal-equipped intersections correlate with *lower* severity accidents.

| Variable | Correlation with Severity |
|----------|--------------------------|
| Temperature | -0.02 |
| Visibility | -0.00 |
| Precipitation | +0.01 |
| Hour | +0.02 |
| Traffic Signal | **-0.12** |

This independence justifies the need for **multivariate interaction modeling** rather than simple regression.

---

### 6. Top 15 States and High-Risk Cities

![Top 15 States](9.png)

Texas, Florida, and California form the "Big Three" — not only in state-level volume but in urban concentration. Houston, Miami, and Los Angeles each carry accident loads exceeding any comparable metro elsewhere. Notably:
- **TX and FL** concentrate risk in 2–3 cities each
- **MI and PA** have high totals but distributed across more cities
- **NC** shows a sharp Charlotte-dominant pattern with Raleigh as secondary

---

### 7. Hourly Environmental Trends vs. Accident Frequency

![Hourly Environmental Trends](6.png)

Plotting Avg_Temp, Avg_Visibility, and Accident_Count on the same log-scale timeline across days of the week produces a decisive result: **temperature and visibility remain nearly flat across all 24 hours, while accident frequency shows dramatic rush-hour spikes.**

This decouples the environmental narrative from the behavioral one — accidents cluster at times of *human activity concentration*, not environmental degradation.

---

### 8. Dynamic Environmental Risk — Hourly Weather & Severity

![Dynamic Environmental Risk](5.png)

The animated bubble chart (shown here at Hour=0) plots **Average Temperature (x-axis) × Accident Volume (y-axis, log scale) × Visibility (bubble size) × Severity (color)**. 

Key observation: The largest bubbles (highest visibility) consistently sit at the *top* of the chart — confirming that **clear visibility = high volume**, not low risk. Severity peaks cluster around 55°F, suggesting speed-confidence conditions.

---

### 9. Critical Segment Identification — K-Means Clustering (K=15 Hotspots)

![K-Means Clustering](4.png)

K-Means with K=15 identifies national hotspot clusters differentiated by both **volume** and **average severity**. The dark red (severity ~2.35+) cluster in the Midwest represents a particularly dangerous profile: relatively lower accident volume but disproportionately high severity. This matches the rural/high-speed hypothesis — fewer accidents, but more fatal ones.

West Coast clusters show the inverse: extremely high volume at lower average severity, consistent with slow urban-crawl traffic.

---

### 10. Hierarchical Clustering — State Similarity by Accident Profile

![Hierarchical Clustering](3.png)

Ward-linkage hierarchical clustering groups US states by their accident profile similarity (not just count). The red dashed line at dissimilarity ~3.5 produces meaningful clusters:

- 🟠 **ID, OR, ME, WV, MN:** Rural/Northern cluster — winter weather, low density
- 🔴 **TX, LA, TN, FL, AZ, SC, CA:** Sun Belt — warm weather, high-volume urban corridors
- 🟣 **IL, IN, NH, MI, OH, KY:** Midwest industrial corridor
- 🟤 **IA, WI, WY, VT:** Sparse rural cluster

This enables **policy grouping** — states within the same cluster can share intervention strategies.

---

### 11. Temporal Ridgeline Analysis — Accident Hour Distribution by Weather

![Temporal Ridgeline Analysis](2.png)

Ridgeline (joy) plots overlay accident hour distributions for each weather type. The key divergences:
- **Clear & Cloudy:** Classic bimodal distribution — two sharp commuter peaks at 7–8 AM and 4–5 PM
- **Fog:** Pronounced early-morning density (6–9 AM) when fog formation is highest
- **Snow:** A **flattened, wide curve** — accidents distributed throughout the day, reflecting the sustained difficulty of snow driving vs. the punctual risk of commuter conditions

---

### 12. Topographic Risk Map — 2D Contours of Accident Severity

![Topographic Risk Map](1.png)

The contour map plots **Temperature (x) × Visibility (y) → Avg. Severity (z, color)**. The non-linear, island-like contours disprove any simple linear relationship. Most strikingly:

- **High visibility (>8 mi) + High temperature (>80°F):** Severity *increases* — the "speed overconfidence" zone
- **Low visibility + moderate temperature:** Severity remains moderate — drivers compensate behaviorally
- **Lower-right zones (high temp, low visibility):** Complex multi-factor interactions

This is the most data-sophisticated visualization in the suite — a genuine decision surface for risk modeling.

---

## 🔑 Key Findings

### Finding 1 — The Infrastructure Paradox
> Complex intersections (signals, crossings, junctions) **increase** accident *frequency* but **decrease** accident *severity*. Open highway segments with no signals are statistically **deadlier** per accident.

### Finding 2 — Visibility vs. Volume (Counter-Intuitive)
> Poor visibility is a **stronger predictor of fatal severity**, but the *majority* of accidents occur in clear conditions due to driver overconfidence and behavioral risk compensation.

### Finding 3 — Human Schedule Dominates Environmental Signal
> Rush-hour commute timing explains more variance in accident frequency than any combination of weather variables. The daily accident curve mirrors human work schedules, not weather cycles.

### Finding 4 — Midwest Severity Gap
> The Midwest hosts a unique risk profile: lower accident volume than coastal metros but significantly higher average severity — consistent with rural high-speed road infrastructure and limited emergency response capacity.

### Finding 5 — Regional Intervention Differentiation
> States cluster into meaningful policy groups. Northeast interventions should target **congestion management**; Midwest strategies should prioritize **speed controls and weather early warning systems**; Sun Belt policies should focus on **urban corridor design**.

---

## 🗂️ Project Structure

```
TrafficRisk-USA/
│
├── 📓 notebooks/
│   └── Traffic-Risk-USA-Pattern.ipynb.ipynb      # Full analysis pipeline
│
├── 📊 visualizations/                     # All 12 exported chart images
│   ├── 01_heatmap_hour_vs_day.png
│   ├── 02_severity_zones_usa.png
│   ├── 03_weather_severity_breakdown.png
│   ├── 04_multidimensional_flow.png
│   ├── 05_correlation_matrix.png
│   ├── 06_top15_states_cities.png
│   ├── 07_hourly_env_trends.png
│   ├── 08_dynamic_env_risk.png
│   ├── 09_kmeans_hotspots.png
│   ├── 10_hierarchical_clustering.png
│   ├── 11_ridgeline_weather.png
│   └── 12_topographic_risk_map.png
│
├── 📄 paper/
│   └── DAV_IEEE_Paper.pdf                 # Full technical paper
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab
- ~8 GB RAM recommended for full dataset

### Installation

```bash
# Clone the repository
git clone https://github.com/zain31197/Crash-Pattern-Engine.git
cd TrafficRisk-USA

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
scipy>=1.11.0
folium>=0.14.0
joypy>=0.2.6
jupyter>=1.0.0
```

---

## 🚀 Usage

### Running the Full Analysis

```bash
# Launch Jupyter
jupyter notebook notebooks/US_Accidents_Case_Study.ipynb
```

### Dataset Download

The dataset is not included due to size. Download from Kaggle:

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d sobhanmoosavi/us-accidents
unzip us-accidents.zip -d data/
```

### Reproducing Individual Visualizations

Each visualization section in the notebook is modular and independently executable. Cell tags follow the format:
- `# VIZ_01` — Heatmap
- `# VIZ_04` — K-Means clustering
- `# VIZ_10` — Hierarchical clustering

---

## 📐 Technical Architecture

```
Raw CSV (7.7M rows)
       │
       ▼
┌─────────────────┐
│  Preprocessing  │  Missing value imputation, type casting, deduplication
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Eng.    │  Temporal decomposition, weather bucketing, hazard score
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┬─────────────────┐
         ▼                  ▼                  ▼                 ▼
   Spatiotemporal      Statistical         Clustering        Network
    Analysis            Analysis            Analysis          Analysis
  (Heatmaps, KDE)   (Correlation,        (K-Means,        (Sankey Flow,
                     Ridgeline)          Hierarchical)     Alluvial)
         │                  │                  │                 │
         └──────────────────┴──────────────────┴─────────────────┘
                                    │
                                    ▼
                          12 Advanced Visualizations
                          + Policy Recommendations
```

---

## 📈 Metrics & Scale

| Metric | Value |
|--------|-------|
| Total Records Analyzed | ~7.7 Million |
| States Covered | 49 |
| Time Period | 2016–2023 (7 years) |
| Unique Cities | 11,000+ |
| Visualizations Produced | 12 |
| Clustering Configurations | K=2 to K=15 tested |
| Feature Variables | 15 core + 5 engineered |

---

## 🔮 Future Work

- [ ] **Real-Time Risk Scoring API** — Expose a REST endpoint that scores any lat/lng + time + weather combination against trained models
- [ ] **Predictive Severity Model** — XGBoost/LightGBM model trained on engineered features
- [ ] **Street-Level Analysis** — OpenStreetMap integration for road-type segmentation
- [ ] **Weather Forecast Integration** — Combine with NOAA forecast data for *prospective* risk windows
- [ ] **Interactive Dashboard** — Streamlit or Dash app for public exploration
- [ ] **Causal Inference** — Move beyond correlation to identify true causal drivers using DoWhy or CausalML

---

## 📬 Contact

**Zain Shahid**  
Data Scientist · Traffic Safety Analytics  
📧 [23i2582@isb.nu.edu.pk](mailto:23i2582@isb.nu.edu.pk)  
🔗 [GitHub](https://github.com/zain31197)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Dataset original source: [Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset." 2019](https://arxiv.org/abs/1906.05409)

---

<div align="center">

**If this project helped you understand road safety data better, consider leaving a ⭐**

*Built with the goal of turning accident statistics into lives saved.*

</div>
