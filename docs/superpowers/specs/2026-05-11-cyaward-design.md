# cyaward-claude — Design Spec

**Date:** 2026-05-11
**Status:** Approved (brainstorming complete, awaiting plan)
**Author:** ccli

## 0. 專案總覽與兩階段框架

本專案目標：建立一個能準確預測 MLB 賽揚獎的迴歸模型，並套用到 2026 賽季即時資料上。

採取**兩階段路線**：

### Phase 1 — Model & Backtest（本 spec 的 MVP）
建立並驗證 vote share 迴歸模型，用近 10 年（2015-2025 排除 2020）BBWAA 投票結果做 backtest，**只有當 model 通過驗證標準時才進入 Phase 2**。

### Phase 2 — Live Dashboard（Phase 1 通過後才做）
把驗證過的 model 套上 2026 即時資料，產出 GitHub Pages 排名儀表板與每週報告。

> **本 spec 詳細描述兩階段的設計，但 implementation plan 只涵蓋 Phase 1。**
> Phase 2 預留為設計參考；待 Phase 1 通過 KPI 後再啟動 Phase 2 的 plan。

### 為什麼兩階段
若 model 在歷史上根本預測不準，後續 dashboard 工程全是白工。先驗證再蓋基礎設施。

---

## 1. Phase 1：Model & Backtest

### 1.1 資料層

**抓取對象**（透過 pybaseball）：

| 來源 | 抓什麼 | 用途 |
|---|---|---|
| `pitching_stats` (FanGraphs) | IP, GS, K, BB, ERA, FIP, xFIP, WHIP, K-BB%, fWAR, ERA-, FIP-, **RS/9** | 主特徵 |
| `statcast_pitcher` 聚合 | xERA, xwOBA against, Stuff+, Location+, Pitching+, Barrel%, Hard Hit% | Statcast 進階特徵 |
| `pitching_stats_bref` | bWAR, W-L, CG, ShO, SV, HLD | W-L 與 saves（投票者仍重視） |
| `team_records` | 各隊各年 final winning pct | Team context |
| Lahman / chadwick `awards` | 歷年 Cy Young 投票完整結果 | 訓練 label |

**訓練資料構造**：
- 2015-2025 排除 2020 = **10 個賽季**
- 每年取每聯盟「IP ≥ 50 的所有投手」（含未得票者，讓模型學完整分布）
- 每行: `(pitcher, year, league, features..., vote_share)`
- `vote_share = total_points_received / 210`（210 = 7×30 max possible）
- 沒得票者 vote_share = 0
- 實際約 3421 行（10 年 × 2 聯盟 × ~170 投手/年）

**快取策略**：
- pybaseball 內建 cache（`pybaseball.cache.enable()`，預設 `~/.pybaseball/`）
- 訓練資料一次性產生：`data/historical/training_2015_2025.parquet`，commit 進 git
- 後續 retrain 直接讀 parquet，不再打 FanGraphs

### 1.2 模型層

**特徵清單**（26 個：10 傳統 + 6 sabermetric + 6 Statcast + 4 context）：

| 類別 | 特徵 |
|---|---|
| 傳統 | W, L, ERA, IP, K, BB, WHIP, CG, ShO, SV |
| Sabermetric | fWAR, FIP, xFIP, K-BB%, ERA-, FIP- |
| Statcast | xERA, xwOBA against, Stuff+, Location+, Barrel%, Hard Hit% |
| Team context | role (SP/RP one-hot), league (AL/NL one-hot), team_winning_pct, **run_support_per_9** |

註 1：`team_winning_pct` 訓練時用 final，推論時用今日當下值；voter 在 11 月投票時也是看 final，所以邏輯一致。
註 2：`run_support_per_9` 投手的「打線支援」，FanGraphs 提供。投票者雖嘴上說 W-L 不重要，歷史投票仍受其隱含影響；此特徵讓 model 能解構 W-L 中的非投手成分。

**模型選擇**：
- **主模型**：`GradientBoostingRegressor` (sklearn)
- **Baseline**：`Ridge` regression（同一 pipeline，調參數）
- **不選 XGBoost/LightGBM**：避免額外 dependency；資料量小差異不大
- 兩個都跑，比較 metric。Phase 2 預設用 Gradient Boosting；若 Ridge 反而更穩定就改用 Ridge。

**訓練輸出**：
- `models/voter_model_gbr_v1.pkl` — Gradient Boosting 主模型
- `models/voter_model_ridge_v1.pkl` — Ridge baseline
- `models/calibrator_v1.pkl` — isotonic regression，把 vote_share 映射到「奪冠機率」（Phase 2 用）

### 1.3 驗證 KPI（三層）

預測賽揚獎的核心張力是「在 elite 群裡誰最強」，所以 Top-10 命中只是底線；真正的 KPI 是冠軍與頒獎台還原。

每個聯盟每年產生 1 個冠軍、1 組 Top 3、1 組 Top 10。10 年 × 2 聯盟 = **20 個獨立的 (year, league) cases**。

| Tier | 指標 | **MVP 通過標準** |
|---|---|---|
| 🥇 Tier 1：冠軍命中 | `predicted_top1 == actual_top1` 命中次數 / 18 | **≥ 14 / 18 (78%)** |
| 🥈 Tier 2：Podium 還原 | 18 次中，predicted Top 3 與 actual Top 3 集合重疊數的平均（不論順序） | **≥ 1.9 / 3 (63%)** |
| 🥉 Tier 3：Top 10 還原 | 18 次中，predicted Top 10 與 actual Top 10 集合重疊數的平均 | **≥ 7 / 10 (70%)** |

**輔助指標**（不是通過標準，但要報告）：
- Vote share MAE
- 真實冠軍落在預測 Top 5 的比率
- Spearman rank correlation between predicted and actual within Top 10
- Outlier list：歷史上預測冠軍 ≠ 真實冠軍的案例（哪幾年、誰被高估、誰被低估）

### 1.4 驗證方法（雙軌）

**主軌：Leave-One-Year-Out CV**
- 9 個 fold；每個 fold 留一年當 test，用其他 8 年訓練
- 收集 10 年的預測，串成 10 年 × 2 聯盟 = 20 個 (year, league) cases，計算上述三層 KPI
- **這是判定 Phase 1 是否通過的主要依據**

**副軌：Time-Series Split**
- Train 2015-2022 / val 2023 / test 2024
- 額外驗證「model 是否會隨時間衰退」（voter behavior 漂移檢測）
- 報告但不是通過門檻

### 1.5 入榜資格 (Eligibility Filter)

訓練資料用固定門檻：**IP ≥ 50** 即進入訓練集（含未得票者 vote_share = 0）。
這個門檻在 Phase 1 用即可；Phase 2 才需要動態縮放（針對賽季進行中）。

### 1.6 Phase 1 Definition of Done

通過以下**全部**才算 Phase 1 完成：

1. `data/historical/training_2015_2025.parquet` 成功生成（≥ 3000 rows，含所有特徵且無 critical NaN）
2. `python -m src.train` 成功訓練 GradientBoosting + Ridge + calibrator，輸出三個 pkl
3. `python -m src.backtest --method loocv` 成功跑完 9 fold，輸出驗證報告 `reports/backtest_v1.md`
4. **Backtest 三層 KPI 全部達標**：
   - 🥇 Tier 1 (冠軍命中) ≥ 15/20
   - 🥈 Tier 2 (Podium 平均) ≥ 1.9/3
   - 🥉 Tier 3 (Top 10 平均) ≥ 7/10
5. Outlier 案例（model 預測錯的年份）有書面分析（`reports/backtest_v1.md` 的「Outlier Analysis」章節）
6. `pytest` 全綠（unit tests for fetch、feature engineering、model train、backtest metric）
7. README.md（Phase 1 部分）寫清楚：怎麼跑、資料來源、特徵清單、KPI 結果

**未通過怎麼辦**：
- 若主要 KPI 未達標 → 進入 model iteration loop（換特徵、換演算法、修 bug、檢查 data leak）
- **不進入 Phase 2** 直到通過為止

---

## 2. Phase 2：Live Dashboard（Phase 1 通過後啟動）

> 本節是設計藍圖，**不在當前 MVP 的 implementation plan 內**。

### 2.1 即時資料抓取

每日抓 2026 賽季當下累計 stats（FanGraphs + Statcast + Bref），存：
- `data/raw/pitching_2026_YYYY-MM-DD.parquet`（每日 snapshot）
- `data/predictions/2026-MM-DD.parquet`（每日 model output）
- 都 commit 進 git，累積後可繪歷史趨勢線

### 2.2 投影管線（Stage A：Pace × Remaining）

```python
projected_IP   = current_IP / current_team_games × 162
projected_K    = current_K  / current_IP × projected_IP
projected_xERA = current_xERA          # rate stat 直接沿用
projected_fWAR = current_fWAR / current_IP × projected_IP
```

抽象介面（為 v3 Marcel 預留）：
```python
class Projector(ABC):
    def project(self, current: pd.DataFrame, asof_date: date) -> pd.DataFrame: ...

class PaceProjector(Projector): ...      # Phase 2 MVP
class MarcelProjector(Projector): ...    # v3
```

**已知缺點**：完全不收斂到聯盟均值；新手投手早季 ERA 0.50 會被外推到全季 0.50（不現實）。README 揭露此限制。

### 2.3 入榜門檻（動態縮放）

```python
SEASON_START = date(2026, 3, 26)
SEASON_END   = date(2026, 9, 27)
days_elapsed    = (today - SEASON_START).days
season_progress = days_elapsed / 183
sp_min_ip = max(25, 162 * season_progress)
rp_min_ip = max(10,  60 * season_progress)
```
- SP 判定: `GS / G > 0.5`；RP 判定: `GS / G ≤ 0.5`
- 季前（today < SEASON_START）→ 不出榜，網頁顯示「season hasn't started」

### 2.4 預測管線串接

```python
def rank_today(asof_date: date) -> pd.DataFrame:
    current   = fetch.current_season_stats(asof_date)
    eligible  = filter_eligibility(current, asof_date)
    projected = PaceProjector().project(eligible, asof_date)
    predicted_share = voter_model.predict(projected)
    win_prob  = calibrator.predict_proba(predicted_share)
    return assemble_ranking(eligible, projected, predicted_share, win_prob)
```

### 2.5 輸出 schema

| Column | 說明 |
|---|---|
| `pitcher_name`, `team`, `league`, `role` | 基本資料 |
| `current_IP`, `current_ERA`, `current_xERA`, `current_fWAR` | 今日累計關鍵指標 |
| `proj_IP`, `proj_ERA`, `proj_fWAR` | Stage A 投影終值 |
| `predicted_vote_share` | Stage B 預測（0-1） |
| `predicted_rank_in_league` | 該聯盟內排名 |
| `win_probability` | 校準後奪冠機率（%） |
| `delta_7d`, `delta_30d` | 排名 7 日 / 30 日變動 |

### 2.6 網頁儀表板 (`site/index.html`)

Vanilla HTML + Jinja2（無 React、無 build pipeline）：
- Header：標題、截至日期、資料更新時間
- 兩欄並排（mobile 直排）：AL Top 10 / NL Top 10
- 每個投手 card：排名、名字、隊伍、預測 vote share %、7 日排名變動、IP/xERA/fWAR、sparkline (≥ 3 天才顯示)
- Footer：方法論、原始碼連結
- 整檔 ~50KB，零 JS 也能看

### 2.7 每週報告 (`reports/2026-Wxx.md`)

每週一台灣時間 10:00 自動生成。內容：
- 本週榜首（AL + NL）
- 升降幅最大者（升 + 降各 3-5 位）
- 新進榜
- 即將出榜邊緣（#11-15）
- Top 10 完整名單
- 模型小註

### 2.8 GitHub Actions

- `daily.yml` cron `0 11 * * *`（每日 UTC 11:00 = 台灣 19:00）：抓資料 → 重建網頁 → commit → push
- `weekly.yml` cron `0 2 * * 1`（週一 UTC 02:00 = 台灣週一 10:00）：生成週報 → commit → push
- 失敗 retry 3 次（exponential backoff）；失敗那天**不 commit**，網頁保留前一天版本

### 2.9 Phase 2 Definition of Done（待 Phase 1 通過後啟動）

1. `python -m src.ranker --date 2026-05-11` 在本機成功跑出 AL+NL Top 10
2. `python -m src.render --date 2026-05-11` 生成可在瀏覽器打開的 `site/index.html`
3. `python -m src.weekly_report --week current` 生成可讀的 markdown
4. GitHub Actions `daily.yml` 與 `weekly.yml` 手動觸發成功
5. GitHub Pages 部署完成、URL 可訪問
6. Phase 2 unit tests + integration test + smoke test 全綠
7. README Phase 2 部分完成

---

## 3. 目錄結構

```
cyaward-claude/
├── data/
│   ├── historical/
│   │   ├── training_2015_2025.parquet     # Phase 1 訓練資料
│   │   └── awards_history.parquet         # Cy Young 投票歷史
│   ├── raw/                                # Phase 2: 每日 snapshot
│   └── predictions/                        # Phase 2: 每日 model output
├── src/
│   ├── fetch.py                # pybaseball wrappers + 訓練資料生成
│   ├── features.py             # 特徵工程（衍生 K-BB%、ERA-、joins、cleanup）
│   ├── voter_model.py          # 訓練 + 預測 model
│   ├── backtest.py             # LOOCV + Time-series split + KPI 計算
│   ├── projector.py            # Phase 2: Stage A 投影
│   ├── ranker.py               # Phase 2: 串接管線
│   ├── render.py               # Phase 2: HTML 生成
│   └── weekly_report.py        # Phase 2: 週報生成
├── models/
│   ├── voter_model_gbr_v1.pkl
│   ├── voter_model_ridge_v1.pkl
│   └── calibrator_v1.pkl
├── reports/
│   ├── backtest_v1.md          # Phase 1 KPI 報告 + outlier 分析
│   └── 2026-Wxx.md             # Phase 2: 每週報告
├── site/                       # Phase 2: GitHub Pages root
│   ├── index.html
│   └── style.css
├── .github/workflows/          # Phase 2
│   ├── daily.yml
│   └── weekly.yml
├── notebooks/                  # 模型開發 / 探索 / outlier 案例研究
├── tests/
├── docs/superpowers/specs/
├── requirements.txt
└── README.md
```

---

## 4. 風險與已知限制

| Risk | Phase | Impact | Mitigation |
|---|---|---|---|
| 訓練樣本 10 年 (~3421 行) | 1 | 對歷史罕見 profile 泛化弱 | LOOCV 嚴格驗證；Ridge baseline 對照 |
| 2020 縮水賽季投票邏輯異常 | 1 | 模型混淆 | 排除 |
| BBWAA 投票偏見漂移 | 1+2 | 例如未來突然又重 W-L | 每年 retrain；Ridge baseline 對照 |
| pybaseball 因 FanGraphs 改版壞掉 | 1+2 | 抓不到資料 | retry + Phase 2 失敗日不 commit |
| RS/9 與 W-L 高相關 → 共線性 | 1 | GradientBoosting 不嚴重，但 Ridge 受影響 | Ridge 用 L2 正則；報告兩個模型差異 |
| Cy Young 偶有「跌破眼鏡」結果（如 2024 NL Sale） | 1 | 部分年份 model 必輸 | KPI 不要求完美（15/20 而非 20/20） |
| Phase 1 KPI 未達標 | 1 | Phase 2 無法啟動 | model iteration loop（換特徵、換演算法）；可能加 ensemble |
| Pace projector 早季嚴重外推 | 2 | 4-5 月排名跳動大 | README 揭露；v3 換 Marcel |

---

## 5. v3+ 路線（Phase 2 通過後）

| Priority | Item | Value |
|---|---|---|
| P1 | Marcel projector 取代 pace | 早季預測準度顯著提升 |
| P1 | 信賴區間（quantile regression） | 早季 Top 10 邊緣更可信 |
| P2 | 投手詳情頁 + SHAP 解釋 | 「為什麼模型看好他」透明化 |
| P2 | 互動權重滑桿 | 讓懷疑「W-L 重要」的人自己證偽 |
| P3 | Vegas odds 對照 | 模型 vs 市場共識 |
| P3 | Email / RSS 訂閱 | 重大異動推播 |
| P4 | MVP / RoY 同框架擴充 | 一魚多吃 |

---

## 附錄 A — 主要技術選型摘要

| 決策 | 選擇 | 替代方案 / 理由 |
|---|---|---|
| 資料來源 | pybaseball | 一站式包裝 FanGraphs + Statcast + Bref + Lahman |
| 訓練年份 | 2015-2024（除 2020）| Statcast 從 2015；BBWAA 邏輯近年才轉向 sabermetric |
| 樣本構成 | IP ≥ 50 含未得票者 | 讓模型學完整分布而非極端值 |
| 特徵 | 26 個 (含 RS/9, team_winning_pct) | RS/9 解構 W-L 非投手成分 |
| 模型 | GradientBoosting + Ridge baseline | XGBoost 對小資料無顯著優勢、增加 dep |
| 驗證 | LOOCV 為主 + time-series split 為輔 | 10 年都被當過 test，最大化驗證樣本 |
| 通過標準 | 12/16 冠軍 + 1.9/3 podium + 7/10 top10 | 8 年 × 2 league = 16 cases；Tier 2 從 2.0 微調到 1.9 因為 1 個 podium swap 是統計噪聲 |
| 投影模型 (P2) | Pace × Remaining MVP | v3 換 Marcel；介面預留 |
| 入榜門檻 (P2) | SP/RP 分組動態縮放 | RP 不會被 162-IP 門檻永遠擋外 |
| 前端 (P2) | Vanilla HTML + Jinja2 | 無 React / 無 build pipeline |
| 部署 (P2) | GitHub Pages + Actions cron | 完全免費、零維運 |
| 更新節奏 (P2) | Daily fetch + Weekly report | 數據新鮮度 vs commit noise 平衡 |
