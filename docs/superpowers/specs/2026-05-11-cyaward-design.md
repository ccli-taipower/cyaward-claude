# cyaward-claude — Design Spec

**Date:** 2026-05-11
**Status:** Approved (brainstorming complete, awaiting plan)
**Author:** ccli

## 1. 系統概觀

### 一句話定位
每天自動抓 pybaseball 數據，用訓練於 2015-2024 BBWAA 投票結果的迴歸模型，預測「若賽季今天結束，每位投手會拿多少賽揚票」，輸出 AL/NL 各 Top 10 排名網頁與每週 markdown 報告。

### 核心使用者旅程
1. **平日**：用戶任何時間打開 GitHub Pages 網址 → 看到當下 AL Top 10 + NL Top 10 表格，附預測 vote share、投影終值。
2. **每週一台灣時間早上 10 點**：repo 自動 commit 一份 `reports/2026-Wxx.md` 週報，記錄「過去一週名次變動」「升降幅最大者」「值得關注的新進榜」。
3. **賽季末**：對照真實 BBWAA 投票結果，model 自我驗證；retrain 為下一季 v2 模型。

### 系統三層
```
[Data Layer]      pybaseball → cache parquet (本地 + Actions cache)
       ↓
[Model Layer]     Stage A: stat projector  →  Stage B: vote share regressor
       ↓
[Presentation]    static HTML (dashboard) + markdown (weekly report)
       ↓
[Delivery]        GitHub Actions cron → Git commit → GitHub Pages
```

### MVP 邊界
**做**：兩聯盟 Top 10 排名表 + sparkline 趨勢線 + 每週報告。
**不做**：互動權重滑桿、投手詳情頁、email 通知、季前預測、信賴區間、Vegas odds 對照、其他獎項擴充。這些是 v2+。

---

## 2. 資料層

### 抓取對象（透過 pybaseball）

| 來源 | 抓什麼 | 用途 |
|---|---|---|
| `pitching_stats` (FanGraphs) | IP, GS, K, BB, ERA, FIP, xFIP, WHIP, K-BB%, fWAR, ERA-, FIP- | 主特徵 + 投影目標 |
| `statcast_pitcher` 聚合 | xERA, xwOBA against, Stuff+, Location+, Pitching+, Barrel%, Hard Hit% | Statcast 進階特徵 |
| `pitching_stats_bref` | bWAR, W-L, CG, ShO, SV, HLD | W-L 與 saves（投票者仍重視） |
| Lahman / chadwick `awards` | 歷年 Cy Young 投票完整結果 | 訓練 label |

### 快取策略
- pybaseball 內建 cache（`pybaseball.cache.enable()`，預設 `~/.pybaseball/`）
- 額外 layer：本地 parquet `data/raw/pitching_2026_YYYY-MM-DD.parquet`，每天一份 snapshot
- GitHub Actions 用 `actions/cache` 快取 `~/.pybaseball/`

### 歷史快照保存
- `data/raw/` 每日 parquet commit 進 git（檔案小，每日 ~50KB）
- `data/predictions/2026-MM-DD.parquet` model output，commit 進 git
- 累積後可繪「過去 30 天 Skubal 預測 vote share 趨勢線」

### 速率控制
- 一天一次 cron
- 失敗重試 3 次，遞增 backoff（1min / 5min / 15min）
- pybaseball 因 FanGraphs 改版壞掉時，Actions 失敗 → GitHub 預設 email 通知

### 目錄結構
```
cyaward-claude/
├── data/
│   ├── raw/                          # 每日 snapshot parquet
│   ├── historical/                   # 2015-2024 訓練資料（一次性）
│   └── predictions/                  # 每日 model output
├── src/
│   ├── fetch.py                      # pybaseball wrappers
│   ├── projector.py                  # Stage A: 投影模型
│   ├── voter_model.py                # Stage B: 得票模型
│   ├── ranker.py                     # 串接 + 過濾 + 排名
│   ├── render.py                     # HTML/markdown 生成
│   └── weekly_report.py              # 週報生成
├── models/
│   └── voter_model_v1.pkl            # 訓練好的 sklearn pipeline
├── site/                             # GitHub Pages root
│   ├── index.html                    # 每日重建
│   └── style.css
├── reports/
│   └── 2026-Wxx.md                   # 每週報告
├── .github/workflows/
│   ├── daily.yml                     # 每日抓資料 + 重建網頁
│   └── weekly.yml                    # 每週生成週報
├── notebooks/                        # 模型開發 / 探索性分析
├── tests/
├── docs/
│   └── superpowers/specs/
└── README.md
```

---

## 3. 模型層

### Stage A: 賽季投影模型 (Projector)

**目標**：把今日累計數據投影到「該投手以類似節奏走完賽季的最終值」。

**MVP v1: Pace × Remaining**
```python
projected_IP   = current_IP / current_team_games × 162
projected_K    = current_K  / current_IP × projected_IP
projected_xERA = current_xERA          # rate stat 直接沿用
projected_fWAR = current_fWAR / current_IP × projected_IP
```
- SP：用 `team_games` 估剩餘 starts
- RP：用 `appearance rate per team game` 估剩餘出場
- **已知缺點**：完全不收斂到聯盟均值；新手投手早季 ERA 0.50 會被外推到全季 0.50（不現實）。Demo 用文字揭露此限制，由 README 說明。

**抽象介面**（為 v2 Marcel 預留）：
```python
class Projector(ABC):
    def project(self, current: pd.DataFrame, asof_date: date) -> pd.DataFrame: ...

class PaceProjector(Projector): ...      # MVP
class MarcelProjector(Projector): ...    # v2，不在本 MVP
```

### Stage B: 得票預測模型 (Voter Model)

**訓練資料**：
- 2015-2024 排除 2020 = 9 個賽季
- 每年取每聯盟「IP ≥ 50 的所有投手」（含未得票者，讓模型學完整分布）
- 每行: `(pitcher, year, league, features..., vote_share)`
- `vote_share = total_points_received / 210`（210 = 7×30 max possible）
- 沒得票者 vote_share = 0
- 預估約 1440 行

**特徵**：

| 類別 | 特徵 |
|---|---|
| 傳統 | W, L, ERA, IP, K, BB, WHIP, CG, ShO, SV |
| Sabermetric | fWAR, FIP, xFIP, K-BB%, ERA-, FIP- |
| Statcast | xERA, xwOBA against, Stuff+, Location+, Barrel%, Hard Hit% |
| Context | role (SP/RP one-hot), league (AL/NL one-hot), team_winning_pct |

**模型**：
- **MVP**: `GradientBoostingRegressor` (sklearn)
- 同時保留 `Ridge` 作為 baseline / sanity check
- **不選 XGBoost/LightGBM**：避免額外 dependency；資料量小差異不大

**驗證**：
- Time-series split: train 2015-2022 / val 2023 / test 2024
- 指標：
  - Vote share MAE
  - **Top-3 hit rate**（業務 KPI：預測 Top 3 跟真實 Top 3 重疊幾個）
  - 真實 winner 落在預測 Top 5 的比率
- Leave-one-year-out CV 看穩定性

**訓練時機**：
- 一次性訓練 → `models/voter_model_v1.pkl`，commit 進 repo
- 每年 11 月 BBWAA 結果出爐後手動 retrain → v2、v3 ...

### 入榜資格 (Eligibility Filter)

SP/RP 分組動態縮放：
```python
days_elapsed   = today - season_start         # season_start 約 3/27
season_progress = days_elapsed / 183
sp_min_ip = max(25, 162 × season_progress)
rp_min_ip = max(10, 60  × season_progress)
```
- SP 判定: `GS / G > 0.5`
- RP 判定: `GS / G ≤ 0.5`

### 預測管線串接

```python
def rank_today(asof_date: date) -> pd.DataFrame:
    current   = fetch.current_season_stats(asof_date)
    eligible  = filter_eligibility(current, asof_date)
    projected = PaceProjector().project(eligible, asof_date)
    predicted_share = voter_model.predict(projected)
    win_prob  = calibrator.predict_proba(predicted_share)
    return assemble_ranking(eligible, projected, predicted_share, win_prob)
```

### 輸出 schema

| Column | 說明 |
|---|---|
| `pitcher_name`, `team`, `league`, `role` | 基本資料 |
| `current_IP`, `current_ERA`, `current_xERA`, `current_fWAR` | 今日累計關鍵指標 |
| `proj_IP`, `proj_ERA`, `proj_fWAR` | Stage A 投影終值 |
| `predicted_vote_share` | Stage B 預測（0-1） |
| `predicted_rank_in_league` | 該聯盟內排名 |
| `win_probability` | 校準後奪冠機率（%）|
| `delta_7d`, `delta_30d` | 排名 7 日 / 30 日變動 |

### Win Probability 校準
- 第一版：用 isotonic regression 在 2015-2024 train set 上學「vote_share → was_winner (0/1)」對應
- 不訓練第二個複雜模型；只是把連續 vote_share 校準成「奪冠機率」直觀數字

---

## 4. Presentation 與 Delivery

### 網頁儀表板 (`site/index.html`)

Vanilla HTML + Jinja2 模板（無 React、無 build pipeline）。

**頁面結構**：
- Header：標題、截至日期、資料更新時間
- 兩欄並排（mobile 直排）：AL Top 10 / NL Top 10
- 每個投手 card：
  - 排名 + 名字 + 隊伍
  - 預測 vote share % + 7 日排名變動 (🟢↑ / 🔴↓ / ─)
  - 關鍵 stats: IP, xERA, fWAR
  - Sparkline（過去 30 天 vote_share 趨勢）— 純 inline SVG，無 lib
- Footer：方法論、原始碼連結

**渲染**：
- `render.py` 用 Jinja2 把 prediction parquet 注入 HTML template
- Sparkline data inline 進 SVG（無 fetch）
- 整檔 ~50KB，無外部 dep；零 JS 也能看（漸進增強）

### 每週報告 (`reports/2026-Wxx.md`)

Markdown 模板，每週一台灣時間 10:00 自動生成。內容：
- 本週榜首（AL + NL，附預測 vote share 與奪冠機率）
- 升降幅最大者（升 + 降各 3-5 位）
- 新進榜
- 即將出榜邊緣（#11-15）
- Top 10 完整名單（兩個 markdown 表格）
- 模型小註（資料規模、最近的 model-vs-voter 分歧投手）

### GitHub Actions

**`.github/workflows/daily.yml`** — `cron: '0 11 * * *'`（每日 UTC 11:00 = 台灣 19:00）：
1. `actions/cache` restore `~/.pybaseball/`
2. `pip install -r requirements.txt`
3. `python -m src.fetch --date today`
4. `python -m src.ranker --date today`
5. `python -m src.render --output site/index.html`
6. commit `data/raw/*.parquet`, `data/predictions/*.parquet`, `site/index.html`
7. push → GitHub Pages 自動 redeploy
8. 失敗 retry 3 次（exponential backoff）

**`.github/workflows/weekly.yml`** — `cron: '0 2 * * 1'`（週一 UTC 02:00 = 台灣週一 10:00）：
1. `python -m src.weekly_report --week current`
2. commit `reports/2026-Wxx.md`
3. push

**失敗處理**：失敗那天**不 commit**，網頁保留前一天版本。GitHub 預設發 email 通知。

### 測試策略

**Unit tests** (pytest)：
- `test_projector.py`：mock current stats → pace projector 輸出符合預期
- `test_voter_model.py`：載入 pkl model → 在預先存好的 test set 上 assert metric 在 baseline 之上
- `test_ranker.py`：mock pipeline output → 排名邏輯正確（含 SP/RP 過濾）
- `test_render.py`：mock prediction df → HTML / markdown 包含預期字段

**Integration test**：
- `test_pipeline.py`：固定歷史日期（2024-05-11）跑完整 pipeline，assert AL/NL Top 1 合理（非空、IP 達門檻、vote share ∈ [0, 1]）

**Smoke test in Actions**：
- workflow 結束前 `pytest tests/test_render_output.py`，assert `site/index.html` 含 "Top 10"、"AL"、"NL"

---

## 5. MVP 範圍與後續路線

### MVP v1 — 包含項目

- ✅ pybaseball 抓 FanGraphs + Statcast + Lahman awards（歷史）
- ✅ Pace × Remaining 賽季投影模型
- ✅ GradientBoosting voter model（2015-22 train / 23 val / 24 test）
- ✅ SP/RP 動態 IP 門檻
- ✅ Daily cron 重建 `site/index.html`（兩欄 Top 10 + sparkline）
- ✅ Weekly cron 生成 `reports/2026-Wxx.md`
- ✅ Predictions parquet 累積進 git
- ✅ Unit tests + 1 integration test + smoke test
- ✅ README 說明方法論、資料來源、限制

### 明確不做（不在 MVP）

- ❌ 互動權重滑桿
- ❌ 投手詳情頁（SHAP 解釋、game log）
- ❌ Email/Slack 通知
- ❌ 季前預測
- ❌ 信賴區間 / uncertainty bands
- ❌ Vegas / FanDuel odds 對照欄位
- ❌ MVP / Rookie of the Year 等其他獎項
- ❌ Marcel-style 投影（介面預留，實作留 v2）

### v2+ 路線（依優先序）

| Priority | Item | Value |
|---|---|---|
| P1 | Marcel projector 取代 pace | 早季預測準度顯著提升 |
| P1 | 信賴區間（quantile regression） | 早季 Top 10 邊緣更可信 |
| P2 | 投手詳情頁 + SHAP 解釋 | 「為什麼模型看好他」透明化 |
| P2 | 互動權重滑桿 | 讓懷疑「W-L 重要」的人自己證偽 |
| P3 | Vegas odds 對照 | 模型 vs 市場共識 |
| P3 | Email / RSS 訂閱 | 重大異動推播 |
| P4 | MVP / RoY 同框架擴充 | 一魚多吃 |

### 風險與已知限制

| Risk | Impact | Mitigation |
|---|---|---|
| FanGraphs 改版 → pybaseball 壞 | 每日 pipeline 失敗 | retry + email + 失敗日不 commit |
| Pace projector 早季嚴重外推 | 4-5 月排名跳動大 | README 揭露；v2 換 Marcel |
| 訓練樣本只 9 年 | 對歷史罕見 profile 泛化弱 | Time-series CV 監控；新賽季 retrain |
| 2020 縮水賽季投票邏輯異常 | 模型混淆 | 排除 |
| BBWAA 投票偏見漂移 | 例如未來突然又重 W-L | 每年 retrain；Ridge baseline 對照 |
| pybaseball cache 體積 | repo 變大 | 只 commit 每日 snapshot；訓練 raw 不 commit |

### Definition of Done

1. `python -m src.ranker --date 2026-05-11` 在本機成功跑出 AL+NL Top 10
2. `python -m src.render --date 2026-05-11` 生成可在瀏覽器打開的 `site/index.html`
3. `python -m src.weekly_report --week current` 生成可讀的 markdown
4. GitHub Actions `daily.yml` 手動觸發成功 commit + Pages 部署
5. GitHub Actions `weekly.yml` 手動觸發成功生成週報
6. `pytest` 全綠（含 1 integration test）
7. README 寫清楚：怎麼跑、模型方法、資料來源、限制
8. **Voter model 在 2024 hold-out test set 上 Top-3 hit rate ≥ 2/3**（兩聯盟平均至少預測對 2 個 Top 3）

---

## 附錄 A — 主要技術選型摘要

| 決策 | 選擇 | 替代方案 / 理由 |
|---|---|---|
| 資料來源 | pybaseball | 一站式包裝 FanGraphs + Statcast + Bref + Lahman |
| 投影模型 | Pace × Remaining (MVP) | v2 換 Marcel；介面預留 |
| 投票模型 | GradientBoostingRegressor | XGBoost 對小資料無顯著優勢、增加 dep |
| 訓練年份 | 2015-2024（除 2020） | Statcast 從 2015；BBWAA 邏輯近年才轉向 sabermetric |
| 樣本構成 | 含未得票者（vote_share=0） | 讓模型學完整分布而非極端值 |
| 入榜門檻 | SP/RP 分組動態縮放 | RP 不會被 162-IP 門檻永遠擋外 |
| 前端 | Vanilla HTML + Jinja2 | 無 React / 無 build pipeline；v2 加互動再升級 |
| 部署 | GitHub Pages + GitHub Actions cron | 完全免費、零維運、自帶版本歷史 |
| 更新節奏 | Daily fetch + Weekly report | 數據新鮮度 vs commit noise 平衡 |
| 失敗處理 | 不 commit，保留前一天 | 避免顯示假資料 |
