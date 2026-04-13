# BakeSquad LangGraph Evaluation Report
**Generated:** 20260412_153206
**Model:** `claude` / `claude-sonnet-4-20250514`

---

## Stage 1A — Classification Accuracy

| Metric | Value |
|--------|-------|
| Total cases | 72 |
| Correct | 61 |
| Accuracy | 84.7% |

**By group:**

| Group | Total | Correct | Accuracy |
|-------|-------|---------|----------|
| ambiguous | 5 | 3 | 60.0% |
| clear_cake | 8 | 7 | 87.5% |
| clear_cookie | 10 | 9 | 90.0% |
| clear_pastry | 1 | 1 | 100.0% |
| clear_quick_bread | 6 | 6 | 100.0% |
| clear_yeasted_bread | 3 | 3 | 100.0% |
| edge_case | 6 | 6 | 100.0% |
| known_issue | 5 | 5 | 100.0% |
| long_query | 4 | 4 | 100.0% |
| medium_query | 6 | 4 | 66.7% |
| preference_rich | 8 | 6 | 75.0% |
| scoring_transparency | 4 | 2 | 50.0% |
| short_query | 6 | 5 | 83.3% |

---

## Stage 1B — Preference/Constraint Extraction Recall

| Metric | Value |
|--------|-------|
| Cases tested | 12 |
| Avg constraint recall | 1.0 |
| Avg preference recall | 0.958 |
| Flour type accuracy   | 1.0 |

**Per-case results:**

| ID | Query | Constraint recall | Pref recall | Flour OK |
|----|-------|-------------------|-------------|----------|
| TC_LQ01 | I want chocolate chip cookies that are chewy  | 100% | 100% | — |
| TC_LQ02 | Looking for a banana bread recipe that stays  | 100% | 100% | — |
| TC_LQ03 | I need a rich, very moist chocolate layer cak | 100% | 100% | — |
| TC_LQ04 | I want bakery-style blueberry muffins with hu | 100% | 100% | — |
| TC_PR01 | chocolate chip cookies without any nuts | 100% | 100% | YES |
| TC_PR02 | moist banana bread that stays soft for at lea | 100% | 100% | YES |
| TC_PR03 | gluten-free chocolate chip cookies using almo | 100% | 100% | YES |
| TC_PR04 | vegan chocolate cake without any dairy produc | 100% | 100% | YES |
| TC_PR05 | oil-based banana bread only, no butter recipe | 100% | 100% | YES |
| TC_PR06 | completely nut-free birthday cake for a schoo | 100% | 100% | YES |
| TC_PR07 | paleo banana bread with almond flour and no r | 100% | 100% | YES |
| TC_PR08 | super chewy brown butter cookies with flaky s | 100% | 50% | YES |

---

## Stage 1C — Query Specificity Analysis

Hypothesis: longer queries → more constraints/preferences extracted.

| Group | n | Avg constraints | Avg preferences | Constraint recall | Pref recall |
|-------|---|-----------------|-----------------|-------------------|-------------|
| short_query | 6 | 0.33 | 0.0 | n/a | n/a |
| medium_query | 6 | 0.67 | 1.17 | n/a | n/a |
| long_query | 4 | 3.0 | 5.0 | 100.0% | 100.0% |

---

## Stage 1D — Turn Classification Routing

4/4 sequences fully correct.

**TC_MT01** [OK] Filter by fat type, then ask a factual question
  - OK  `new_search` (expected `new_search`)  — 'chocolate chip cookies'
  - OK  `re_filter` (expected `re_filter`)  — 'only show me oil-based ones'
  - OK  `factual` (expected `factual`)  — 'why does oil make cookies stay softer longer than '

**TC_MT02** [OK] Exclude ingredient then re-search with different intent
  - OK  `new_search` (expected `new_search`)  — 'moist banana bread'
  - OK  `re_filter` (expected `re_filter`)  — 'find me one without walnuts'
  - OK  `re_search` (expected `re_search`)  — 'actually I want chocolate banana bread instead'

**TC_MT03** [OK] Factual question mid-session, then narrow results
  - OK  `new_search` (expected `new_search`)  — 'blueberry muffins'
  - OK  `factual` (expected `factual`)  — 'what does the leavening ratio number mean for muff'
  - OK  `re_search` (expected `re_search`)  — 'find me a low-sugar version of these'

**TC_MT04** [OK] Re-search with dietary restriction
  - OK  `new_search` (expected `new_search`)  — 'red velvet cake'
  - OK  `re_search` (expected `re_search`)  — 'actually find me a dairy-free version instead'

---

## Stage 2 — Scoring Transparency

| Metric | Value |
|--------|-------|
| Pipeline runs OK | 5 |
| Runs with errors | 0 |
| Category reconciliation events | 0 |
| Novel technique delta events | 0 |
| Avg technique signal recall | 0.083 |

**Most frequent technique signals extracted:**

- `double_leavening`: 3
- `brown_butter`: 1

**Scoring traces:**

### TC_ST01 — brown butter chocolate chip cookies chilled overnight before
- Category: `step1` → `cookie`
- Top recipe: Brown Butter Salted Chocolate Chip Cookies (75.3/100)
- Technique signals: brown_butter
- Expected signals: ['brown_butter', 'chill_dough', 'overnight_rest']  found: ['brown_butter']  missed: ['chill_dough', 'overnight_rest']  recall: 33%

  **#1 Brown Butter Salted Chocolate Chip Cookies** — composite: 75.3/100  (weights from: defaults)
    - Moisture Retention: 62/100 (w=0.45)
    - Structure & Leavening: 83/100 (w=0.30)
    - Sugar Balance: 100/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: none

  **#2 Healthier Browned Butter Chocolate Chip Cookies** — composite: 70.9/100  (weights from: defaults)
    - Moisture Retention: 42/100 (w=0.45)
    - Structure & Leavening: 97/100 (w=0.30)
    - Sugar Balance: 100/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: none

  **#3 Nelly's Rippled Salted Browned Butter Chocolate Ch** — composite: 54.7/100  (weights from: defaults)
    - Moisture Retention: 42/100 (w=0.45)
    - Structure & Leavening: 38/100 (w=0.30)
    - Sugar Balance: 100/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: brown_butter

### TC_ST02 — moist banana bread using the fold method with rested batter
- Category: `step1` → `quick_bread`
- Top recipe: Classic Banana Bread Recipe (83.0/100)
- Technique signals: double_leavening
- Novel technique note: _Recipe emphasizes resting the batter for improved texture and flavor development._
- Expected signals: ['fold_method', 'batter_rest']  found: []  missed: ['fold_method', 'batter_rest']  recall: 0%

  **#1 Classic Banana Bread Recipe** — composite: 83.0/100  (weights from: defaults)
    - Moisture Retention: 93/100 (w=0.85)
    - Structure & Leavening: 42/100 (w=0.20)
    - Sugar Balance: 100/100 (w=0.15)
    - Technique Quality: 50/100 (w=0.10) | signals: none

  **#2 Unlock the Secret to Perfect Banana Bread: Rest Yo** — composite: 74.2/100  (weights from: defaults)
    - Moisture Retention: 76/100 (w=0.85)
    - Structure & Leavening: 60/100 (w=0.20)
    - Sugar Balance: 100/100 (w=0.15)
    - Technique Quality: 50/100 (w=0.10) | signals: none | note: Recipe emphasizes resting the batter for improved texture an

  **#3 Gordon Ramsay Banana Bread Recipe That Always Work** — composite: 68.2/100  (weights from: defaults)
    - Moisture Retention: 61/100 (w=0.85)
    - Structure & Leavening: 93/100 (w=0.20)
    - Sugar Balance: 86/100 (w=0.15)
    - Technique Quality: 50/100 (w=0.10) | signals: none

  **#4 Overnight Banana Bread** — composite: 61.1/100  (weights from: defaults)
    - Moisture Retention: 69/100 (w=0.85)
    - Structure & Leavening: 22/100 (w=0.20)
    - Sugar Balance: 77/100 (w=0.15)
    - Technique Quality: 50/100 (w=0.10) | signals: double_leavening

### TC_ST03 — New York style cheesecake baked in a water bath at low tempe
- Category: `step1` → `cake`
- Top recipe: Rich & Creamy New York Cheesecake (62.3/100)
- Technique signals: none
- Expected signals: ['water_bath', 'bake_low']  found: []  missed: ['water_bath', 'bake_low']  recall: 0%

  **#1 Rich & Creamy New York Cheesecake** — composite: 62.3/100  (weights from: defaults)
    - Moisture Retention: 70/100 (w=0.45)
    - Structure & Leavening: 65/100 (w=0.30)
    - Sugar Balance: 50/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: none

### TC_ST04 — red velvet cupcakes with cream cheese frosting using room te
- Category: `step1` → `cake`
- Top recipe: Red Velvet Cupcakes (61.2/100)
- Technique signals: double_leavening
- Expected signals: ['cream_method', 'room_temp_butter']  found: []  missed: ['cream_method', 'room_temp_butter']  recall: 0%

  **#1 Red Velvet Cupcakes** — composite: 61.2/100  (weights from: defaults)
    - Moisture Retention: 45/100 (w=0.45)
    - Structure & Leavening: 60/100 (w=0.30)
    - Sugar Balance: 96/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: none

  **#2 Red Velvet Cupcakes with Cream Cheese Frosting** — composite: 38.8/100  (weights from: defaults)
    - Moisture Retention: 44/100 (w=0.45)
    - Structure & Leavening: 60/100 (w=0.30)
    - Sugar Balance: 0/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: none

  **#3 Red Velvet Cupcakes with Cream Cheese Frosting** — composite: 37.6/100  (weights from: defaults)
    - Moisture Retention: 41/100 (w=0.45)
    - Structure & Leavening: 60/100 (w=0.30)
    - Sugar Balance: 0/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: double_leavening

### TC_SQ01 — cookies
- Category: `step1` → `cookie`
- Top recipe: The Most Amazing Chocolate Chip Cookies (59.9/100)
- Technique signals: double_leavening

  **#1 The Most Amazing Chocolate Chip Cookies** — composite: 59.9/100  (weights from: defaults)
    - Moisture Retention: 52/100 (w=0.45)
    - Structure & Leavening: 42/100 (w=0.30)
    - Sugar Balance: 99/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: double_leavening

  **#2 Thin and Crispy Oatmeal Cookies** — composite: 11.9/100  (weights from: defaults)
    - Moisture Retention: 12/100 (w=0.45)
    - Structure & Leavening: 0/100 (w=0.30)
    - Sugar Balance: 11/100 (w=0.25)
    - Technique Quality: 50/100 (w=0.10) | signals: double_leavening

---

## Stage 3 — DAG vs LangGraph Comparison

_Run with `--compare-langgraph` to enable Stage 3._

---

*Generated by evaluate_langgraph.py*