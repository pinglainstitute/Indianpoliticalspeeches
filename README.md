# A Computational Framework for Tracking Jargon, Style, and Multi-Dimensional Ad Hominem Rhetoric in Political Discourse

This repository contains an end-to-end computational linguistics and text-mining pipeline designed to track, classify, and analyze the rhetorical and stylistic evolution of political discourse over extended longitudinal timelines. 

Operating at the intersection of **Computational Linguistics**, **Political Communication**, and **Machine Learning**, this project moves away from traditional monolithic text tracking. It introduces granular, multi-label classifications of *ad hominem* attacks, tracks shifts in individual vs. collective pronoun framing, measures lexical complexity, and uses machine learning to scale semantic categorization across large speech corpora.

---

## ⚙️ Core Architecture & Project Foundations

The project systematically resolves four primary challenges in computational political science through a modular python pipeline:

### 1. Longitudinal Political Jargon Tracking
* **The Problem:** Political rhetoric shifts dynamically over time in response to socio-economic crises, legislative transitions, and electoral cycles. Manually discovering and tracking the systematic emergence, peak, and decay of national/populist buzzwords across thousands of dense transcripts is highly inefficient.
* **Project Resolution:** The data processing layer aggregates speeches by calendar year and applies an engineered Term Frequency-Inverse Document Frequency (`TfidfVectorizer`) configured to capture multi-word sequences (`ngram_range=(2, 3)`). By adjusting max feature thresholds, the script filters out universal baseline noise and isolates year-specific jargon (e.g., automatically capturing phrases like `digital india` in 2014 or `corona pandemic` in 2020).

### 2. Multi-Dimensional Ad Hominem Profiling
* **The Problem:** Traditional sentiment tools treat hostile rhetoric or political incivility as a single, negative monolith. This completely obscures the specific rhetorical vectors being weaponized—such as questioning an opponent's financial transparency versus targeting their family dynasty.
* **Project Resolution:** We engineered a **10-Point Analytical Framework** mapped to localized regular expressions (`re`) to decompose transcripts into distinct dimensions:
    1.  **Character Integrity:** Focused on honesty and transparency (e.g., *corrupt, fake, fraud*).
    2.  **Personal Traits:** Highlighting character flaws (e.g., *arrogant, ego, weak*).
    3.  **Ridicule/Mockery:** Converting debate into farce (e.g., *laughable, clown, ridiculous*).
    4.  **Derogatory Language:** Straightforward targeted labeling (e.g., *fool, stupid, liar*).
    5.  **Misconduct/Criminality:** Allegations of illegal actions (e.g., *scam, bribe, loot, mafia*).
    6.  **Emotional Outbursts:** Direct expressions of explicit outrage (e.g., *hate, disgusting, shame*).
    7.  **Incompetence/Fitness:** Doubting operational capabilities (e.g., *incompetent, clueless, unfit*).
    8.  **Family/Personal Life:** Highlighting lineage and nepotism (e.g., *dynasty, nepotism, son, daughter*).
    9.  **Physical Appearance/Health:** Critiques focusing on age or visible stature (e.g., *sick, old, health*).
    10. **Guilt by Association:** Alleging a shadow network or corrupt nexus (e.g., *puppet, handler, syndicate*).

### 3. Stylistic Fingerprinting & Pacing Metrics
* **The Problem:** Quantifying the shift between populist collective messaging ("We" / "Us" / "Countrymen") and individualistic ego-framing ("I" / "Me" / "My") over a political career, while tracking overall structural pacing.
* **Project Resolution:** The script extracts and scales individual-to-collective pronoun ratios (`I_Ratio (%)` vs. `We_Ratio (%)`), tracks sentence boundaries to isolate average sentence lengths (`Avg_Sentence_Length`), and maps vocabulary diversity through mathematical **Shannon Entropy Modeling**.

### 4. Weak Supervision & Machine Learning Scalability
* **The Problem:** Manual annotation is impossible at scale across massive document databases, yet pure unsupervised clustering models lack the categorical focus needed to evaluate concrete political science theories.
* **Project Resolution:** The pipeline builds a weak-supervision engine. It leverages the 10-point keyword regex metrics to automatically bootstrap training targets across the corpus (tagging text blocks as an attack or standard policy statement). This auto-labeled matrix is then passed into an optimized machine learning pipeline—utilizing a `LinearSVC` classifier tuned via a 5-fold cross-validation `GridSearchCV`—which effectively scales up and classifies entirely unseen text segments.

---

## 📊 Empirical Discoveries & Data Hurdles

As the machine learning architecture transitioned from basic binary classification to an advanced multi-label framework—where **10 cloned classifiers run in parallel** to track all attack dimensions simultaneously—a clear data hurdle was uncovered.

### The Data Sparsity & Imbalance Paradox

The multi-label evaluation report revealed very high F1-scores for categories like `Q9_Physical_Appearance` (0.79) and `Q8_Family_Personal_Life` (0.67), but threw complete **0.00 F1-scores** for categories such as `Q1_Character_Integrity`, `Q3_Ridicule_Mockery`, and `Q10_Guilt_By_Association`.

The cause of this drop-off is directly tied to the dataset's class distribution profile:

* **The High-Support Triggers:** Keyword matching functions highly effectively on blunt, literal categories. Markers for family lineage or physical age are vocabulary-stable and highly recurring, providing ample training data for the model.
* **The Sparse Classes:** More complex rhetorical attacks—like mocking an opponent or implying guilt by association—are often highly ironic, sarcastic, or syntactically complex. Because these structures avoid simple keywords, the rigid regex dictionary missed them, leaving categories starved with **only 1 or 2 positive samples** in the entire test split. 
* **The Technical Conclusion:** A machine learning model cannot find a reliable mathematical boundary based on 1 or 2 samples. This data sparsity directly informs the next development milestone for this project.

---

## 🚀 Repository Roadmap: Next Research Extensions

To scale this pipeline into a high-impact, publication-grade research paper, development is moving toward three core extensions:

### 1. Shifting to Semantic Zero-Shot Transformers
To rescue the `0.00` F1-scores caused by keyword dependency, we are replacing the rigid string-matching labeling step with **Natural Language Inference (NLI) Zero-Shot classification** using deep-learning transformers (e.g., `BART-Large-MNLI`). 

Instead of searching for exact substrings, the pipeline will evaluate if a block of text semantically satisfies the *concept* of your target category. This allows the framework to correctly tag subtle or sarcastic attacks even if they do not contain a single keyword from a pre-defined list.

### 2. Deep Learning Baseline Benchmarking
We are upgrading our classification layer by fine-tuning specialized deep sequence models (such as `RoBERTa-political` or `DistilBERT`). This allows us to construct a formal model comparison matrix within the final paper:

$$\text{Classical Bag-of-Words Pipeline (LinearSVC + TF-IDF)} \quad \longleftrightarrow \quad \text{Context-Aware Deep Learning Transformer}$$

This baseline comparison adds strong technical and experimental rigor favored by major machine learning and computational science journals.

### 3. Integrated Stylistic-Semantic Regression Modeling
The project will unify the results of our **Stylistic Fingerprinting layer** and our **Multi-Label Attack Classifier**. By piping these metrics into an OLS linear regression model or an ANOVA test, we will formally evaluate the following core behavioral hypothesis:

$$\mathcal{H}_1: \text{The implementation of high-density personal attacks correlates with a statistically significant reduction in syntactic text complexity.}$$

This step allows you to prove mathematically whether a speaker's language structurally and measurably simplifies the moment they transition from discussing policy to delivering a targeted personal attack.

---

## 🛠️ Getting Started

### Installation
Ensure you have Python 3.10+ installed along with the required core dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
