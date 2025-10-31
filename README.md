# Lab Environmental Comfort Index (LECI)

### Predicting and Quantifying Environmental Comfort in Laboratory Zebrafish Tanks Using Synthetic Data and Machine Learning

---

## Phase 1 — Project Setup

### 1. Research Problem Definition

#### **Project Concept**

The **Lab Environmental Comfort Index (LECI)** project aims to develop a **species-agnostic computational framework** for quantifying environmental comfort in laboratory animal housing.

* **Initial Focus:** Zebrafish, a widely used vertebrate model in biomedical research.
* **Scalability:** Methodology is adaptable to other species, including rodents and rabbits.
* **Objective:** Provide an **objective, reproducible measure of tank environmental quality** to improve both welfare and experimental reliability.

---

### 2. Biological and Veterinary Rationale: Why Zebrafish?

Zebrafish serve as an ideal candidate for modeling environmental comfort due to their:

* High sensitivity to water quality parameters (temperature, dissolved oxygen, pH, etc.).
* Small size, high fecundity, and strong genetic similarity to humans.
* Proven reproducibility in experimental settings.

These attributes make zebrafish an optimal species for developing a scalable environmental comfort framework.

---

### 3. Environmental Comfort Index (ECI)

#### **Purpose**

To create a standardized **0–100 numerical score** reflecting the environmental comfort of zebrafish tanks.

#### **Comfort Categories**

| Category       | Score Range |
| -------------- | ----------- |
| Low Comfort    | 0–39        |
| Medium Comfort | 40–69       |
| High Comfort   | 70–100      |

#### **Features Included**

* **Water Quality:** Temperature, pH, dissolved oxygen, hardness, nitrogen load, flow rate.
* **Tank Environment:** Substrate, noise level, light/dark cycle, room humidity, stocking density.

#### **Exclusions (Phase 1)**

* Behavioral data.
* Enrichment activities.
* Inter-species interactions.

**Scope:** Phase 1 focuses exclusively on static environmental parameters to maximize reproducibility and reduce model complexity.

---

### 4. Machine Learning Framework

#### **Target Variable**

* Environmental Comfort Index (ECI) – continuous score between 0 and 100.

#### **Predictors**

* Environmental and housing features described above.

#### **Models Evaluated**

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosted Trees (**best performer**)

#### **Model Optimization**

* Cross-validation for generalization.
* Feature regularization to prevent overfitting.
* Exclusion of dynamic, non-static factors for model simplicity.

---

### 5. Significance and Impact

#### **Animal Welfare and the 3Rs**

* **Refinement:** Continuous, objective comfort measurement.
* **Reduction:** Higher reproducibility decreases the need for redundant studies.
* **Replacement (Indirect):** Strengthens zebrafish as a validated non-mammalian model.

#### **Veterinary Application**

Real-time, data-driven insights into zebrafish housing quality enable proactive welfare interventions.

#### **Scientific Reproducibility**

A standardized comfort index mitigates environmental variability, enhancing consistency across research sites.

#### **Operational Efficiency**

Automated environmental assessment minimizes manual data entry and subjective evaluation.

#### **Regulatory Alignment**

Provides **auditable, quantitative metrics** supporting IACUC and AAALAC compliance.

---

### 6. Continuity and Previous Work

LECI extends the foundation established by the **Lab Animal Growth Prediction** framework (murine model) and transitions from growth modeling to environmental comfort analysis.

Key advancements include:

* Use of realistic synthetic datasets.
* Expanded machine learning toolkit (Linear, Ridge, Random Forest, Gradient Boosted Trees).
* Identification of **Gradient Boosted Trees** as the most effective and reproducible model for deployment.

---

### 7. Software Environment Setup

#### **Core Libraries**

* `numpy`, `pandas` — data handling and numerical operations.

#### **Visualization**

* `matplotlib`, `seaborn` — exploratory data visualization.

#### **Modeling**

* `scikit-learn`:

  * Models: `LinearRegression`, `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`.
  * Evaluation: `train_test_split`, `cross_val_score`.
  * Preprocessing: `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`.
  * Interpretability: `PartialDependenceDisplay`, `set_config`.

#### **Explainability**

* `shap` — interpretable machine learning analysis.

#### **Utilities**

* `joblib`, `os`, `warnings` — model persistence, file management, warning suppression.

**Rationale:**
This modular and transparent environment ensures reproducible, explainable analyses in alignment with veterinary welfare and research reporting standards.

---

## Phase 2 — Data Creation and Exploration

### 1. Synthetic Dataset Generation

#### **Biological and Veterinary Rationale**

Zebrafish welfare is tightly linked to water quality stability.
Deviations in temperature, dissolved oxygen, or nitrogen levels induce physiological stress, leading to lower comfort scores.
To simulate realistic conditions, controlled random noise (±5%) was introduced to mimic sensor variability.

#### **Machine Learning Rationale**

Synthetic data enable model prototyping without reliance on sensitive facility data.
Balanced sampling across feature ranges prevents bias, while rule-based ECI scoring provides supervised regression targets.

#### **Dataset Overview**

* **Size:** 2,000 tank samples.
* **Goal:** Sufficient sample size for training/testing without overfitting.

#### **Feature Ranges**

| Feature                 | Range or Categories        |
| ----------------------- | -------------------------- |
| Temperature (°C)        | 26–28                      |
| Dissolved Oxygen (mg/L) | ≥6                         |
| pH                      | 6.8–7.5                    |
| Hardness (ppm)          | 50–150                     |
| Humidity (%)            | 40–60                      |
| Substrate               | Sand / Gravel / Bare Glass |
| Noise                   | Low / Medium / High        |
| Light/Dark              | 14/10, 12/12, 10/14        |
| Density (fish/L)        | 3–5                        |
| Nitrogen Load (mg/L)    | <50                        |
| Flow Rate               | Low / Medium / High        |

#### **Scoring Logic**

* Full points for optimal range.
* Partial deductions for mild deviations.
* Strong deductions for severe deviations.
* Final scores normalized to 0–100.

---

### 2. Data Quality Assurance

#### **Biological and Veterinary Checks**

* Verify that all environmental values remain within physiologically safe ranges.
* Detect unrealistic or outlier tank conditions prior to modeling.

#### **Machine Learning Quality Checks**

* Validate data integrity to prevent *garbage-in, garbage-out* issues.
* Confirm balanced representation across comfort categories (Low, Medium, High).
* Flag and handle outliers systematically.

---

### 3. Exploratory Data Analysis (EDA)

#### **Biological and Veterinary Perspective**

EDA visualizes environmental parameter distributions and identifies potential welfare stressors.
It ensures all variables remain within zebrafish safety thresholds before modeling.

#### **Machine Learning Perspective**

EDA identifies correlated features, skewed distributions, and outliers affecting model performance.
It informs preprocessing, feature selection, and model interpretability strategies.

#### **Core Analytical Tasks**

* Compute descriptive statistics (mean, median, standard deviation).
* Generate correlation heatmaps to identify feature relationships.
* Plot variable distributions and boxplots to visualize outliers.
* Use scatterplots to assess deviations from optimal environmental ranges.

---

## Phase 3 — Feature Preparation

### 1. Feature Encoding

#### **Biological and Veterinary Rationale**

Several environmental tank parameters are categorical in nature (e.g., substrate type, noise level, water flow).
While these categories must be numerically encoded for machine learning, their **biological meaning** must remain interpretable.

* **Substrate** influences stress levels and waste accumulation.
* **Noise level** affects chronic stress and behavioral stability.
* **Light/dark cycles** modulate circadian rhythms and endocrine responses.
* **Water flow** impacts oxygenation and swimming behavior.

#### **Machine Learning Rationale**

* One-hot encoding converts categorical variables into binary columns suitable for regression algorithms.
* The *drop-first* approach is applied to prevent multicollinearity in linear models.

#### **Implementation Tasks**

* Apply one-hot encoding to:

  * Substrate
  * Noise Level
  * Light/Dark Cycle
  * Water Flow

---

### 2. Derived Features (Feature Engineering)

#### **Biological and Veterinary Rationale**

Zebrafish thrive at approximately **27 °C** and **dissolved oxygen (DO) ≥ 6 mg/L**.
Although slight deviations in temperature or oxygen are individually tolerable, their **interaction** can lead to amplified physiological stress — for instance, warmer water naturally retains less oxygen.
Capturing this *synergistic stress effect* ensures the dataset reflects authentic biological responses.

#### **Machine Learning Rationale**

Raw features alone may not capture nonlinear biological relationships.
By engineering derived features based on domain knowledge, we embed biological insight directly into the data, improving **model interpretability** and **predictive robustness**.

#### **Computation Task**

Compute the **Thermal–Oxygen Stress Index (TOSI)** as follows:

$$
\text{Stress} = \lvert \text{Temperature} - 27 \rvert \times \max(0,, 6 - \text{Dissolved Oxygen})
$$

#### **Formula Explanation**

* ( \lvert \text{Temperature} - 27 \rvert ): Deviation from optimal thermal conditions.
* ( \max(0,, 6 - \text{DO}) ): Oxygen deficiency component, active only when DO < 6 mg/L.
* The product models combined physiological burden; if either variable is optimal, stress = 0.

#### **Interpretation**

Higher **TOSI** values indicate reduced comfort and are inversely correlated with the **Environmental Comfort Index (ECI)**.
This feature identifies tanks appearing normal by isolated parameters but exhibiting compounded biological stress when conditions interact.

---

### 3. Dataset Splitting

#### **Biological and Veterinary Rationale**

Evaluating the model on unseen data ensures it generalizes across diverse zebrafish housing conditions, reducing the risk of overfitting and supporting real-world welfare monitoring.

#### **Machine Learning Rationale**

* The dataset is divided into **training (80%)** and **testing (20%)** subsets.
* Stratification by comfort category (Low, Medium, High) maintains balanced biological representation across subsets, ensuring fair evaluation.

---

## Phase 4 — Modeling Framework

### 1. Baseline Comfort Index (Rule-Based)

#### **Biological and Veterinary Rationale**

Before machine learning implementation, a transparent **rule-based model** is established using standard zebrafish husbandry guidelines.
This step verifies that the system aligns with established welfare standards and facilitates veterinary validation.

#### **Machine Learning Rationale**

The rule-based index serves as a **baseline comparator** to assess whether ML regressors meaningfully improve predictive accuracy.

#### **Implementation Tasks**

* Apply defined environmental scoring rules and deductions.
* Normalize final scores to a 0–100 scale.
* Compare baseline results with ML predictions in subsequent evaluation.

---

### 2. Comfort Category Definition (Post-Processing)

#### **Biological and Veterinary Rationale**

Translating continuous ECI predictions into categorical comfort levels improves interpretability for animal care staff and facility operators.
Categories correspond to practical welfare monitoring thresholds.

#### **Machine Learning Rationale**

Categories are derived directly from regression outputs, avoiding a separate classification model.
This preserves pipeline simplicity and consistency with biological scoring.

| Comfort Category | ECI Range | Interpretation                                       |
| ---------------- | --------- | ---------------------------------------------------- |
| Low Comfort      | 0–39      | Suboptimal environment requiring urgent intervention |
| Medium Comfort   | 40–69     | Acceptable but warrants monitoring                   |
| High Comfort     | 70–100    | Optimal environmental conditions                     |

---

### 3. Supervised Machine Learning Models

#### **Biological and Veterinary Rationale**

Machine learning enables detection of complex, nonlinear interactions among environmental factors influencing zebrafish welfare.
This approach uncovers subtle welfare risks that static rule-based scoring may overlook.

#### **Machine Learning Rationale**

Four regression models were trained and compared:

| Model                   | Type                | Key Characteristics                                        |
| ----------------------- | ------------------- | ---------------------------------------------------------- |
| Linear Regression       | Parametric          | Simple and interpretable baseline.                         |
| Ridge Regression        | Regularized Linear  | Controls overfitting via L2 regularization.                |
| Random Forest Regressor | Ensemble Tree-Based | Captures nonlinear relationships and resists noise.        |
| Gradient Boosted Trees  | Boosted Ensemble    | High predictive power; best performer in cross-validation. |

#### **Overfitting Control Methods**

* **5-fold cross-validation** for generalization testing.
* **Regularization** (Ridge).
* **Depth and leaf constraints** for tree models.
* **Early stopping** for boosting algorithms.

#### **Model Training Tasks**

* Train all four regression models on the training set.
* Evaluate performance using:

  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * Coefficient of Determination (R²)
* Compare model outputs to rule-based ECI for validation and interpretability.

---

## Phase 5 — Evaluation and Interpretation

### 1. Model Performance Metrics

#### **Biological and Veterinary Rationale**

* **Linear Regression** and **Ridge Regression** explain approximately **50%** of the variation in comfort scores, indicating they are too simplistic to capture the nonlinear biological relationships underlying zebrafish welfare.
* **Random Forest** achieves a performance of roughly **74%**, demonstrating the advantage of models that capture feature interactions.
* **Gradient Boosted Trees** reach approximately **95% predictive accuracy**, making them the most reliable approach for quantifying environmental comfort.

#### **Machine Learning Rationale**

* Tree-based ensemble models clearly outperform linear ones due to their capacity to model nonlinearities.
* Ridge regularization provided minimal improvement, confirming that comfort is not a purely linear function of environmental parameters.
* Gradient Boosting achieved superior performance with less overfitting, benefiting from **sequential learning**, **regularization**, and **early stopping**.

#### **Implementation Tasks**

* Retain **Gradient Boosting** as the **primary production model**.
* Retain **Random Forest** as a **secondary benchmark model** for robustness testing.
* Archive **Linear** and **Ridge** models as baselines for documentation and reproducibility but exclude them from deployment.

---

### 2. Feature Importance Analysis

#### **Biological and Veterinary Rationale**

Understanding which environmental factors most influence zebrafish comfort provides actionable insights for refining husbandry protocols.
Model interpretation revealed the following biological hierarchy:

1. **Nitrogen Load (mg/L):** Primary driver of comfort — water quality directly impacts welfare.
2. **Temperature (°C):** Deviations from optimal range increase physiological stress.
3. **pH:** Perturbations affect homeostasis and gill function.
4. **Density (fish/L):** Stocking stress significantly affects comfort and behavior.
5. **Dissolved Oxygen (mg/L):** Important but less variable in this dataset.
6. **Thermal–Oxygen Stress Index (TOSI):** Reflects combined thermal and hypoxic stress.
   7–10. **Noise, Flow Rate, Light–Dark Cycle, Substrate:** Minor contributors in this model; may reflect longer-term or subtle welfare effects.

#### **Machine Learning Rationale**

Tree-based algorithms (Random Forest and Gradient Boosting) compute **feature importance scores**, revealing the contribution of each variable to ECI predictions.
This transparency strengthens scientific trust and aligns computational analysis with veterinary reasoning.

#### **Implementation Tasks**

* Extract feature importance values from the Gradient Boosting model.
* Rank features by contribution to predictive performance.
* Visualize rankings using a horizontal bar plot for interpretability and presentation.

---

### 3. Partial Dependence and SHAP Analysis

#### **Biological and Veterinary Rationale**

* **Partial Dependence Plots (PDPs)** illustrate the average effect of changing one environmental factor (e.g., temperature) while keeping others constant.
* **SHAP (SHapley Additive exPlanations)** values provide tank-level insight into why specific predictions occur.
  Together, these tools enable veterinarians to identify *cause–effect relationships* between environmental conditions and comfort outcomes.

#### **Machine Learning Rationale**

* **PDPs:** Offer a *global* interpretive view of feature–response dynamics.
* **SHAP:** Combines *local* (individual prediction) and *global* (overall trend) explainability, now standard in transparent AI for life sciences.

#### **Implementation Tasks**

* Generate PDPs for top features (e.g., Temperature, DO, Density).
* Apply SHAP to the trained Gradient Boosting model to derive:

  * Individual tank-level explanations.
  * Aggregated feature importance distributions.

---

## Phase 6 — Prediction and Deployment

### 1. Prediction Function (Development and Testing)

#### **Biological and Veterinary Rationale**

Developing a predictive function allows researchers and veterinarians to assess zebrafish tank conditions *in silico* before actual setup, supporting proactive welfare management.

#### **Machine Learning Rationale**

The prediction function enables **unit testing** of model reliability and **validation** of biological plausibility.
Integration with SHAP interpretability provides transparent explanations for each predicted comfort score.

#### **Implementation Tasks**

* Define a `predict_comfort()` function for internal validation.
* Output should include:

  * Predicted **comfort score** (0–100).
  * Assigned **comfort category** (Low/Medium/High).
  * Top contributing environmental factors (from SHAP analysis).
* Verify predictions for biological coherence and numerical stability.

---

### 2. Saving Artifacts

#### **Biological and Veterinary Rationale**

Archiving data and model artifacts ensures **traceability**, **reproducibility**, and **long-term validation** for future comparative studies.

#### **Machine Learning Rationale**

Model persistence eliminates retraining overhead and guarantees deployment reproducibility across computational environments.

#### **Implementation Tasks**

* Save both **raw** and **encoded** datasets.
* Persist the **trained Gradient Boosting model** using `joblib`.
* Save the **preprocessing pipeline** (e.g., encoders, scalers) for consistent input handling during deployment.

---

### 3. Deployment Interface

#### **Biological and Veterinary Rationale**

A streamlined deployment interface provides an accessible tool for animal care staff and researchers to input tank conditions and instantly obtain ECI predictions, along with explanations of driving factors.

#### **Machine Learning Rationale**

The deployment function consolidates preprocessing, prediction, and interpretability steps into a reproducible, production-ready framework.

#### **Implementation Tasks**

* Define `predict_comfort_deploy()` as the **deployment interface**.
* Outputs should include:

  * Final **comfort score** (0–100).
  * Corresponding **comfort category**.
  * Top three **SHAP-based explanatory features**.
* Ensure the function operates independently of the full notebook, suitable for integration into dashboards or laboratory management systems.

---

## Phase 7 — Conclusion, Ethics, and Future Work

### Summary of Findings

The Linear and Ridge regression models demonstrated limited capacity to capture the complex relationships underlying zebrafish welfare, explaining approximately half of the observed variation. In contrast, tree-based ensemble approaches—particularly **Gradient Boosting**—achieved superior predictive accuracy (≈95%), effectively modeling nonlinear environmental interactions and improving generalization.
Feature importance analyses identified **Nitrogen Load**, **Temperature**, **pH**, and **Density** as the dominant determinants of comfort, corroborating established veterinary and aquatic biology evidence. These findings highlight that water chemistry and stocking density remain critical for maintaining zebrafish well-being in research environments.
Interpretability methods, including **Partial Dependence Plots (PDPs)** and **SHAP** analyses, provided transparent insights into model behavior, enhancing scientific credibility and supporting biologically grounded interpretation.

### Veterinary Significance

The **Post-Lab Environmental Comfort Index (LECI)** provides a data-driven, reproducible, and interpretable framework to quantify and monitor zebrafish welfare.
By integrating artificial intelligence with veterinary science, LECI enables:

* Early detection of suboptimal tank conditions before physiological or behavioral stress occurs.
* Refinement of husbandry protocols in alignment with evidence-based welfare principles.
* Enhanced decision-making for veterinarians and facility managers in aquatic research programs.

### Ethical and Regulatory Framework

LECI aligns with the **3Rs principles**—**Replacement, Reduction, and Refinement**—by emphasizing welfare optimization through predictive analytics rather than additional animal use.
The framework is conceptually designed to operate within the ethical oversight systems of institutional programs, including:

* **IACUC** (Institutional Animal Care and Use Committee) guidelines for protocol refinement and humane endpoint assessment.
* **AAALAC International** accreditation standards for continuous quality improvement and welfare monitoring.
* **USDA Animal Welfare Regulations** and **NIH Office of Laboratory Animal Welfare (OLAW)** expectations regarding environmental enrichment and comfort assessment.

By supporting data-informed husbandry refinement, LECI contributes to reducing animal stress, enhancing reproducibility, and promoting the ethical advancement of biomedical research.

### Future Directions

* **Temporal Dynamics:** Incorporate time-series data reflecting environmental fluctuations and adaptive responses to better simulate real-world tank conditions.
* **Species Generalization:** Extend the framework to additional aquatic research species (e.g., medaka, guppy, goldfish) for comparative comfort modeling.
* **Automated Integration:** Embed LECI within intelligent vivarium systems for continuous, AI-assisted welfare monitoring and environmental optimization.
* **Cross-Facility Validation:** Conduct benchmarking across multiple research facilities to validate robustness under diverse environmental and operational conditions.
* **Ethical Impact Assessment:** Collaborate with institutional ethics committees to evaluate how predictive comfort modeling contributes to measurable welfare outcomes.

In conclusion, the **Post-Lab Environmental Comfort Index (LECI)** represents a foundational step toward **AI-driven ethical welfare analytics** in aquatic biomedical research—merging computational precision with regulatory compliance and veterinary ethics to advance the humane care and scientific integrity of laboratory animals.

---

## Continuity & Advancements  

This project builds on the [**Lab Animal Growth Prediction**](https://github.com/Ibrahim-El-Khouli/lab-animal-growth-prediction) framework:  

- Evolves from growth modeling → **environmental comfort assessment**.  
- Introduces **realistic datasets** and an expanded **modeling toolkit** (Linear, Ridge, Random Forest, Gradient Boosted Trees).  
- Gradient Boosted Trees proved **most effective and precise**, supporting reproducibility and deployment.  

---

## **License**

**Lab Environmental Comfort Index (LECI)** is released under the **MIT License** — free for academic, research, and non-commercial use.
