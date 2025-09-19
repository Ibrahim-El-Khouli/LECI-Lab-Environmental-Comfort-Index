# Lab Environmental Comfort Index (LECI)  

## Project Summary  
The **Lab Environmental Comfort Index (LECI)** provides a **quantitative metric for environmental comfort** in laboratory tanks, initially validated for **zebrafish**.  
It predicts a **0–100 Environmental Comfort Index (ECI)**, assigns a **comfort category** (Low, Medium, High), and highlights the **top contributing environmental factors**.  

The framework is designed to **scale to other species** (e.g., rodents, rabbits) and to support **reproducible welfare assessments**.  
By aligning with the **3Rs principles (Replacement, Reduction, Refinement)**, LECI helps refine housing conditions, improve welfare monitoring, and enhance reproducibility in animal research.  

---

## Objectives  
- Provide an **objective, quantitative assessment** of zebrafish housing conditions.  
- Support **veterinarians, lab technicians, and researchers** in maintaining optimal environmental parameters.  
- Align with the **3Rs principles**, especially **Refinement**, by identifying and mitigating environmental stressors.  
- Build a **scalable, species-agnostic framework** adaptable to multiple experimental setups.  

---

## Key Outputs  
1. **ECI Score (0–100)**  
2. **Comfort Category**: Low, Medium, High  
3. **Top 3 contributing environmental features**  

---

## Features  

**Numeric Features:**  
- Temperature (°C)  
- Dissolved Oxygen (mg/L)  
- pH  
- Water Hardness (ppm)  
- Humidity (%)  
- Stocking Density (fish/L)  
- Nitrogen Load (mg/L)  

**Categorical Features:**  
- Substrate Type: Sand, Gravel, Glass  
- Noise Level: Low, Medium, High  
- Light-Dark Cycle: 10/14, 12/12, 14/10  
- Water Flow: Low, Medium, High  

**Derived Feature Example:**  
- **Thermal-Oxygen Stress Index** → captures the combined effect of temperature deviations and low dissolved oxygen levels.  

**Target / Output:**  
- **ECI (0–100)**  
- **Comfort Category (Low, Medium, High)**  

---

## Comfort Categories  

| ECI Range | Category       |
|-----------|----------------|
| 0–39      | Low Comfort    |
| 40–69     | Medium Comfort |
| 70–100    | High Comfort   |  

---

## Methodology  

1. **Data Generation** – synthetic datasets reflect biologically-informed ranges and natural variability.  
2. **Preprocessing** – categorical encoding + derived feature engineering.  
3. **Modeling** – tested multiple supervised regressors:  
   - Linear Regression  
   - Ridge Regression  
   - Random Forest Regressor  
   - Gradient Boosted Trees (**highest accuracy & robustness**)  
4. **Evaluation** – R², MAE, RMSE with cross-validation.  
5. **Interpretability** – feature importance, partial dependence plots (PDPs), SHAP values.  
6. **Deployment** – ready-to-use prediction function outputs:  
   - ECI  
   - Comfort category  
   - Top contributing features  
7. **Artifacts Saved:**  
   - `data/zebrafish_environment.csv` – raw dataset  
   - `data/zebrafish_environment_encoded.csv` – encoded dataset  
   - `models/LECI_gradient_boosting_model.pkl` – trained model  
   - `models/LECI_preprocessing_pipeline.pkl` – preprocessing pipeline  

---

## Welfare & 3Rs Relevance  

- **Refinement** – actionable insights to improve animal housing conditions, reducing stress.  
- **Reduction** – stabilized environments → less variability → fewer animals needed.  
- **Replacement (indirect)** – synthetic datasets reduce pilot testing needs.  

---

## Continuity & Advancements  

This project builds on the [**Lab Animal Growth Prediction**](https://github.com/Ibrahim-El-Khouli/lab-animal-growth-prediction) framework:  

- Evolves from growth modeling → **environmental comfort assessment**.  
- Introduces **realistic datasets** and an expanded **modeling toolkit** (Linear, Ridge, Random Forest, Gradient Boosted Trees).  
- Gradient Boosted Trees proved **most effective and precise**, supporting reproducibility and deployment.  

---

## Notes  

For the complete workflow, predictions, and visualizations, see the Jupyter notebook in the `notebooks/` directory. 

