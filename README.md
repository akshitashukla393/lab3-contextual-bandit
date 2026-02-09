# Lab 3: Contextual Bandit-Based News Article Recommendation System

---

## Project Overview

This project implements a **contextual multi-armed bandit system** for personalized news article recommendations. The system learns user preferences across three distinct user contexts (User1, User2, User3) and recommends news articles from four categories (Entertainment, Education, Tech, Crime) using reinforcement learning strategies.

The primary objective is to maximize cumulative reward by balancing exploration (discovering new preferences) and exploitation (recommending known-good articles) through different bandit algorithms.

---

## Methodology

### 1. **Data Description**

The project utilizes three datasets:

- **`news_articles.csv`**: Contains news articles with categories (Entertainment, Education, Tech, Crime)
- **`train_users.csv`**: Training set with user features and labels (User1, User2, User3)
- **`test_users.csv`**: Test set for evaluating the recommendation engine

### 2. **Data Preprocessing**

- Removed missing values from all datasets
- Separated features from target labels
- Encoded categorical user labels using LabelEncoder
- Applied StandardScaler for feature normalization

### 3. **User Classification Model**

**Algorithm:** Gradient Boosting Classifier

**Hyperparameters:**
- Number of estimators: 300
- Learning rate: 0.05
- Max depth: 3
- Random state: 42

**Train-Test Split:** 80-20 stratified split

**Purpose:** Classify users into contexts (User1, User2, User3) to provide context for the bandit algorithms.

### 4. **Contextual Bandit Algorithms**

Three distinct exploration-exploitation strategies were implemented:

#### **A. Epsilon-Greedy Strategy**

- **Concept:** With probability ε, select a random arm (explore); otherwise, select the arm with highest estimated reward (exploit)
- **Tested ε values:** 0.01, 0.05, 0.1
- **Update Rule:** Incremental Q-value updates using sample means
- **Formula:**
  - If random() < ε: arm = random(0, num_arms)
  - Else: arm = argmax(Q[context])
  - Q[context, arm] ← Q[context, arm] + (reward - Q[context, arm]) / N[context, arm]

#### **B. Upper Confidence Bound (UCB) Strategy**

- **Concept:** Select arm with highest upper confidence bound incorporating both estimated reward and uncertainty
- **Tested C values:** 0.1, 1.0, 2.0, 5.0, 10
- **Update Rule:** UCB = Q[context, arm] + C × √(ln(t) / N[context, arm])


#### **C. Softmax Strategy**

- **Concept:** Probabilistically select arms using softmax distribution over Q-values
- **Temperature (τ):** 1.0
- **Update Rule:** P(arm) = softmax(Q[context] / τ); select arm based on probability
- **Advantage:** Smooth exploration with gradient-based action selection

### 6. **Reward Sampling**

Rewards are sampled using a seeded reward sampler initialized with student roll number (119), ensuring reproducible and fair comparison across algorithms.

---

## Results

### Classification Performance

**Gradient Boosting Classifier Accuracy:** 88.12%

The high accuracy indicates robust user context prediction which is essential for effective contextual bandits.

### Reinforcement Learning Simulation Results

**Simulation Horizon:** T = 10,000 time steps

#### **Epsilon-Greedy Results**

All three epsilon values demonstrated convergence toward optimal recommendations:

- **ε = 0.01:** Highest long-term average reward 
- **ε = 0.05:** Moderate long-term performance 
- **ε = 0.1:** Lower steady-state performance due to excessive exploration 

**Key Observation:** Smaller epsilon values consistently outperformed larger values confirming that the exploitation-heavy strategy achieves better cumulative rewards.

#### **UCB Results**

Varying exploration constant C showed distinct early learning behavior:

- **C = 0.1:** Fastest convergence 
- **C = 1.0:** Balanced exploration-exploitation
- **C = 5.0:** Slower convergence but more thorough exploration reaching similar final rewards

**Key Observation:** All C values converged to similar long-term rewards indicating robust performance across different parameters

#### **Softmax Results**

Temperature τ = 1.0 produced smooth stable learning:

- **Average Reward Trajectory:** Gradual improvement with low variance
- **Convergence Rate:** Moderate 


**Key Observation:** Softmax provided the most stable convergence with smooth reward curves, though slightly lower than optimized Epsilon-Greedy.

---

## Key Insights & Discussion

### 1. **Exploration-Exploitation Tradeoff**

- **High exploration (high ε, high C):** Accelerates early discovery but introduces reward variance and reduces long-term performance
- **Low exploration (low ε, low C):** Sacrifices early learning but achieves superior cumulative rewards
- **Optimal balance:** Achieved with ε = 0.01 in Epsilon-Greedy and C = 1.0 in UCB


### 3. **Contextual Learning Effectiveness**

The system successfully separated user contexts, learning distinct optimal policies for each user group. Different reward trajectories per user context demonstrate that contextual information significantly improves personalization.

### 4. **Hyperparameter Sensitivity**

- **Epsilon-Greedy:** Highly sensitive; small changes in ε produce significant reward differences
- **UCB:** Moderate sensitivity; larger range of C values produce acceptable performance
- **Softmax:** Robust; temperature variation has limited impact on final rewards

---

## Strengths

✅ Contextual separation significantly improves learning efficiency
✅ Hyperparameters strongly influence convergence speed and stability
✅ Probabilistic exploration produces smoother learning behavior  
✅ High classification accuracy (88.12%) ensures reliable context prediction   


---

## Limitations

⚠️ **Classification Noise:** 11.88% misclassification rate introduces contextual errors that may affect reward optimality  
⚠️ **Limited Context Scale:** System tested with only 3 user contexts; scalability to larger populations uncertain  
⚠️ **Reward Distribution:** Reward distributions may favor certain algorithms; more diverse reward structures would strengthen conclusions  
⚠️ **Simulation Length:** T=10,000 may be insufficient for convergence with high exploration rates  
⚠️ **Stationarity Assumption:** Algorithms assume stationary reward distributions; real-world preferences may drift over time  

---

## Key Insights

Contextual separation significantly improves learning efficiency

Hyperparameters strongly influence convergence speed and stability

Exploitation-heavy strategies yield higher long-term rewards

Probabilistic exploration produces smoother learning behavior

Accurate user classification is crucial for reliable contextual policies

## Conclusion

The study demonstrates that contextual multi-armed bandits can effectively personalize recommendations by adapting to user-specific reward distributions. While greedy strategies maximize long-term reward, controlled exploration enables faster learning and robustness. Combining supervised context prediction with reinforcement learning provides a powerful framework for more efficient and adaptive decision making systems.