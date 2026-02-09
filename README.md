Contextual Bandit-Based Personalized Recommendation System

This project implements and evaluates multiple contextual multi-armed bandit strategies for personalized news recommendation. A supervised learning model is first used to classify users into contextual groups followed by reinforcement learning policies that adaptively select optimal content categories to maximize user reward.

Problem Setup

Users are represented through behavioral and demographic features.
A multi-class classifier predicts the user context which is then used to maintain independent reward models for each user group.

Each context applies a contextual bandit strategy to select from multiple news categories (arms) receiving stochastic rewards based on user preferences.

⚙️ Methodology
1. User Context Classification

A Gradient Boosting Classifier was trained to identify user groups based on input features.

Classification Accuracy: 88.12%

2. Contextual Bandit Strategies

Three reinforcement learning approaches were implemented:

Epsilon-Greedy

Upper Confidence Bound (UCB)

SoftMax Action Selection

Each context maintained independent action-value estimates to learn optimal category selections over time.

📊 Experimental Results
🔹 Epsilon-Greedy

Small exploration rates (ε = 0.01) achieved the highest long-term rewards
Moderate exploration accelerated early learning for some users
High ε consistently reduced steady-state performance

🔹 UCB

Smaller exploration constants converged faster
Larger constants induced prolonged exploration
All configurations achieved similar long-term rewards

🔹 SoftMax

Smoothest learning curves
Stable convergence
Slightly lower final rewards compared to greedy strategies

Distinct reward trajectories across user contexts lead to effective personalization.

📈 Key Insights

Contextual separation significantly improves learning efficiency

Hyperparameters strongly influence convergence speed and stability

Exploitation-heavy strategies yield higher long-term rewards

Probabilistic exploration produces smoother learning behavior

Accurate user classification is crucial for reliable contextual policies

✅ Conclusion

The study demonstrates that contextual multi-armed bandits can effectively personalize recommendations by adapting to user-specific reward distributions. While greedy strategies maximize long-term reward, controlled exploration enables faster learning and robustness. Combining supervised context prediction with reinforcement learning provides a powerful framework for more efficient and adaptive decision making systems.