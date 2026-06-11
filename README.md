# GWR Sentiment Analysis & Topic Modelling
A comprehensive NLP project analysing 2,115 customer reviews of Great Western Railway (GWR) from Trustpilot (2019–2024) using sentiment analysis, topic modelling, and machine learning classification — with actionable service improvement recommendations using Python.

## Problem Statement
The UK government has committed to net-zero emissions by 2050 and a 68% reduction by 2030. Achieving these targets requires a significant shift from private vehicles to public transport. However, GWR's customer satisfaction scores reflect declining rail productivity and persistent service pain points. This project analyses GWR Trustpilot reviews to uncover what customers are actually saying — classifying sentiment, identifying recurring complaint themes, and providing data-driven recommendations to improve service quality, public perception, and long-term ridership.

## Dataset
- **Source:** Trustpilot — GWR reviews scraped using `requests` and `BeautifulSoup`
- **Scope:** 2,115 reviews collected across 106 pages covering 2019–2024
- **Fields collected:** Username, star rating, review title, review content, date of experience
- **Format:** Exported as CSV (`gwr_reviews.csv`) for transparent, replicable downstream processing

## Key Analyses

### 1. Data Collection — Web Scraping
- Custom scraper built with `requests` and `BeautifulSoup` targeting Trustpilot's static HTML structure
- Browser-mimicking headers (User-Agent) and randomised sleep intervals (3–7 seconds) used to avoid IP blocking
- Iterative page-by-page extraction across 106 pages, capturing username, rating, title, content, and date

### 2. Text Preprocessing
- Lowercasing, removal of special characters, digits, URLs, and punctuation via regex
- Stopword removal using NLTK's built-in list plus a custom domain-specific list (`gwr`, `get`, `would`, `still`, etc.)
- Tokenisation, lemmatisation (WordNet), and bigram/trigram construction to preserve multi-word phrases (`train station`, `customer service`)
- TF-IDF vectorisation to weight rare but semantically important terms for classification and topic modelling

### 3. VADER Sentiment Analysis
- VADER (Valence Aware Dictionary and Sentiment Reasoner) applied to all 2,115 reviews
- Sentiment distribution: **1,008 negative**, **997 positive**, **110 neutral** — strongly polarised customer experiences
- Average sentiment scores by rating confirm VADER reliability: 1-star = −0.28, 5-star = +0.71
- **Positive word cloud:** helpful, staff, refund, ticket — satisfaction with service resolution
- **Negative word cloud:** delay, cancelled, refund, poor — frustration with punctuality and disruptions

### 4. Naïve Bayes Classifier
- MultinomialNB trained on TF-IDF features (unigrams, bigrams, trigrams), 70/30 train-test split
- Class weights applied to handle imbalance (1,538 negative vs. 577 positive reviews)
- **Overall accuracy: 90%**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.88 | 1.00 | 0.94 |
| Positive | 1.00 | 0.67 | 0.80 |

- Perfect recall on negative reviews (448/448 correctly identified) — critical for complaint monitoring
- Lower recall on positive reviews (125/187) — some satisfied customers misclassified as negative

### 5. Logistic Regression with Gradient Descent
- Custom logistic regression implemented from scratch using gradient descent (no sklearn fit)
- Multiple learning rates tested: α = 0.1 identified as optimal — rapid convergence without oscillation
- α = 1.0 caused divergence and parameter instability; α = 0.001 showed slow convergence
- **Overall accuracy: 91%**, correctly classifying 444/448 negative and 134/187 positive reviews

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.89 | 0.99 | 0.94 |
| Positive | 0.97 | 0.72 | 0.82 |

### 6. Model Comparison — Naïve Bayes vs Logistic Regression

| Model | Neg. Precision | Neg. Recall | Pos. Precision | Pos. Recall | Accuracy |
|-------|---------------|-------------|----------------|-------------|----------|
| Naïve Bayes | 0.88 | 1.00 | 1.00 | 0.67 | 90% |
| Logistic Regression | 0.89 | 0.99 | 0.97 | 0.72 | 91% |

Logistic Regression outperforms Naïve Bayes on positive recall (0.72 vs 0.67) and precision (0.97 vs 1.00) while maintaining near-identical negative detection — making it the preferred model for balanced, real-world deployment.

### 7. Topic Modelling — LDA
- LDA applied with TF-IDF-weighted corpus; optimal topic count selected via coherence scoring (tested 3–7 topics)
- **Best coherence score: C_V = 0.4224 at 3 topics**

| Topic | Top Keywords | Interpretation |
|-------|-------------|----------------|
| Topic 1 | September, utterly, shocking, service, bother, reservation, seat | Strong emotional dissatisfaction — severe disruptions or incidents |
| Topic 2 | website, refund, ticket, app, worst, page, assistant, week | Digital service frustration — refund delays, app failures, poor online support |
| Topic 3 | train, time, ticket, journey, seat, service, minute, delay, hour | Operational and punctuality concerns — delays, seat allocation, journey logistics |

### 8. Topic Modelling — BERTopic
- BERTopic applied with CountVectorizer (1–3 ngrams) and domain-specific stopword removal
- **Coherence score: 0.47** — outperforming LDA's 0.42 with greater topic separation and contextual richness

| Topic | Top Keywords | Interpretation |
|-------|-------------|----------------|
| Topic 1 | seat, travel, time, staff, service, journey | Core operational feedback — recurring concerns about journeys and staff |
| Topic 2 | bike, reservation, space, cycle | Niche but well-received service — cycling accommodation feedback |
| Topic 3 | customer, service, Twitter, staff, thank | Digital customer support — positive sentiment around social media responsiveness |

BERTopic's contextual embeddings captured nuances LDA's bag-of-words approach missed — including named staff members and specific digital channels — making it the stronger model for actionable business intelligence.

## Key Findings
- GWR customer sentiment is sharply polarised — nearly equal negative and positive reviews with very few neutral, suggesting strong reactions driven by specific incidents rather than moderate experiences
- **Delays, cancellations, and refund difficulties** are the three dominant complaint categories across both LDA and BERTopic
- Staff quality is a genuine strength — positive reviews consistently highlight individual staff members, presenting a retention and recognition opportunity
- **Digital services are a major weak point** — website usability, app reliability, and online refund processing generate disproportionate frustration
- Logistic Regression with gradient descent (α = 0.1) is the recommended production model — 91% accuracy with superior positive class recall and interpretable gradient descent convergence
- BERTopic (coherence 0.47) outperforms LDA (0.42) for business use — its transformer-based embeddings surface more specific, actionable topics

## Tech Stack
- **Language:** Python
- **Libraries:** `requests`, `BeautifulSoup`, `pandas`, `numpy`, `nltk`, `scikit-learn`, `gensim`, `bertopic`, `matplotlib`, `seaborn`, `wordcloud`, `textblob`
- **Environment:** Google Colab / Jupyter Notebook
- **Techniques:** Web Scraping, VADER Sentiment Analysis, Naïve Bayes Classification, Logistic Regression (Gradient Descent), LDA Topic Modelling, BERTopic, TF-IDF Vectorisation, Coherence Scoring

## How to Run
1. Clone the repo
```bash
git clone https://github.com/GokulKumar-7/sentiment-analysis.git
cd sentiment-analysis
```
2. Install dependencies
```bash
pip install requests beautifulsoup4 pandas numpy nltk scikit-learn gensim bertopic matplotlib seaborn wordcloud textblob
```
3. Download NLTK resources
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
```
4. The scraped dataset is available in `data/gwr_reviews.csv` — no need to re-scrape
5. Open `notebooks/Sentiment_and_topic_modelling_GWR.ipynb` and run all cells

## Conclusion
This project demonstrates how NLP can transform unstructured customer reviews into structured, actionable intelligence for a public transport operator. GWR's core service problems such as delays, refund failures, and digital friction; are clearly surfaced across all methods, validating both the models and the business case for continuous review monitoring. The combination of VADER for rapid polarity scoring, logistic regression for reliable classification, and BERTopic for nuanced theme extraction provides a complete, deployable sentiment intelligence pipeline. Integrating this into GWR's operations would enable faster complaint detection, more targeted service improvements, and measurable progress toward both customer satisfaction and the UK's broader sustainability goals. 
