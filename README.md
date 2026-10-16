# Amazon Reviews Analysis: NLP, Time Series & Neural Networks

Summative lab for a data science course. Playing a junior analyst at an analytics firm, I analyse ~1,000 Amazon product reviews three ways: text mining, forecasting review activity over time, and predicting star ratings with neural networks.

## Dataset

`amazon_reviews_lab.csv`: 997 reviews of 40 products (2000–2014), mostly e-readers/tablets (Nook, Kindle) and TV mounts.

| Column | Description |
|---|---|
| `product_id` | Product identifier |
| `rating` | Star rating, 1–5 (58% are 5★) |
| `review_text`, `review_summary` | Full review and short headline |
| `review_time_raw`, `unix_review_time`, `timestamp` | Review date in three formats |
| `sentiment_label` | negative (1–2★), neutral (3★), positive (4–5★) |
| `review_length_words` | Word count of the review |

## What the notebook does

**Part 1: NLP**
- Cleaning pipeline: lowercase, strip URLs/HTML, expand contractions, tokenize, remove stopwords (negations like "not" are kept on purpose), lemmatize.
- Compares stemming with lemmatization, sentence length by sentiment, and POS tagging.
- Word, bigram and trigram frequencies, overall and by sentiment, plus words most associated with negative vs positive reviews.

**Part 2: Time series**
- Monthly review volume and average rating, with a 3-month rolling mean.
- Stationarity tests (ADF and KPSS) before and after log and difference transforms, seasonal decomposition, ACF/PACF.
- ARIMA model comparison by AIC/BIC and hold-out error against naive baselines, with residual diagnostics and a 12-month forecast.

**Part 3: Neural networks**
- TF-IDF features (3,000 unigrams and bigrams) feeding a multi-layer perceptron that predicts the 1–5 star rating.
- Same architecture in **PyTorch and Keras**; six Keras variants compared (dropout, L2, class weights, SGD vs RMSprop), plus a logistic regression benchmark.
- Evaluation with classification report, confusion matrix, within-one-star accuracy and 3-class sentiment accuracy.

## Key findings

- **Language:** Negative reviews are mostly about returns, billing/customer service, delivery, battery life and a limited app store. Positive ones describe working features. Word counts reflect product type as much as sentiment.
- **Trends:** Reviews are sparse before 2010, then grow to a peak in 2013. Average rating rises from about 3.9 to 4.3, and review volume peaks in December/January.
- **Forecasting:** Monthly counts need first differencing (d = 1). ARIMA(1,1,1) has the best AIC and ARIMA(0,1,1) the best hold-out error, but both only slightly beat a "repeat the last value" forecast, so treat forecasts as ranges.
- **Neural networks:** Always predicting 5★ already gives 58% accuracy, so models are compared on macro-F1. The best network reaches macro-F1 ≈ 0.34 (54% accuracy), 81% of predictions are within one star, and 3-class sentiment accuracy is 79%. 2★ reviews are the hardest class, and logistic regression (macro-F1 0.38) matches the networks. PyTorch and Keras give the same macro-F1.

## Run it

```bash
git clone https://github.com/<your-username>/amazon-reviews-nlp-timeseries-neuralnets.git
cd amazon-reviews-nlp-timeseries-neuralnets
pip install -r requirements.txt
jupyter notebook Summative_Lab_Customer_Reviews_Analysis.ipynb
```

Keep the notebook and `amazon_reviews_lab.csv` in the same folder. NLTK data downloads automatically on the first run.

**Cell order matters in Part 3.** PyTorch is imported and trained *before* TensorFlow is imported. In some environments, importing TensorFlow first makes PyTorch training crash with a segmentation fault. Run the cells top to bottom and don't move the TensorFlow import earlier.

## Limitations

- Only ~1,000 reviews, and 14 of them predate 2010, so time series work uses 2010 onward (55 months).
- The test set has 200 reviews (10 two-star), so small differences between models are within noise.
- Seasonality is estimated from about four years of data, so treat it as a hypothesis.

## Tools

Python, pandas, NLTK, scikit-learn, statsmodels, seaborn/matplotlib, TensorFlow/Keras, PyTorch.
