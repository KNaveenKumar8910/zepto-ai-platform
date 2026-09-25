# Zepto Data & AI Platform Capstone Project

An end-to-end AI/ML repository consisting of three core modules:
- `/data_pipeline`: Scrapes catalog data from books.toscrape.com, applies fixed currency conversion (1 GBP = 105.50 INR), and loads into a normalized SQLite database.
- `/analytics`: Profiles the Titanic dataset, handles missing data using threshold-based rules, trains classifiers within a leak-free ColumnTransformer pipeline, evaluates SMOTE imbalance handling, performs GridSearchCV on Random Forest with OOB scoring, and trains a linear regression model for fare prediction.
- `/support_assistant`: Embeds Zepto policy documents in ChromaDB with all-MiniLM-L6-v2, routes queries using a LangGraph StateGraph, enforces Pydantic output schemas, runs under deterministic MOCK_LLM=1 mode, and serves via FastAPI on port 7860.
