# Module 1 — Data Pipeline

Scrapes catalog data from books.toscrape.com, transforms raw fields, applies currency conversion, and stores results in a normalized SQLite database.

## Currency Conversion
- **Fixed Baseline Rate**: 1 GBP = 105.50 INR (project constant).

## Data Cleaning & Parsing Choices
- **Price**: Stripped currency symbol `£` and cast values to `float` (`price_gbp`). Calculated `price_inr` using the 105.50 fixed rate.
- **Rating**: Converted word ratings (`One`–`Five`) to integers (1–5).
- **Stock Status**: Parsed availability text to boolean `in_stock` (1/0).
- **Messy Rows**: Dropped rows with missing critical catalog fields to prevent biased metrics.

## Schema
- `categories`: `category_id` (PK), `category_name` (UNIQUE)
- `books`: `book_id` (PK), `title`, `price_gbp`, `price_inr`, `rating`, `in_stock`, `category_id` (FK)

## Query & Join Verification
Includes 5 executed SQL queries (`SELECT/WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN/IN`, `JOIN`) and validates that `pd.read_sql` matches `pd.merge` on in-memory DataFrames.
