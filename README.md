# Vector Space Document Retrieval

A document retrieval system built on the Vector Space Model, implemented in pure
Python with no numpy, pandas, or scikit-learn. Given a user query, it ranks a
document collection by relevance and returns the top 10 matches using cosine
similarity over term-weighted vectors.

It was built and evaluated on the CACM benchmark, a standard IR test collection
of computing-literature abstracts with gold-standard relevance judgements.

## Features

- Three term-weighting schemes: binary, term frequency (TF), and TF-IDF,
  selectable at runtime.
- Two optional preprocessing steps: stoplisting (removing common
  non-informative words) and stemming (reducing words to their root form),
  giving four preprocessing configurations to compare.
- Cosine-similarity ranking over an inverted index, returning the 10 most
  relevant documents per query.

## How it works

Precomputation at initialisation. Document frequency (DF), inverse document
frequency (IDF), and each document's vector magnitude are computed once when the
system loads rather than per query. IDF is calculated as log(N / DF), and
document magnitudes are derived in a single pass over the inverted index. When
TF-IDF is used, IDF is computed before the magnitudes, since the magnitudes
depend on the weighted values.

Query processing. Each query is converted to a vector, weighting its terms
according to the selected scheme. Query terms that don't appear in the
collection are given a zero IDF weight, so only shared terms contribute to the
score.

Candidate filtering. Only documents containing at least one query term can score
above zero, so the inverted index is used to gather this candidate set and the
rest are skipped. Retrieval cost then scales with the number of documents that
actually share terms with the query rather than with the size of the whole
collection.

Scoring. For each candidate, the dot product of the query and document vectors
is accumulated and only positive scores are kept. Each score is divided by the
document's precomputed vector magnitude to give the cosine value. The
query-vector magnitude is left out because it is constant across all documents
for a given query and so has no effect on ranking. Scores are sorted in
descending order and the top 10 document IDs are returned.

## Results

Evaluated against the gold-standard relevance file using Precision, Recall, and
F-measure across all weighting and preprocessing configurations.

| Weighting | Preprocessing | Precision | Recall | F-measure |
|-----------|---------------|-----------|--------|-----------|
| TF-IDF    | stem + stop   | 0.27      | 0.22   | 0.24      |
| TF-IDF    | stem only     | 0.26      | 0.21   | 0.23      |
| TF        | stem + stop   | 0.19      | 0.15   | 0.17      |
| Binary    | none          | 0.07      | 0.06   | 0.06      |

Findings:

- TF-IDF beat TF and binary in every configuration, which fits its ability to
  downweight frequent terms and reward rarer, more discriminative ones.
- Applying stemming and stoplisting together gave the best results for every
  weighting scheme, and no preprocessing was consistently the worst.
- With only one preprocessing step applied, TF-IDF did better with stemming
  while binary and TF did better with stoplisting, which suggests TF-IDF gains
  most from root-based term matching.

Full results and discussion are in report.pdf.

## Tech

Python, standard library only. math is used for square roots; the vector space
model, IDF weighting, and cosine ranking are all implemented by hand.
