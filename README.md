# Topic Modeling

This repository demonstrates basic topic modeling using gensim's LDA.

## Running the example

The `lda_topics.py` script trains an LDA model on a `DataFrame` containing a `comment` column. The script outputs the top keywords for each topic and two example comments.

```bash
python lda_topics.py
```

You can also import the `extract_topics` function and apply it to your own DataFrame:

```python
from lda_topics import extract_topics
result = extract_topics(df_simulated)
print(result)
```

