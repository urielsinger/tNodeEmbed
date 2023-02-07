---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

A link to the "baby" CORA dataset. Not used and should be the same as the included `.tsv` files from https://relational.fit.cvut.cz/dataset/CORA.

```{code-cell} ipython3
#!wget http://www.cs.umd.edu/~sen/lbc-proj/data/cora.tgz
```

The is the full CORA dataset version 1.0.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
!wget -N http://people.cs.umass.edu/~mccallum/data/cora-classify.tar.gz
!tar zxvf cora-classify.tar.gz
```

```{code-cell} ipython3
!wget -N http://people.cs.umass.edu/~mccallum/data/cora-classify.tar.gz
```

```{code-cell} ipython3
!tar --skip-old-files -zxvf cora-classify.tar.gz
```
