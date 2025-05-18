# Using as a local dependency

### Install hatch in the project venv

Create a `requirements.txt` with:

```
hatch==1.13.*
```

then run:

```
pip install -r requirements.txt
```

### Add the dependency

Assuming this project is a sibling to the project you want to use it in, add the following to your `pyproject.toml`:

```
dependencies    = [
    "llm-token-finder @ {root:uri}/../TokenFinder"
]
```

Then (on windows) run `pip install (hatch dep show requirements)`