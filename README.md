dataset source:
https://www.kaggle.com/datasets/msambare/fer2013

compressed dataset "archive.zip" contains 2 folders: "train" and "test" 

add extracted data folders ("train" and "test") to directory in the project: /data/raw/fer2013 so after extraction it looks like;

```text
/data
    /raw
        /fer2013
                /train
                        /happy
                        /sad
                        /...
                /test 
                        /happy
                        /sad
                        /...
```

quickstart for testing:

1) install dependencies

```bash
python -m pip install -e .
```

2) demo presentation of progres until 14.09

```bash
python tests/test_run_prediction.py
```

original publication regarding datasource:
* (note ot self) doublecheck
* (nts^) also add it to cited sources

https://arxiv.org/pdf/1307.0414