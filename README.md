# Running Python Code in the Command Line

To run our implementation of the Maximum likelihood estimate, run:
```
python main.py
```

See command line options below:
```
"--feature" "-f" : The N-gram model to use
    Choices: "unigram", "bigram", "trigram"

"--smoothing" "-s" : The smoothing type to use
    Choices: "none", "add1", "linear"
```

By default, runs unigram model with no smoothing