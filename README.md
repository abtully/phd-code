# phd-code
Code written during PhD for C60/Au(111) tr-ARPES data plus general analysis.



## Outputs of notebook files cleared

The notebook `.ipynb` files were too big to upload to GitHub because of the
data stored in their outputs. So a `pre-commit` hook was used to clean out the
outputs before uploading to GitHub.

This is the script that was used:

```
#!/bin/sh
# This script is used to clear the outputs from Jupyter notebooks before committing

# Activate the Conda environment
call conda activate ARPES

# Find all .ipynb files in the commit, clear their outputs, and add them to the commit
git diff --cached --name-only --diff-filter=ACM | grep '\.ipynb$' | while read file; do
    # Clear the outputs from the current file
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$file"

    # Add the file back to the commit
    git add "$file"
done
```

To use it, place it in a file called `pre-commit` in `.git/hooks`.

Note: This assumes you have a `conda` environment called `ARPES` that has
`jupyter` and `nbconvert` installed.

Note: `call conda ...`  seems to work on Windows. For `unix` systems, `source activate` is probably the command to use instead.
