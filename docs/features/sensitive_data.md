# Handling sensitive data

In certain data-sensitive contexts (for instance, private health
records), sharing a report that includes a sample would violate privacy
constraints. The following configuration shorthand groups together
various options so that only aggregate information is provided in the
report and no individual records are shown:

``` python linenums="1"
report = df.profile_report(sensitive=True)
```

Additionally, `data-profiling` does not send data to external
services, making it suitable for private data.

## Sample and duplicates

Explicitly showing a dataset\'as sample and duplicate rows can be
disabled, to guarantee the report does not directly leak any data:

``` python linenums="1"
report = df.profile_report(duplicates=None, samples=None)
```

Alternatively, it is possible to still show a sample but The following
snippet demonstrates how to generate the report but using mock/synthetic
data in the dataset sample sections. Note that the `name` and `caption`
keys are optional.

``` python linenums="1" title="Generate profiling with sensitive data: mocked sample"
# Replace with the sample you'd like to present in the report (can be from a mock or synthetic data generator)
sample_custom_data = pd.DataFrame()
sample_description = "Disclaimer: the following sample consists of synthetic data following the format of the underlying dataset."

report = df.profile_report(
    sample={
        "name": "Mock data sample",
        "data": sample_custom_data,
        "caption": sample_description,
    }
)
```

!!! warning

    Be aware when using `pandas.read_csv` with sensitive data such as phone
    numbers. pandas' type guessing will by default coerce phone numbers
    such as `0612345678` to numeric. This leads to information leakage
    through aggregates (min, max, quantiles). To prevent this from
    happening, keep the string representation.

``` python linenums="1"
pd.read_csv("filename.csv", dtype={"phone": str})
```

Note that the type detection is hard. That is why
[visions](https://github.com/dylan-profiler/visions), a type system to
help developers solve these cases, was developed.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />