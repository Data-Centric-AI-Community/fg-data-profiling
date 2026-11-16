# Concepts

!!! question "Text/corpus data - your input is needed!"

    `data-profiling` team is considering the support of a new set of features for corpus data
    and we want to hear from you! We're particularly interested in understanding why you think
    these features would be useful, and your input will help us prioritize and refine this development.

    👉 Upvote [add here link for the request form]

## Data Structures supported

The profiling offers comprehensive insights into various types of data, including tabular, time-series text and image data. 

- **Tabular data:** when dealing with tabular data, such as spreadsheets or databases, the profiling provides valuable statistics on data distribution, central tendencies, and categorical variable frequencies.
It identifies multivariate relations such as correlations and interactions in a visual manner. It also identifies missing data.  
- **Time-series data:** when dealing with data with temporal dimensions, the profiling extends its capabilities to capture trends, seasonality, cyclic patterns and missing data gaps. 
It can reveal information about data volatility, periodicity, and anomalies, facilitating a deeper understanding of time-dependent trends.
- **Text:**  when it comes to text data, such as strings or documents, the profiling offers insightful statistics on the distribution of word frequencies, common phrases, and unique words. 

## Data types
Types that go beyond the storage dtype (integer, float, object) decide which analysis each column gets.
A column of strings such as `"1.5"`, `"2.0"` is profiled as a number, and a column of `"yes"`/`"no"` as a boolean.
``data-profiling`` recognizes the following types:

- Boolean
- Numeric
- DateTime (dates and datetimes)
- Categorical
- Text
- TimeSeries (with `tsmode=True`)
- URL, Path, File and Image: detected only when enabled in the config (all off by
  default), e.g. `ProfileReport(df, vars={"url": {"active": True}})`. File requires
  Path, and Image requires File.
- Unsupported: columns that match no other type (lists, dicts, mixed values,
  all-missing columns). They get the generic statistics only: missing values,
  distinct and unique counts, and memory size.

Each column starts as Unsupported and moves to the most specific type it satisfies. Some columns satisfy
more than one: `["1", "2", ..., "10"]` repeated over many rows can be read both as numbers and as categories.
In that case a fixed precedence decides, so the same column always gets the same type:

1. Boolean (strings listed in `vars.bool.mappings`)
2. Numeric
3. DateTime
4. URL, Path, File, Image
5. Categorical (at most `vars.num.low_categorical_threshold` distinct numbers, or at most
   `vars.cat.cardinality_threshold` distinct strings that are less than `vars.cat.percentage_cat_threshold` of the rows)
6. TimeSeries
7. Text

Two string patterns that a date parser accepts are not read as dates: values without a digit (month or weekday
names such as `"Jan"`) and values that are plain numbers (years such as `"2024"`).

When inference gets a column wrong, set its type with `type_schema` (see [Dataset schema](../features/metadata.md#dataset-schema)).
To see why a column got its type, ask the typeset:

``` python
report = ProfileReport(df)
report.typeset.explain_type(df["zip"])
# [{'from': 'Unsupported', 'matched': ['Text'], 'chosen': 'Text'},
#  {'from': 'Text', 'matched': ['Numeric', 'Categorical'], 'chosen': 'Numeric'}]
```

A step with more than one entry in `matched` is a column where the precedence decided the type.
Custom types and summaries are supported: subclass `ProfilingTypeSet` or build a typeset from
`data_profiling.model.typeset_base`, where the order of the types is the precedence. See the
[default typeset](https://github.com/Data-Centric-AI-Community/fg-data-profiling/blob/develop/src/data_profiling/model/typeset.py)
for reference.

## Data quality alerts
<figure markdown>     
   ![Data quality warnings data-profiling](../_static/img/warnings_section.png){width="700"}
   <figcaption>Data quality warnings</figcaption>
</figure>

Alerts section in the *NASA Meteorites* dataset's report. Some alerts include numerical indicators. 

The **Alerts** section of the report includes a comprehensive and automatic list of potential data quality issues. Although useful, the decision on whether an alert is in fact a data quality issue always requires domain validation. Some warnings refer to a specific column, others refer to inter-column relationships and others are dataset-wide. The table below lists all possible data quality alerts and their meanings.

{{ read_csv('./data_quality_alerts.csv') }}

Information on the default values and the specific parameters/thresholds used in the computation of these alerts, 
as well as settings to disable specific ones, can be consulted in the [documentation](../advanced_settings/available_settings.md).

## Univariate profiling                                

<figure markdown>     
   ![Univariate profiling data-profiling](../_static/img/univariate_profiling.png){width="700"}
   <figcaption>Univariate profiling metrics and visualization</figcaption>
</figure>

This section provides a comprehensive overview of individual variables within a given dataset, this feature is particularly useful for exploratory data analysis (EDA)
as it automatically calculated detailed statistics, visualizations, and insights for each variable in the dataset. It offers information such as data type, missing values, unique values, basic descriptive statistics
, histogram plots, and distribution plots. This allows data analysts and scientists to quickly understand the characteristics of each variable, identify potential data quality issues, and gain initial insights into the data's distribution and variability. 

For more details about the different metrics and visualizations check the Univariate section details page. 

## Multivariate profiling

<figure markdown>     
   ![Multivariate profiling data-profiling](../_static/img/multivariate_profiling.png){width="700"}
   <figcaption>Multivariate profiling metrics and visualization</figcaption>
</figure>

This section provides essentials insights into the relationships between variables through correlations matrices and interactions. 
The correlation view computes and presents correlation coefficients between pairs of numerical variables, helping to identify potential linear relationships.
This assists data analysts and scientists in understanding how variables change together and highlights possible multicollinearity issues.

On the other hand, the interactions section goes beyond correlation by exploring potential nonlinear relationships and interactions between variables, providing a more comprehensive understanding of how variables interact with one another. 
This can be crucial in identifying hidden patterns that might not be captured through traditional correlation analysis.

Check the section about interactions configuration and correlation matrix metrics for more details.  

## Missing data

This section offers valuable insights into the presence and distribution of missing data within a dataset. It can be particularly helpful for data preprocessing and quality assesment as
provides a comprehensive summary of missing values across variables, indicating the percentage of missing data for each variable. Additionally, it displays a visual representation of missing data patterns through bar plots and heatmaps, 
allowing users to quickly identify which variables have the most significant amount of missing information.

Check how you can configure your missing data visualization. 
                
## Preview data 
For a quick overview of the data, data-profiling provides the following sections that can be easily configure by the user:
- First n records of a given dataset
- Last n records of a given dataset
- A table containing observed duplicates (exact matches)

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />