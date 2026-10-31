## How to contribute to Data-Profiling

Data-profiling aims to ease exploratory data analysis for structured datasets, including time-series. 
Our focus is to provide users with useful and robust statistics for such datasets encountered in industry, academia and elsewhere.
Data-profiling is open-source and stimulates contributions from passionate community users.

#### Setup

```sh
python -m venv .venv
make install-dev
make test
``

#### Themes to contribute
In line with our aim, we identify the following themes:

- **Exploratory data analysis**: 
  The core of the package is a dataset summarization by its main characteristics, which is complemented with warnings on data issues and visualisations.

  _Suggestions for contribution_: 
  Extend the support of more data types (think of paths, location or GPS coordinates and ordinal data types),
  text data (e.g. encoding, vocabulary size, spelling errors, language detection), 
  time series analysis, 
  or even images (e.g. dimensions, EXIF).
  
  _Related_: [#7][i7], [#129][i129], [#190][i190], [#204][i204] or [create one](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/new/choose).

- **Stability, Performance and Restricted environment compatibility:** 
  Data exploration takes place in all kinds of conditions, on the latest machine learning platforms with enormous dataset to managed environments in large corporations.
  `fg-data-profiling` helps analysts, researchers and engineers alike in these cases.
  We do this by fixing bugs, improving performance on big datasets and adding environment compatibility.
  
  _Suggestions for contribution (Performance)_: 
  Perform concurrency analysis or profile execution times and leverage the gained insights for improved performance (e.g. multiprocessing, cython, numba) or test the performance of `fg-data-profiling` with [big data sets](https://www.stats.govt.nz/large-datasets/csv-files-for-download/) and corresponding commonly used data formats (such as parquet). 
  
  _Suggestions for contribution (Stability)_: 
  Either review the code and add tests or watch the [issues page](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues) and [Stackoverflow tag](https://stackoverflow.com/questions/tagged/ydata-profiling) to find current issues.
     
  _Related_: [#98][i98], [#122][i122] or [create one](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/new/choose).

- **Interaction, presentation and user experience**: 
  As `fg-data-profiling` eases exploratory data analysis, working with the package should reflect that.
  Interaction and user experience plays a central role in working with the package.
  Working on interactive and static features is possible through the modular nature of the package: the user can configure which features to use.

  _Suggestions for contribution (interactivity)_:
  Interactivity allows for more user friendly applications, including but not limited to on demand analysis (don't compute what you don't want to see) and interactive histograms and correlations. 
  This is ideal for smaller datasets, where we can compute this on-the-fly. 
  `ipywidgets` would be a great place to start (e.g. [widget based view](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html)).

  _Suggestions for contribution (presentation)_:
  Other forms of distribution than HTML (for example PDF or packaged as an GUI application via [PyQt](https://riverbankcomputing.com/software/pyqt/intro))
  Users should be able to share reports (improve size of labels in graph, add explanations to correlation matrices and allow for styling/branding).

  _Related_: [#161][i161], [#175][i175], [#191][i191] or [create one](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/new/choose).

- **Community**: 
  The success of this package demonstrates the power of sharing and working together.
  You are welcome as part of this community.
  
  _Suggestions for contribution_:
  Share with us if this package is of value to you, let us know [in our community](https://discord.com/invite/mw7xjJ7b7s).
  We are interested in how you use `fg-data-profiling` in your work.
  
  _Related_: [#87][i87] or [create one](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/new/choose).

- **Machine learning:** 
  `fg-data-profiling` is not a machine learning package, even though many of our users use EDA as a step prior to developing their models.
  Our focus lies in the exploratory data analysis.
  Any functionality that enables machine learning applications by more effective data profiling, is welcome.

  _Related_: [#124][i124], [#173][i173], [#198][i198] or [create one](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/new/choose).

#### **Did you find a bug?**

* **Ensure the bug was not already reported** by searching on Github under [Issues](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues).

* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/new/choose). 
If possible, use the relevant bug report templates to create the issue. 

#### **Did you write a patch that fixes a bug?**

* Open a new Github pull request with the patch.

* Ensure the PR description clearly describes the problem and solution. 
Include the relevant issue number if applicable.


#### Acknowledgements

We would like to thank everyone who has helped getting us to where we are now.

See the [Contributor Graph](https://github.com/Data-Centric-AI-Community/fg-data-profiling/graphs/contributors)

[i7]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/7
[i129]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/129
[i190]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/190
[i204]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/204
[i98]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/98
[i122]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/122
[i124]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/24
[i173]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/173
[i198]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/198
[i87]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/87
[i161]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/161
[i175]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/175
[i191]: https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/191

