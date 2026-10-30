# Installation

## Using pip

You can install using the ``pip`` package manager by running:

```console
    pip install -U fg-data-profiling
```
If you are in a notebook (locally, LambdaLabs, Google Colab or Kaggle), you can run:

```python linenums="1"
    import sys
    !{sys.executable} -m pip install -U "fg-data-profiling[notebook]"
    !pip install jupyter-contrib-nbextensions
```
Afterwards you can run the following command

```python linenums="1"
    !jupyter nbextension enable --py widgetsnbextension
```
You may have to restart the kernel or runtime for the package to work.

## Using conda

[fg-data-profiling through Conda](https://anaconda.org/conda-forge/fg-data-profiling)

A new conda environment containing the module can be created via: 

```console
    conda create -n data-profiling
    conda activate data-profiling
    conda install -c conda-forge fg-data-profiling
```

!!! tip 
        Don't forget to specify the ``conda-forge`` channel.       
        Omitting it **will not** lead to an error, as an outdated package lives on the ``main`` channel and will be installed. See :doc:`../support_contrib/common_issues` for details. 

## Widgets in Jupyter Notebook/Lab

For the Jupyter widgets extension (used for progress bars and the interactive widget-based report) to work, you might need to install and activate the corresponding extensions. 
This can be done via ``pip``: 

```console
  pip install "fg-data-profiling[notebook]"
  jupyter nbextension enable --py widgetsnbextension
```

Or via ``conda``:
```console
  conda install -c conda-forge ipywidgets
```

In most cases, this will also automatically configure Jupyter Notebook and Jupyter Lab (``>=3.0``). For older versions of both or in more complex
environment configurations, refer to [the official ipywidgets documentation](https://ipywidgets.readthedocs.io/en/stable/user_install.html).

## From source

Download the source code by cloning the repository or by clicking on [Download ZIP](https://github.com/Data-Centric-AI-Community/fg-data-profiling/archive/master.zip).
Install it by navigating to the uncompressed directory and running:

```console
   python setup.py install
```

This can also be done via the following one-liner: 

```console
  pip install https://github.com/Data-Centric-AI-Community/fg-data-profiling/archive/master.zip
```
    

## Extras
The package declares some "extras", sets of additional dependencies.

* ``[notebook]``: support for rendering the report in Jupyter notebook widgets.
* ``[unicode]``: support for more detailed Unicode analysis, at the expense of additional disk space.
* ``[spark]``: support for pyspark engine to run the profile on big datasets

Install these with e.g.
```console
  pip install -U "fg-data-profiling[notebook,unicode,spark]"
```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />