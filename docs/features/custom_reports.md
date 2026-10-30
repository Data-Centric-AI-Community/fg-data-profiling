# Customizing the report

In some situations, a user might want to customize the appearance
of the report to match personal preferences or a corporate brand.
``data-profiling`` offers two major customization dimensions: 
the **styling of the HTML report** and the **styling of the visualizations
and plots** contained within. 

## Customizing the report's theme

Several aspects of the report can be customized. The table below shows the available settings:

{{ read_csv('./tables/config_html.csv') }}

See the [available changing settings](../advanced_settings/changing_settings.md)
to see how to change and apply these settings.

## Customizing the visualizations

### Plot rendering options

A way how to pass arguments to the underlying ``matplotlib`` visualization
engine is to use the ``plot`` argument when computing the profile.
It is possible to change the default format of images to png (default is SVG)
using the key-pair ``image_format: "png"`` and also the resolution of the images using ``dpi: 800``.
An example would be:

```python linenums="1" title="Customize plots rendering"
profile = ProfileReport(
    planets,
    title="YData Profiling Report",
    explorative=True,
    plot={"dpi": 200, "image_format": "png"},
)
```

### Pie charts

Frequency plots display the frequency of categories in categorical
(or boolean) features. By default, they use bars and are shown for features
with no more than 10 distinct values. To use pie charts, set
``plot.cat_freq.type`` to ``"pie"``. The display threshold can be
configured with the ``plot.cat_freq.max_unique`` setting; it does not
change the inferred variable type.

``` python linenums="1" title="Control pie charts frequency"
profile = ProfileReport(pd.DataFrame(["a", "b", "c"]))
profile.config.plot.cat_freq.type = "pie"
# Hide the frequency plot when there are more than 2 distinct values
profile.config.plot.cat_freq.max_unique = 2
```

If the feature exceeds this threshold, the frequency plot will not be
displayed. All frequency plots can therefore be removed by setting: ``plot.cat_freq.max_unique = 0``.

The pie chart's colors can be configured to any [recognised matplotlib colour](https://matplotlib.org/stable/tutorials/colors/colors.html) using the `plot.cat_freq.colors` setting.

``` python linenums="1" title="Control pie charts colors"
profile = ProfileReport(pd.DataFrame([1, 2, 3]))
profile.config.plot.cat_freq.type = "pie"
profile.config.plot.cat_freq.colors = ["gold", "b", "#FF796C"]
```

### Colour palettes

The palettes used in visualizations like correlation matrices and missing
values overview can also be customized via the ``plot`` argument. To customize
the palette used by the correlation matrix, use the ``correlation`` key:

``` python linenums="1" title="Changing visualizations color palettes"
  from data_profiling import ProfileReport

  profile = ProfileReport(
      df,
      title="YData Profiling Report",
      explorative=True,
      plot={"correlation": {"cmap": "RdBu_r", "bad": "#000000"}},
  )
```

Similarly, the palette for *Missing values* can be changed using ``missing`` argument:

``` python linenums="1" python

  from data_profiling import ProfileReport

  profile = ProfileReport(
      df,
      title="YData Profiling Report",
      explorative=True,
      plot={"missing": {"cmap": "RdBu_r"}},
  )
```

``data-profiling`` accepts all ``cmap`` values (colormaps) accepted by ``matplotlib``.
The list of available colour maps can [be accessed here](https://matplotlib.org/stable/tutorials/colors/colormaps.html>).
Alternatively, it is possible to create [custom palettes](https://matplotlib.org/stable/gallery/color/custom_cmap.html>).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />