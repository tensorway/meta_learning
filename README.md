# meta_learning

Implementations of MAML, FOMAML and REPTILE as meta learning algorithms.

## REPTILE visualization

the learning rate was set to max possible
#### Legend
- blue   -> net that was trained with reptile
- black  -> baseline net
- red    -> true function
- orange -> samples

### REPTILE learns much faster 
![](gifs/reptile_sine.gif) ![](gifs/reptile_plot.gif)

### REPTILE learns better sines with 10 samples
![](gifs/reptile_sine_less.gif) ![](gifs/reptile_plot_less.gif)


REPTILE works for classification and regression, MAML seems to work on classification.
Classification is tested on Omniglot and regression on a Sine Dataset.
