# Plot extraction

This tool extracts plot data from papers. It is inspired by Plot2Spectra, but modernized with the ability to automatically extract figures from papers. Axis and tick detection is replaced with Claude Vision, via Claude Code backend. 

## Plot detection algorithm

The plot detection algorithm is inspired by optical flow, which is the idea that a moving object can be tracked if we follow pixels of similar brightness of smooth motion through time. This principle is applied to plot detection; we can follow pixels of similar colour and intensity and of smooth motion as you move along the $x$-axis. 

Jiang et al. propose an heuristic to decide whether a pixel is part of a curve, as you sweep across the $x$ axis. We can consider four sources of "loss" and sum them up. Among candidate neighbouring pixels, the pixel with the lowest loss is part of our curve, and we can move in that direction.

The four losses are from (i) segmentation, (ii) smoothness, (iii) intensity and (iv) semantic parameters

### Segmentation loss

A helper function creates a probability distribution $\tilde{C}$ which labels each pixel with the probability it is a curve. We call the ground truth distribution $C$ i.e. if $C=1$ then a given pixel is on a curve, and if $C=0$ then that pixel is not on a curve. So, the probability a pixel is indeed on a curve given $\tilde{C}$, $P(C|\tilde{C})$, is $P(C|\tilde{C})=\tilde{C}^C (1-\tilde{C})^{1-C}$.

Defining $P(C|\tilde{C})$ is mathematical shorthand: if $C=1$, then $P(C|\tilde{C})=\tilde{C}$ and if $C=0$, $P(C|\tilde{C})=1-\tilde{C}$. This loss function is more convenient if we take its log (like maximum likelihood estimation !!). 

Hence, $\mathcal{L}^{\text{segmentation}}=C\log(\tilde{C})+(1-C)\log(1-\tilde{C})$

### Intensity
The pixel intensity (for us this means brightness) should be approximately same along a curve. Hence, we define the loss function as the difference in brightness between a pixel and its neighbour.

$I_p(x,y)$ is pixel brightness at point $(x,y)$ and $W_p$ is the width of the image. To track the curve, we are moving left to right across the width of the image. This is why we care about the image width in the loss functions below. 

$$\mathcal{L}^{\text{intensity}} = \sum_{i=0}^{W^p-1} \|I^p(x_{i+1}, y_{i+1})-I^p(x_i,y_i)\|^2  $$

### Smoothness
As we move left to right, changes in pixel brightness should be smooth (i.e. the curve is continuous). So, $I^p(x, y) = I^p(x+\mathrm{d}x, y + \mathrm{d}y)$. Next, we consider the first order Taylor series expansion around $I^p(x+\mathrm{d}x, y + \mathrm{d}y)$.

$$
\begin{align}
    I^p(x+\mathrm{d}x, y+\mathrm{d}y) &\approx I^p(x, y) + I_y^p \mathrm{d} y + I_x^p \mathrm{d} x \\
    0 &= I_y^p \mathrm{d} y + I_x^p \mathrm{d} x \\
    \frac{\mathrm{d}y}{\mathrm{d}x} &= - \frac{I_y^p}{I_x^p} \\
    V(x,y) &\coloneqq - \frac{I_y^p}{I_x^p}
\end{align}
$$

Here, we define $V(x,y)$ as how we want the slope of pixel brightness to change. The loss function is therefore the difference between how we want it to change and how it is measured to change. Conveniently, we take step sizes of 1 so $\Delta x = 0$ i.e. $\Delta y / \Delta x = \Delta y$. Hence:

$$\mathcal{L}^{\text{smoothness}} = \sum_{i=0}^{W^p-1} \| y_{i+1}-y_{i} - V(x_i, y_i)\|^2  $$


### Semantic parameters
We want to land on a pixel that is a plot line, so we want to minimize $P(\neg \tilde{C})$. Hence:

$$\mathcal{L}^{\text{semantic}} = \sum_{i=0}^{W^p-1} \| 1-\tilde{C}\|^2  $$


A helper function creates a probability distribution $\tilde{C}$ which labels each pixel with the probability it is a curve. 

This project is intened to help automate efforts in paper replication for integrated silicon photonic devices. It was written with help from Claude. Claude handled the busy work (creating files, docstrings, creating pytests). I implemented the plot extraction algorithm, so there may be errors there. 