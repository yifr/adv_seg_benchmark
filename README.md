# adv_seg_benchmark
## Adversarial Segmentation Benchmarking


Image segmentation is an important task in computer vision, but models are still far from human-level segmentation abilities.
Most segmentation metrics assess average case performance for models, comparing things like mean IOU, or AP. While those metrics
can be used to train models or assess abilities, they can also obscure some important features in the kinds of errors
that these models make with respect to humans. This repo contains code for sampling "worst case" model errors in a way that highlights 
differences versus people, and other models.