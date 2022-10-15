# ray-march-fractals-cuda
Ray marcher implementation to render fractals in c++ with CUDA. This program is currently in development
and only supports the Mandelbox and primitives.

![Demo Image](out.png)

## Attribution and Dependencies
The program uses the stb_image_write header file (from https://github.com/nothings/stb) 
to save renders as images. You must download the file and place it in the `main` directory
to run the program.

The Mandelbox distance estimator algorithm was translated and modified from the [Syntopia blog](http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/).