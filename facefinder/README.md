# FaceFinder: OpenCV-based facial recognition in Python

## Background

FaceFinder is a set of Python Scripts which detect and identify faces in either still images or a video stream.  The code is based on the examples in the OpenCV documentation with some minor adjustments to make it a little more practical.

There are three (3) scripts in this package:

### recogniser.py

Recogniser builds the training datasets, labels and subject lists.  This is a time-consuming process. so it is broken out from the idenification utilities.  The end result is a set of data files (written with pickle) which contain the results of the identification and labelling processes.

Each person has training images stored in a directory underneath 'training-data'.  These directories must start with the letter 's' and the remainder of the name is decimal numerals.  For example s00000001, s00000002.

My experience is that the recognition engine will work when trained on a single image, but somewhere from 5 - 10 seems to make it far more accurate.

At present, the 'subjects' set is composed using in-line code.  This is poor practice, and I will break it out at some point soon.  But for now it works.  I am considering either having a 'subject.txt' file in each training directory, or just a master text list.  I am leaning towards the individual files, because it would address problems of sort order variations across different platforms.

### searcher.py

This is very much an example of how searching can be done, and is not really complete yet.  It loads the data created by Recogniser and then searches through the 'test-data' directory and attempts to identify what it finds there.

### videosearcher.py

This is where the real magic happens.  Videosearcher can use a supported video stream input or read from a compatible video file.  It also uses the data sets created by Recogniser.  Exactly what streams and formats it will support depend on the options compiled into your OpenCV runtime.

The performance is not brilliant yet, but it works reliably.  With a well-trained dataset you can expect 15-20fps on a machine without CUDA support.  I will providing timings for accelerated machines soon.

## Requirements

### Python 3

I have only tested this on Python 3.6.  It might work on version 2, but you are likely to need to make changes to the code.

### OpenCV

I build my own OpenCV from the release sets on GitHub.  Currently I am targeting the 3.4.x release

### NumPy

We need this for the image buffering in OpenCV

### Pickle

Used to write and read the trained datasets.

## Considerations

This is a bit of a pig.  With approximately 200 sets of faces (10 - 20 faces per set), Recogniser needs about 1.5GB of memory to process.  The data files are close to 40MB.  On an Intel NUC i5/16GB RAM, it took about 11 minutes to generate the datasets, and the CPU cores were all at 100%. Running the same code on an ODroid XU4 (4-Core ARMv7) took closer to 20 minutes, so while I would recommend against training on the ARM, you can certainly import the data sets, and that works well.

The indexing for label is very primative at the moment.  Directories must be numerically sequential.  Do not leave gaps in the numbers or you will likely have an index out of range condition.
