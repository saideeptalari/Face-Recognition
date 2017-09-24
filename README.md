# Face-Recognition
A complete face recognition pipeline using dlib pre-trained models

This repository supports the blog post at [hackEvolve](http://www.hackevolve.com/face-recognition-deep-learning/)

#### Usage:
To enroll people to system,

`$ python enroll.py --dataset <path to dataset>`

##### Dataset structure
```bash
person_1/
    img1.jpg
    img2.jpg
    ........
person_2/
    image1.png
    image2.png
    ..........
..........
```

To perform recognition,

`$ python recognize.py --image <path to image>`
