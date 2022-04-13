# OCR-Dotted-Matrix
OCR to detect and recognize dot-matrix text written with inkjet-printed on medical PVC bag


## TEXT DETECTION wiht CRAFT (Character-Region Awareness For Text detection)
https://github.com/clovaai/CRAFT-pytorch/blob/master/README.md#craft-character-region-awareness-for-text-detection

### Test instruction using pretrained model
- Download the trained models
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)


## TEXT RECOGNITION with TESSERACT 
https://tesseract-ocr.github.io/

