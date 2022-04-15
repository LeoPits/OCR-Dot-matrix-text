# OCR-Dotted-Matrix
OCR to detect and recognize dot-matrix text written with inkjet-printed on medical PVC bag

* **Run script**
```
python test_Image.py  --image [folder path to test images]  --folder_res [folder path to folder result images] --label [string label to check]
```


### Arguments

* `--image`: test image
* `--label`: label
* `--folder_res`: folder result


* `--trained_model`: pretrained model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--cuda`: use cuda for inference (default:True)
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--test_folder`: folder path to input images
* `--refine`: use link refiner for sentense-level dataset
* `--refiner_model`: pretrained refiner model



Images example:
<p float="left">
 
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/A_0.png" width=30% height=30%>
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/B_9.png" width=30% height=30%>
</p>


## TEXT DETECTION wiht CRAFT (Character-Region Awareness For Text detection)

The code preprocessed the images with OpenCV function for enhanced the text detection with CRAFT with (https://github.com/clovaai/CRAFT-pytorch/blob/master/README.md#craft-character-region-awareness-for-text-detection). 
The weights of pre-train network are available on this link https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view.
The recognize label  is a string of the text, so the CRAFT parameters are set to find a unique block of text. it is possible to change `--text_threshold`,`--low_text` ,`--link_threshold` to have different detection results, but it is necessary to modify the label and recognition method after.


Craft result:
<p float="left">
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/result_CRAFT/res_preprocessed.jpg" width=30% height=30% >
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/result_CRAFT/res_preprocessed_mask.jpg" width=50% height=50%>
</p>



## TEXT RECOGNITION with TESSERACT 

The code extract the area around text on original image and fix the text oriention.

The cropped image example:

<img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/Preprocess/_original_0.jpg" width=30% height=30% >

Morphology Transformations (OpenCV function)  and rescaling of chars with different parameters are applied to the cropped image.

Pre-process cropped image example:

<p float="left">
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/Preprocess/_preprocess_00.jpg" width=30% height=30%>
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/Preprocess/_preprocess_140.jpg" width=30% height=30%>
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/Preprocess/_preprocess_90.jpg" width=30% height=30%>

</p>


I use  Tesseract OCR engine (https://tesseract-ocr.github.io/) with default  page segmentation , the experiments show the LCDDot_FT_500.traineddata performs the best results in this case.
Two methods are used to control the label:
* **SequenceMatcher** is a class available in python module named *difflib*. It can be used for comparing pairs of input sequences. With the function *ratio( )* returns the similarity score ( float in [0,1] ) between input strings. It sums the sizes of all matched sequences returned by function.
* **Regular expression** is a class available in python module named *re*.  The function *re.match()* checks for a match only at the beginning of the string.

Saving all result in json file:
```
        {
            "Name_original_file": "A_0.png",
            "Name_preprocess": "_preprocess_150.jpg",
            "check_label": "LOTTO:L21X45SCAD.:10-2023",
            "tesseract_LCDDot_FT_500_psm3_result": "LOTTO:L21X45SCAD.:10-2023",
            "LCDDot_FT_500_psm3_sequence_matcher_ratio_result": 1.0,
            "LCDDot_FT_500_psm3_bool_re_result": true
        }
    ],
    [
        {
            "Name_original_file": "A_0.png",
            "Name_preprocess": "_preprocess_160.jpg",
            "check_label": "LOTTO:L21X45SCAD.:10-2023",
            "tesseract_LCDDot_FT_500_psm3_result": "LOTTO:L21X4SCAD.:1625555",
            "LCDDot_FT_500_psm3_sequence_matcher_ratio_result": 0.78,
            "LCDDot_FT_500_psm3_bool_re_result": false
        }
    ],
    [
        {
            "Name_original_file": "A_0.png",
            "Name_preprocess": "_preprocess_170.jpg",
            "check_label": "LOTTO:L21X45SCAD.:10-2023",
            "tesseract_LCDDot_FT_500_psm3_result": "LOTTO:L21X45SCAD.:10-2023",
            "LCDDot_FT_500_psm3_sequence_matcher_ratio_result": 1.0,
            "LCDDot_FT_500_psm3_bool_re_result": true
        }
```



## Getting started
### Install dependencies
#### Requirements
- PyTorch>=1.9.0
- torchvision>=0.2.2
- opencv-python>=4.5.2
```
conda env create -f environment.yml
```




