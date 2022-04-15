# OCR-Dotted-Matrix
OCR to detect and recognize dot-matrix text written with inkjet-printed on medical PVC bag



Images example:
<p float="left">
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/A_0.png" width=30% height=30%>
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/B_9.png" width=30% height=30%>
</p>


## TEXT DETECTION wiht CRAFT (Character-Region Awareness For Text detection)
https://github.com/clovaai/CRAFT-pytorch/blob/master/README.md#craft-character-region-awareness-for-text-detection


Images example:
<p float="left">
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/result_CRAFT/res_preprocessed.jpg" width=30% height=30%>
 <img src="https://github.com/LeoPits/OCR-Dotted-Matrix/blob/main/Image_readme/result_CRAFT/res_preprocessed_mask.jpg" width=50% height=50%>
</p>





## TEXT RECOGNITION with TESSERACT 
https://tesseract-ocr.github.io/

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




* Run with pretrained model
``` (with python 3.7)
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```



The result image and socre maps will be saved to `./result` by default.

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


