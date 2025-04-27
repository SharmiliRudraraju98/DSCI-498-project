We have used our own data of blurred or old images and uploaded them to the inference models to enhance them.

If you don't have your own images to test this on:

Note 1:  
You can find an image on the web or download sample images from the RealSRSet (proposed in BSRGAN, ICCV 2021) at:  
https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/RealSRSet+5images.zip

Note 2:  
You may need to use Google Chrome browser to enable file uploading when using the Streamlit app.

Note 3:  
If you encounter an out-of-memory error during enhancement, set `test_patch_wise = True` in your model inference settings to split the image into smaller patches.
