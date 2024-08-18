# myDentalAI
'myDentalAI' is a project for conrstructing a big data collection platform for dental treatment.

# Dataset
## Code
This section provides the code for leave-one-out cross-validation using two types of classifiers.

### “rocket-chair.py”, “rocket+chair.py"
- These scripts evaluate classifiers using sktime RocketClassifier.
- Please input the kernel size as an argument.
- rocket-chair.py uses only the time-series data of instruments for training, while rocket+chair.py uses both the instrument and chair time-series data for training.
- Validation results are saved as text files.

### “informer-chair.py”, “informer+chair.py”:
- These scripts evaluate classifiers based on the Informer model.
- Please input the number of epochs as an argument.
- informer-chair.py uses only the time-series data of instruments for training, while informer+chair.py uses both the instrument and chair time-series data for training.
- Validation results are saved as text files.

## Data Directory
The “data” directory includes time-series data of **273 dental treatments** and the corresponding treatment data.

### “cat4all-chair.npy” [273 cases × 1313 length × 24 types]
- This time-series data contains the detected characteristic parts of dental instruments captured in tray footage during 273 dental treatments, recognized using image recognition (https://doi.org/10.1038/s41598-022-26372-y).
- The data consists of the number of detections for **24 types of dental objects** (_Canal_syringe_blue, Canal_syringe_white, Clamp, Clamp_forceps, Composite_instrument, Condenser, Condenser_disk, Condenser_round, Dental_mirror, Dish, Excavator, Excavator_spoon, Explorer, Finger_ruler, Probe, Reamer, Reamer_guard, Syringe, Tweezers, Articulating_paper_holder, Spreader, Plugger, Hand, Cotton_) every 5 seconds.
- For shorter treatment durations, the remaining sequence is zero-padded.

### “cat4all+chair.npy” [273 cases × 1313 length × 31 types]
- This dataset combines the instrument detection results with **the chair log data of 7 types** (_Air turbine, Micromotor, Ultrasonic scaler, 3-way syringe, Intraoral vacuum, Electric root canal measurement device, Rinse cup_). The log data is represented as “1” if detected during the 5-second interval and “0” if not detected.
- For shorter treatment durations, the remaining sequence is zero-padded.

### “cat4allans.npy” [273 cases × 4 items]
- This summary dataset includes the dental insurance claim items entered for the 273 treatments.
- It indicates the presence (1) or absence (0) of **4 items** (_Periodontal Treatment, Caries Filling, Root Canal Treatment, Root Canal Filling_).

Typical video examples mimicking the flow of 4 types of treatments can be obtained from the following link:
https://drive.google.com/drive/folders/1vSPMth1i4dHXyKpvCJ9aL6dat31_wobP?usp=drive_link  
These videos are not actual treatment videos. They were filmed with the consent of the participants acting as dentists and patients.

## Dataset Availability / Maintenance
Upon successful completion of the anonymized review process, we will host and manage the dataset and code on GitHub.

## Code License
The code for validation is licensed under the MIT license:  
Copyright 2024  
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Dataset License
The project is CC BY-NC-ND 4.0 (allowing only non-commercial and no-derivative use).
